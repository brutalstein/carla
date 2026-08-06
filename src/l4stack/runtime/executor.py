from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from l4stack.runtime.clock import Clock, SteadyClock
from l4stack.runtime.contracts import ExecutorProfile


class ExecutorClosed(RuntimeError):
    """Kapatılmış executor'a görev gönderildiğinde üretilir."""


@dataclass(order=True)
class _QueuedTask:
    priority: int
    sequence: int
    name: str = field(compare=False)
    callback: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    kwargs: dict[str, Any] = field(compare=False)
    future: Future[Any] = field(compare=False)
    deadline_at: float | None = field(compare=False, default=None)
    drop_if_expired: bool = field(compare=False, default=True)
    stop: bool = field(compare=False, default=False)


class PriorityExecutor:
    """Öncelik sıralı, bounded ve blocking worker executor.

    Python interpreter hard real-time garanti vermez. Bu executor'ın amacı katmanları
    doğru sahiplik ve öncelik sınıflarıyla ayırmak, queue büyümesini sınırlamak ve
    worker'ların veri yokken uyumasını sağlamaktır.
    """

    def __init__(
        self,
        name: str,
        workers: int = 1,
        queue_capacity: int = 128,
        clock: Clock | None = None,
    ) -> None:
        if workers <= 0 or queue_capacity <= 0:
            raise ValueError("workers and queue_capacity must be positive")
        self.name = name
        self._workers_count = workers
        self._queue: queue.PriorityQueue[_QueuedTask] = queue.PriorityQueue(queue_capacity)
        self._clock = clock or SteadyClock()
        self._counter = itertools.count(1)
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self._started = False
        self._closed = False

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise ExecutorClosed(self.name)
            if self._started:
                return
            self._started = True
            for index in range(self._workers_count):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"{self.name}-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def submit(
        self,
        name: str,
        callback: Callable[..., Any],
        *args: Any,
        priority: int = 100,
        deadline_at: float | None = None,
        drop_if_expired: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[Any]:
        with self._lock:
            if self._closed:
                raise ExecutorClosed(self.name)
            if not self._started:
                self.start()
        future: Future[Any] = Future()
        task = _QueuedTask(
            priority=int(priority),
            sequence=next(self._counter),
            name=name,
            callback=callback,
            args=args,
            kwargs=kwargs,
            future=future,
            deadline_at=deadline_at,
            drop_if_expired=drop_if_expired,
        )
        try:
            self._queue.put(task, timeout=timeout)
        except queue.Full as exc:
            future.set_exception(TimeoutError(f"Executor queue is full: {self.name}"))
            raise TimeoutError(f"Executor queue is full: {self.name}") from exc
        return future

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if cancel_pending:
            while True:
                try:
                    task = self._queue.get_nowait()
                except queue.Empty:
                    break
                if task is not None and not task.future.done():
                    task.future.cancel()
                self._queue.task_done()
        # Her worker için en düşük öncelikli bir sentinel eklenir. PriorityQueue içinde
        # None kullanmak, gerçek görevlerle karşılaştırma sırasında TypeError üretebilir.
        for _ in self._threads:
            self._queue.put(
                _QueuedTask(
                    priority=10**9,
                    sequence=next(self._counter),
                    name="__shutdown__",
                    callback=lambda: None,
                    args=(),
                    kwargs={},
                    future=Future(),
                    stop=True,
                )
            )
        if wait:
            current = threading.current_thread()
            for thread in self._threads:
                if thread is not current:
                    thread.join()

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task.stop:
                    return
                if task.future.cancelled():
                    continue
                if (
                    task.deadline_at is not None
                    and task.drop_if_expired
                    and self._clock.now() > task.deadline_at
                ):
                    task.future.set_exception(
                        TimeoutError(f"Task expired before execution: {task.name}")
                    )
                    continue
                if not task.future.set_running_or_notify_cancel():
                    continue
                try:
                    result = task.callback(*task.args, **task.kwargs)
                except BaseException as exc:
                    task.future.set_exception(exc)
                else:
                    task.future.set_result(result)
            finally:
                self._queue.task_done()


@dataclass(frozen=True, slots=True)
class PeriodicTaskHandle:
    name: str
    period_s: float


@dataclass(order=True)
class _PeriodicTask:
    next_release: float
    sequence: int
    name: str = field(compare=False)
    period_s: float = field(compare=False)
    callback: Callable[[], Any] = field(compare=False)
    priority: int = field(compare=False)
    cancelled: bool = field(compare=False, default=False)


class PeriodicScheduler:
    """Fixed-rate işleri busy-spin olmadan PriorityExecutor'a bırakan scheduler."""

    def __init__(self, executor: PriorityExecutor, clock: Clock | None = None) -> None:
        self._executor = executor
        self._clock = clock or SteadyClock()
        self._condition = threading.Condition()
        self._tasks: list[_PeriodicTask] = []
        self._counter = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="periodic-scheduler",
                daemon=True,
            )
            self._thread.start()

    def add(
        self,
        name: str,
        period_s: float,
        callback: Callable[[], Any],
        *,
        priority: int = 100,
        start_immediately: bool = False,
    ) -> PeriodicTaskHandle:
        if period_s <= 0.0:
            raise ValueError("period_s must be positive")
        now = self._clock.now()
        task = _PeriodicTask(
            next_release=now if start_immediately else now + period_s,
            sequence=next(self._counter),
            name=name,
            period_s=period_s,
            callback=callback,
            priority=priority,
        )
        with self._condition:
            if self._closed:
                raise ExecutorClosed("periodic-scheduler")
            self._tasks.append(task)
            self._tasks.sort()
            self._condition.notify_all()
        self.start()
        return PeriodicTaskHandle(name, period_s)

    def cancel(self, handle: PeriodicTaskHandle) -> None:
        with self._condition:
            for task in self._tasks:
                if task.name == handle.name:
                    task.cancelled = True
            self._condition.notify_all()

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait and self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._tasks = [task for task in self._tasks if not task.cancelled]
                if self._closed:
                    return
                if not self._tasks:
                    self._condition.wait()
                    continue
                self._tasks.sort()
                task = self._tasks[0]
                delay = task.next_release - self._clock.now()
                if delay > 0.0:
                    self._condition.wait(delay)
                    continue
                self._tasks.pop(0)
                # Fixed-rate: önceki release zamanına period eklenir; drift biriktirilmez.
                task.next_release += task.period_s
                self._tasks.append(task)
            self._executor.submit(
                task.name,
                task.callback,
                priority=task.priority,
                deadline_at=task.next_release,
                drop_if_expired=False,
            )


class ExecutorRegistry:
    """Merkezi executor profillerini isimle yöneten ve topluca kapatan registry."""

    def __init__(
        self,
        profiles: Mapping[str, ExecutorProfile],
        clock: Clock | None = None,
    ) -> None:
        self._profiles: dict[str, ExecutorProfile] = dict(profiles)
        self._clock = clock or SteadyClock()
        self._executors: dict[str, PriorityExecutor] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> PriorityExecutor:
        with self._lock:
            existing = self._executors.get(name)
            if existing is not None:
                return existing
            try:
                profile = self._profiles[name]
            except KeyError as exc:
                raise KeyError(f"Executor profile is not configured: {name}") from exc
            executor = PriorityExecutor(
                name=f"executor-{name}",
                workers=profile.workers,
                queue_capacity=profile.queue_capacity,
                clock=self._clock,
            )
            executor.start()
            self._executors[name] = executor
            return executor

    def shutdown_all(self, wait: bool = True, cancel_pending: bool = False) -> None:
        with self._lock:
            executors = list(self._executors.values())
            self._executors.clear()
        for executor in reversed(executors):
            executor.shutdown(wait=wait, cancel_pending=cancel_pending)
