from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


class LifecycleState(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"
    FINALIZED = "FINALIZED"


class LifecycleTransition(str, Enum):
    CONFIGURE = "CONFIGURE"
    ACTIVATE = "ACTIVATE"
    DEACTIVATE = "DEACTIVATE"
    CLEANUP = "CLEANUP"
    SHUTDOWN = "SHUTDOWN"
    ERROR = "ERROR"


class LifecycleError(RuntimeError):
    """Geçersiz lifecycle geçişi veya hook hatası."""


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    component: str
    transition: LifecycleTransition
    from_state: LifecycleState
    to_state: LifecycleState
    success: bool
    detail: str = ""


class ManagedComponent:
    """Harici supervisor tarafından yönetilen standart bileşen tabanı.

    Alt sınıflar yalnızca ``on_*`` hook'larını uygular. State geçişleri ve hata
    izolasyonu merkezi biçimde burada tutulur; böylece her katman farklı lifecycle
    kuralları üretmez.
    """

    def __init__(self, name: str, dependencies: tuple[str, ...] = ()) -> None:
        self.name = name
        self.dependencies = tuple(dependencies)
        self._state = LifecycleState.UNCONFIGURED
        self._lock = threading.RLock()
        self._history: list[TransitionRecord] = []

    @property
    def state(self) -> LifecycleState:
        with self._lock:
            return self._state

    def transition_history(self) -> tuple[TransitionRecord, ...]:
        with self._lock:
            return tuple(self._history)

    def configure(self) -> None:
        self._run_transition(
            LifecycleTransition.CONFIGURE,
            expected=LifecycleState.UNCONFIGURED,
            target=LifecycleState.INACTIVE,
            hook=self.on_configure,
        )

    def activate(self) -> None:
        self._run_transition(
            LifecycleTransition.ACTIVATE,
            expected=LifecycleState.INACTIVE,
            target=LifecycleState.ACTIVE,
            hook=self.on_activate,
        )

    def deactivate(self) -> None:
        self._run_transition(
            LifecycleTransition.DEACTIVATE,
            expected=LifecycleState.ACTIVE,
            target=LifecycleState.INACTIVE,
            hook=self.on_deactivate,
        )

    def cleanup(self) -> None:
        self._run_transition(
            LifecycleTransition.CLEANUP,
            expected=LifecycleState.INACTIVE,
            target=LifecycleState.UNCONFIGURED,
            hook=self.on_cleanup,
        )

    def shutdown(self) -> None:
        with self._lock:
            if self._state is LifecycleState.FINALIZED:
                return
            from_state = self._state
            try:
                self.on_shutdown()
            except Exception as exc:
                self._state = LifecycleState.ERROR
                self._history.append(
                    TransitionRecord(
                        self.name,
                        LifecycleTransition.SHUTDOWN,
                        from_state,
                        LifecycleState.ERROR,
                        False,
                        str(exc),
                    )
                )
                raise LifecycleError(f"{self.name} shutdown failed: {exc}") from exc
            self._state = LifecycleState.FINALIZED
            self._history.append(
                TransitionRecord(
                    self.name,
                    LifecycleTransition.SHUTDOWN,
                    from_state,
                    LifecycleState.FINALIZED,
                    True,
                )
            )

    def fail(self, detail: str) -> None:
        with self._lock:
            from_state = self._state
            self._state = LifecycleState.ERROR
            self._history.append(
                TransitionRecord(
                    self.name,
                    LifecycleTransition.ERROR,
                    from_state,
                    LifecycleState.ERROR,
                    False,
                    detail,
                )
            )
            self.on_error(detail)

    def require_active(self) -> None:
        if self.state is not LifecycleState.ACTIVE:
            raise LifecycleError(f"Component is not active: {self.name} ({self.state.value})")

    def _run_transition(
        self,
        transition: LifecycleTransition,
        *,
        expected: LifecycleState,
        target: LifecycleState,
        hook,
    ) -> None:
        with self._lock:
            if self._state is not expected:
                raise LifecycleError(
                    f"Invalid {transition.value} transition for {self.name}: "
                    f"expected={expected.value} actual={self._state.value}"
                )
            from_state = self._state
            try:
                hook()
            except Exception as exc:
                self._state = LifecycleState.ERROR
                self._history.append(
                    TransitionRecord(
                        self.name,
                        transition,
                        from_state,
                        LifecycleState.ERROR,
                        False,
                        str(exc),
                    )
                )
                raise LifecycleError(f"{self.name} {transition.value} failed: {exc}") from exc
            self._state = target
            self._history.append(
                TransitionRecord(self.name, transition, from_state, target, True)
            )

    # Aşağıdaki hook'lar bilinçli olarak no-op'tur; kaynak tahsisi gereken katmanlar override eder.
    def on_configure(self) -> None:
        pass

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_cleanup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass

    def on_error(self, detail: str) -> None:
        pass
