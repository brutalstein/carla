from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Mapping
from typing import Any

from l4stack.perception.backend_contracts import (
    BackendHealth,
    BackendUnavailable,
    ProcessBackendConfig,
)
from l4stack.perception.protocol import BackendProtocolError, InferenceRequest, InferenceResponse


class JsonlProcessBackend:
    """Model ortamını ana stack'ten ayıran tek-istek-sıralı JSONL backend'i."""

    def __init__(self, config: ProcessBackendConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._stderr: list[str] = []
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            command_ok, detail = self._config.validate_command()
            if not command_ok:
                raise BackendUnavailable(detail)
            self._drain_response_queue()
            self._stderr.clear()
            environment = os.environ.copy()
            environment.update(dict(self._config.environment))
            self._process = subprocess.Popen(
                list(self._config.command),
                cwd=(
                    None
                    if self._config.working_directory is None
                    else str(self._config.working_directory)
                ),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._reader = threading.Thread(
                target=self._read_stdout,
                name=f"perception-backend-stdout-{self._process.pid}",
                daemon=True,
            )
            self._stderr_reader = threading.Thread(
                target=self._read_stderr,
                name=f"perception-backend-stderr-{self._process.pid}",
                daemon=True,
            )
            self._reader.start()
            self._stderr_reader.start()

        try:
            self._send({"protocol_version": 1, "type": "ping"})
            response = self._wait_response(self._config.startup_timeout_s)
            if response.get("type") != "ready" or int(response.get("protocol_version", -1)) != 1:
                raise BackendUnavailable(f"Backend readiness handshake failed: {response}")
        except Exception:
            self.stop()
            raise

    def infer(self, request: InferenceRequest, timeout_s: float) -> Mapping[str, Any]:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.start()
        with self._lock:
            self._send(request.as_dict())
            raw = self._wait_response(timeout_s)
        response = InferenceResponse.from_dict(raw)
        if response.request_id != request.request_id:
            raise BackendProtocolError(
                f"Backend response id mismatch: expected={request.request_id} "
                f"actual={response.request_id}"
            )
        if not response.ok:
            raise BackendUnavailable(response.error or "Model inference failed")
        return response.payload

    def health(self) -> BackendHealth:
        process = self._process
        if process is None:
            return BackendHealth(False, "not started")
        code = process.poll()
        if code is not None:
            tail = " | ".join(self._stderr[-5:])
            return BackendHealth(False, f"exited with code {code}: {tail}", process.pid)
        return BackendHealth(True, "ready", process.pid)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(
                        json.dumps({"protocol_version": 1, "type": "shutdown"}) + "\n"
                    )
                    process.stdin.flush()
                process.wait(timeout=self._config.shutdown_timeout_s)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _send(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise BackendUnavailable("Backend process is not running")
        process.stdin.write(json.dumps(dict(payload), separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _wait_response(self, timeout_s: float) -> dict[str, Any]:
        try:
            item = self._responses.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise BackendUnavailable(
                f"Backend response timeout after {timeout_s:.3f}s; "
                f"health={self.health().detail}"
            ) from exc
        if isinstance(item, BaseException):
            raise BackendUnavailable(str(item)) from item
        return item

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("JSONL response root must be an object")
                except Exception as exc:
                    self._responses.put(BackendProtocolError(f"Invalid backend JSON: {exc}"))
                    continue
                self._responses.put(value)
        finally:
            if process.poll() is not None:
                self._responses.put(
                    BackendUnavailable(f"Backend process exited: {process.returncode}")
                )

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line.rstrip())
            if len(self._stderr) > 100:
                del self._stderr[:50]

    def _drain_response_queue(self) -> None:
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                return
