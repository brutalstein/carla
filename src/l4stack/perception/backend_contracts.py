from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from l4stack.perception.protocol import InferenceRequest


class BackendUnavailable(RuntimeError):
    """Model backend'i başlatılamadığında veya yanıt vermediğinde üretilir."""


@dataclass(frozen=True, slots=True)
class BackendHealth:
    ready: bool
    detail: str
    pid: int | None = None


class ModelBackend(Protocol):
    def start(self) -> None: ...

    def infer(self, request: InferenceRequest, timeout_s: float) -> Mapping[str, Any]: ...

    def health(self) -> BackendHealth: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProcessBackendConfig:
    command: tuple[str, ...]
    working_directory: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    required_environment: tuple[str, ...] = ()
    startup_timeout_s: float = 60.0
    shutdown_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("Process backend command cannot be empty")
        if self.startup_timeout_s <= 0.0 or self.shutdown_timeout_s <= 0.0:
            raise ValueError("Process backend timeouts must be positive")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        if len(self.required_environment) != len(set(self.required_environment)):
            raise ValueError("required_environment contains duplicates")

    def validate_command(self) -> tuple[bool, str]:
        executable = self.command[0]
        if os.path.sep in executable:
            path = Path(executable)
            if self.working_directory is not None and not path.is_absolute():
                path = self.working_directory / path
            if not path.exists():
                return False, f"executable does not exist: {path}"
        elif shutil.which(executable) is None:
            return False, f"executable is not on PATH: {executable}"
        if self.working_directory is not None and not self.working_directory.is_dir():
            return False, f"working directory does not exist: {self.working_directory}"

        for argument in self.command[1:]:
            if argument.startswith("-"):
                continue
            candidate = Path(argument)
            looks_like_path = candidate.suffix in {".py", ".sh"} or argument.startswith(
                ("./", "../", os.path.sep)
            )
            if not looks_like_path:
                continue
            if self.working_directory is not None and not candidate.is_absolute():
                candidate = self.working_directory / candidate
            if not candidate.exists():
                return False, f"backend command path does not exist: {candidate}"

        available_environment = {**os.environ, **dict(self.environment)}
        missing = [
            name for name in self.required_environment if not available_environment.get(name)
        ]
        if missing and available_environment.get("L4STACK_PERCEPTION_MOCK") != "1":
            return False, f"required environment variables are missing: {missing}"
        return True, "command and required environment are resolvable"
