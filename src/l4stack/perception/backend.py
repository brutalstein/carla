"""Perception backend public API.

Üretim paketinde sentetik/fake backend bulunmaz. Unit testler gerekiyorsa test dizini
kendi protocol fixture process'ini başlatır; runtime her zaman gerçek worker kullanır.
"""

from l4stack.perception.backend_contracts import (
    BackendHealth,
    BackendUnavailable,
    ModelBackend,
    ProcessBackendConfig,
)
from l4stack.perception.backend_process import JsonlProcessBackend

__all__ = [
    "BackendHealth",
    "BackendUnavailable",
    "JsonlProcessBackend",
    "ModelBackend",
    "ProcessBackendConfig",
]
