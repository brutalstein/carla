from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from l4stack.perception.protocol import PROTOCOL_VERSION
from l4stack.perception.types import PerceptionInput


class JsonlBackendServer:
    """Model container'larının kullandığı persistent CUDA worker protokolü.

    ``release_handler`` worker-owned raster/output ring slotlarını host snapshot yaşam
    süresi bittiğinde geri alır. Raster üreten worker bu callback'i sağlamalıdır.
    """

    def __init__(
        self,
        handler: Callable[[PerceptionInput], Mapping[str, Any]],
        *,
        ready_metadata: Mapping[str, Any],
        release_handler: Callable[[tuple[str, ...]], None] | None = None,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        required = {
            "device",
            "cuda_available",
            "cpu_fallback",
            "model_loaded",
            "compute_capability",
            "precision",
        }
        missing = required - ready_metadata.keys()
        if missing:
            raise ValueError(f"ready_metadata eksik alanlar içeriyor: {sorted(missing)}")
        self._handler = handler
        self._release_handler = release_handler
        self._ready = dict(ready_metadata)
        self._input = input_stream
        self._output = output_stream

    def serve_forever(self) -> None:
        for line in self._input:
            value: dict[str, Any] = {}
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                message_type = value.get("type")
                if int(value.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol version")
                if message_type == "ping":
                    self._write(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "type": "ready",
                            **self._ready,
                        }
                    )
                    continue
                if message_type == "shutdown":
                    return
                if message_type == "release":
                    self._release(value)
                    continue
                if message_type != "infer":
                    raise ValueError(f"unsupported message type: {message_type}")
                request_id = str(value["request_id"])
                payload = PerceptionInput.from_dict(value["payload"])
                result = dict(self._handler(payload))
                self._write(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "type": "result",
                        "request_id": request_id,
                        "ok": True,
                        "payload": result,
                    }
                )
            except Exception as exc:
                request_id = str(value.get("request_id", ""))
                self._write(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "type": "result",
                        "request_id": request_id or "unknown",
                        "ok": False,
                        "payload": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    def _release(self, value: Mapping[str, Any]) -> None:
        request_id = str(value.get("request_id", ""))
        raw_artifacts = value.get("artifacts", [])
        if not request_id or not isinstance(raw_artifacts, list):
            raise ValueError("invalid release request")
        artifacts = tuple(str(item) for item in raw_artifacts)
        if self._release_handler is None and artifacts:
            raise RuntimeError("worker does not implement output artifact release")
        if self._release_handler is not None:
            self._release_handler(artifacts)
        self._write(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "released",
                "request_id": request_id,
            }
        )

    def _write(self, value: Mapping[str, Any]) -> None:
        self._output.write(json.dumps(dict(value), separators=(",", ":")) + "\n")
        self._output.flush()
