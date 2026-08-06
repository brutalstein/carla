from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from l4stack.perception.protocol import PROTOCOL_VERSION
from l4stack.perception.types import PerceptionInput


class JsonlBackendServer:
    """Model ortamlarında kullanılacak küçük ve ortak stdin/stdout sunucusu."""

    def __init__(
        self,
        handler: Callable[[PerceptionInput], Mapping[str, Any]],
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        self._handler = handler
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
                    self._write({"protocol_version": PROTOCOL_VERSION, "type": "ready"})
                    continue
                if message_type == "shutdown":
                    return
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
                request_id = ""
                if isinstance(value, dict):
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

    def _write(self, value: Mapping[str, Any]) -> None:
        self._output.write(json.dumps(dict(value), separators=(",", ":")) + "\n")
        self._output.flush()
