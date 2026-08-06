from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
