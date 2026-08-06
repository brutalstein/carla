from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from typing import Any


def destroy_actors(actors: Iterable[Any]) -> None:
    for actor in reversed(list(actors)):
        with suppress(Exception):
            if actor is not None and getattr(actor, "is_alive", True):
                actor.stop() if hasattr(actor, "stop") else None
                actor.destroy()
