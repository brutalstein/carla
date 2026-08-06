from __future__ import annotations

from typing import Any

from l4stack.errors import CarlaConnectionError


def import_carla() -> Any:
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise CarlaConnectionError(
            "CARLA Python API import edilemedi. CARLA sürümünüzle eşleşen 'carla' "
            "paketini/egg dosyasını Python ortamına ekleyin."
        ) from exc
    return carla


def connect(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    carla = import_carla()
    connection = config["connection"]
    try:
        client = carla.Client(str(connection["host"]), int(connection["port"]))
        client.set_timeout(float(connection.get("timeout_seconds", 10.0)))
        world = client.get_world()
    except Exception as exc:
        raise CarlaConnectionError(f"CARLA sunucusuna bağlanılamadı: {exc}") from exc
    return carla, client, world
