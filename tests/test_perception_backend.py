# ruff: noqa: F403, F405
from tests.perception_helpers import *  # noqa: F403


def test_manifest_checks_size_hash_and_command(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"correct")
    runner = tmp_path / "runner.py"
    runner.write_text("print('x')", encoding="utf-8")
    digest = hashlib.sha256(b"correct").hexdigest()
    config = PerceptionConfig.from_mapping(
        tmp_path,
        {
            "perception": {
                "models": {
                    "city": {
                        "enabled": True,
                        "adapter": "citysemsegformer",
                        "model_version": "test",
                        "executor": "perception_city",
                        "runtime_component": "perception_citysemsegformer",
                        "target_rate_hz": 5,
                        "request_timeout_s": 1,
                        "cameras": ["camera_front"],
                        "backend": {"command": [sys.executable, str(runner)]},
                        "artifacts": [
                            {
                                "path": "model.onnx",
                                "min_size_bytes": 7,
                                "sha256": digest,
                            }
                        ],
                    }
                }
            }
        },
    )
    report = verify_installation(config)[0]
    assert report.ready
    model.write_bytes(b"bad")
    assert not verify_installation(config)[0].ready


def test_jsonl_process_backend_round_trip(tmp_path: Path) -> None:
    script = tmp_path / "backend.py"
    script.write_text(
        """
import json, sys
for line in sys.stdin:
    value=json.loads(line)
    if value['type']=='ping':
        print(json.dumps({'protocol_version':1,'type':'ready'}), flush=True)
    elif value['type']=='shutdown':
        break
    elif value['type']=='infer':
        print(json.dumps({
            'protocol_version': 1,
            'type': 'result',
            'request_id': value['request_id'],
            'ok': True,
            'payload': {'detections_3d': []},
        }), flush=True)
""",
        encoding="utf-8",
    )
    backend = JsonlProcessBackend(ProcessBackendConfig((sys.executable, str(script))))
    backend.start()
    adapter = BevFusionDetectionAdapter(
        adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
    )
    request = adapter.build_request(input_value(), "request-1")
    response = backend.infer(request, 1.0)
    assert response == {"detections_3d": []}
    assert backend.health().ready
    backend.stop()


def test_factory_builds_only_enabled_models_and_isolated_start(tmp_path: Path) -> None:
    config = PerceptionConfig.from_mapping(
        tmp_path,
        {
            "perception": {
                "enabled": True,
                "models": {
                    "det": {
                        "enabled": True,
                        "adapter": "bevfusion_detection",
                        "model_version": "test",
                        "executor": "perception_det",
                        "runtime_component": "perception_det",
                        "target_rate_hz": 10,
                        "request_timeout_s": 0.2,
                        "cameras": list(CAMERAS),
                        "lidar": "lidar_top",
                        "backend": {"command": [sys.executable, "unused.py"]},
                    },
                    "map": {
                        "enabled": False,
                        "adapter": "maptrv2",
                        "model_version": "test",
                        "executor": "perception_map",
                        "runtime_component": "perception_map",
                        "target_rate_hz": 5,
                        "request_timeout_s": 0.2,
                        "cameras": list(CAMERAS),
                        "backend": {"command": [sys.executable, "unused.py"]},
                    },
                },
            }
        },
    )

    class RuntimeConfig:
        namespace = "test"

        def contract(self, name):
            return contract(name)

    clock = ManualClock(1.0)
    registry = ExecutorRegistry(
        {
            "perception_det": ExecutorProfile("perception_det", 1, 2, 50),
            "perception_map": ExecutorProfile("perception_map", 1, 2, 55),
        },
        clock,
    )
    built = build_perception_runtime(
        perception=config,
        runtime_config=RuntimeConfig(),
        runtime_context=runtime(clock),
        executors=registry,
        backend_factory=lambda _model: FakeBackend([detection_payload()]),
        verify_artifacts=False,
    )
    try:
        assert set(built.components) == {"det"}
        start_results = built.start()
        assert start_results[0].active
        assert built.components["det"].state.value == "ACTIVE"
        built.stop()
    finally:
        registry.shutdown_all(wait=True, cancel_pending=True)
