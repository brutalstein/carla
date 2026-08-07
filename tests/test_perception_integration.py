# ruff: noqa: F403, F405
from tests.perception_helpers import *  # noqa: F403


def test_artifact_store_writes_bgra_lidar_and_publishes_message(tmp_path: Path) -> None:
    class Camera:
        frame = 12
        timestamp = 1.2
        width = 2
        height = 1
        raw_data = bytes(range(8))

    class Lidar:
        frame = 12
        timestamp = 1.2
        raw_data = b"\x00" * 32

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    store = PerceptionArtifactStore(tmp_path / "artifacts", retention_frames=2)
    camera = store.write_camera("camera_front", Camera())
    assert store.write_camera("camera_front", Camera()) is camera
    lidar = store.write_lidar("lidar_top", Lidar())
    assert store.write_lidar("lidar_top", Lidar()) is lidar
    value = PerceptionInput(
        frame=12,
        timestamp=1.2,
        cameras=(camera,),
        lidar=lidar,
        calibration=store.calibration_ref(calibration),
        localization_message_id="loc/12",
    )
    clock = ManualClock(1.2)
    message = PerceptionInputPublisher(clock, "test", 0.25).publish(
        value, localization_message_id="loc/12"
    )
    assert Path(camera.uri.removeprefix("file://")).read_bytes() == Camera.raw_data
    assert lidar.shape == (2, 4)
    assert message.parents == ("loc/12",)


def test_pipeline_due_models_and_selected_submission() -> None:
    clock = ManualClock(1.0)
    context = runtime(clock)
    component = PerceptionModelComponent(
        adapter=BevFusionDetectionAdapter(
            adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
        ),
        backend=FakeBackend([detection_payload()]),
        runtime=context,
        contract=contract("perception_bevfusion_detection"),
        request_timeout_s=0.2,
        namespace="test",
    )
    component.configure()
    component.activate()
    executor = PriorityExecutor(
        name="test-due-models", workers=1, queue_capacity=2, clock=clock
    )
    pipeline = PerceptionPipeline(
        {
            "bevfusion": ModelRoute(
                component=component,
                executor=executor,
                target_rate_hz=5.0,
            )
        },
        snapshot_max_age_s=0.5,
    )
    try:
        assert pipeline.due_models(1.0) == ("bevfusion",)
        message = envelope(clock, input_value(timestamp=1.0))
        future = pipeline.submit(message, model_names=("bevfusion",))["bevfusion"]
        future.result(timeout=1.0)
        assert pipeline.due_models(1.1) == ()
        assert pipeline.due_models(1.2) == ("bevfusion",)
    finally:
        executor.shutdown(wait=True, cancel_pending=True)


def test_pipeline_disables_component_after_permanent_failure() -> None:
    clock = ManualClock(1.0)
    context = runtime(clock)
    component = PerceptionModelComponent(
        adapter=BevFusionDetectionAdapter(
            adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
        ),
        backend=FakeBackend([]),
        runtime=context,
        contract=contract("perception_bevfusion_detection"),
        request_timeout_s=0.2,
        namespace="test",
    )
    component.configure()
    component.activate()
    executor = PriorityExecutor(
        name="test-route-disable", workers=1, queue_capacity=2, clock=clock
    )
    pipeline = PerceptionPipeline(
        {
            "bevfusion": ModelRoute(
                component=component,
                executor=executor,
                target_rate_hz=5.0,
            )
        },
        snapshot_max_age_s=0.5,
    )
    try:
        future = pipeline.submit(envelope(clock, input_value()))["bevfusion"]
        with pytest.raises(Exception):
            future.result(timeout=1.0)
        assert pipeline.due_models(2.0) == ()
        assert pipeline.metrics().failed["bevfusion"] == 1
    finally:
        executor.shutdown(wait=True, cancel_pending=True)


def test_synchronizer_returns_latest_measurements_at_or_before_frame() -> None:
    class Measurement:
        def __init__(self, frame: int, timestamp: float) -> None:
            self.frame = frame
            self.timestamp = timestamp

    synchronizer = SensorSynchronizer()
    synchronizer.callback("camera_front")(Measurement(10, 0.5))
    synchronizer.callback("lidar_top")(Measurement(11, 0.55))
    result = synchronizer.wait_for_latest_at_or_before(
        11,
        ("camera_front", "lidar_top"),
        0.1,
    )
    assert result["camera_front"].frame == 10
    assert result["lidar_top"].frame == 11


@pytest.mark.parametrize(
    "model_name",
    [
        "bevfusion_detection",
        "bevfusion_segmentation",
        "maptrv2",
        "tld_ready",
        "citysemsegformer",
    ],
)
def test_repository_model_wrapper_mock_handshake(model_name: str) -> None:
    script = ROOT / "models" / "perception" / model_name / "run_backend.sh"
    environment = os.environ.copy()
    environment["L4STACK_PERCEPTION_MOCK"] = "1"
    process = subprocess.Popen(
        ["bash", str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps({"protocol_version": 1, "type": "ping"}) + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline()) == {
            "protocol_version": 1,
            "type": "ready",
        }
        payload = input_value().as_dict()
        process.stdin.write(
            json.dumps(
                {
                    "protocol_version": 1,
                    "type": "infer",
                    "request_id": f"test-{model_name}",
                    "model_name": model_name,
                    "source_timestamp": 1.0,
                    "payload": payload,
                }
            )
            + "\n"
        )
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["ok"] is True
        assert response["request_id"] == f"test-{model_name}"
        process.stdin.write(
            json.dumps({"protocol_version": 1, "type": "shutdown"}) + "\n"
        )
        process.stdin.flush()
        assert process.wait(timeout=3.0) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1.0)


def test_backend_handshake_failure_cleans_up_process(tmp_path: Path) -> None:
    script = tmp_path / "bad_backend.py"
    script.write_text(
        """
import json, sys
for line in sys.stdin:
    value = json.loads(line)
    if value['type'] == 'ping':
        print(json.dumps({'protocol_version': 1, 'type': 'not-ready'}), flush=True)
    elif value['type'] == 'shutdown':
        break
""",
        encoding="utf-8",
    )
    backend = JsonlProcessBackend(
        ProcessBackendConfig((sys.executable, str(script)), startup_timeout_s=1.0)
    )
    with pytest.raises(BackendUnavailable):
        backend.start()
    assert not backend.health().ready


def test_inference_response_requires_boolean_ok() -> None:
    with pytest.raises(BackendProtocolError):
        InferenceResponse.from_dict(
            {
                "protocol_version": 1,
                "type": "result",
                "request_id": "request-1",
                "ok": "false",
                "payload": {},
            }
        )
