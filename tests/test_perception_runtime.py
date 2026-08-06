# ruff: noqa: F403, F405
from tests.perception_helpers import *  # noqa: F403


def test_component_publishes_health_snapshot_and_lineage() -> None:
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
    output = component.process(envelope(clock, input_value()))
    assert output.parents
    assert component.output_snapshot.require().value == output
    assert context.health.get(component.name).state is RuntimeHealth.NOMINAL
    assert context.lineage.get(output.message_id) is not None


def test_component_rejects_stale_input() -> None:
    clock = ManualClock(2.0)
    context = runtime(clock)
    component = PerceptionModelComponent(
        adapter=BevFusionDetectionAdapter(
            adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
        ),
        backend=FakeBackend([detection_payload()]),
        runtime=context,
        contract=contract("perception_bevfusion_detection", max_age=0.1),
        request_timeout_s=0.2,
        namespace="test",
    )
    component.configure()
    component.activate()
    with pytest.raises(StalePerceptionInput):
        component.process(envelope(clock, input_value(timestamp=1.0), lifespan=2.0))
    assert context.health.get(component.name).state is RuntimeHealth.STALE


def test_component_failure_isolated_in_error_state() -> None:
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
    with pytest.raises(Exception):
        component.process(envelope(clock, input_value()))
    assert component.state.value == "ERROR"
    assert context.health.get(component.name).state is RuntimeHealth.FAILED


def test_pipeline_rate_gate_and_snapshot() -> None:
    clock = ManualClock(1.0)
    context = runtime(clock)
    component = PerceptionModelComponent(
        adapter=BevFusionDetectionAdapter(
            adapter_requirements(CAMERAS, True), "mit-bevfusion-swint"
        ),
        backend=FakeBackend([detection_payload(), detection_payload()]),
        runtime=context,
        contract=contract("perception_bevfusion_detection"),
        request_timeout_s=0.2,
        namespace="test",
    )
    component.configure()
    component.activate()
    executor = PriorityExecutor(
        name="test-perception", workers=1, queue_capacity=4, clock=clock
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
        first = envelope(clock, input_value(timestamp=1.0))
        futures = pipeline.submit(first)
        futures["bevfusion"].result(timeout=1.0)
        second = envelope(clock, input_value(timestamp=1.05))
        assert not pipeline.submit(second)
        assert len(pipeline.snapshot(1.1).outputs) == 1
        metrics = pipeline.metrics()
        assert metrics.submitted["bevfusion"] == 1
        assert metrics.skipped_by_rate["bevfusion"] == 1
    finally:
        executor.shutdown(wait=True, cancel_pending=True)
