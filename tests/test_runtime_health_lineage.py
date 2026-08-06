from __future__ import annotations

from l4stack.runtime import HealthRegistry, HealthReport, LineageStore, ManualClock, MessageFactory
from l4stack.runtime.health import RuntimeHealth


def test_health_registry_marks_old_report_stale() -> None:
    registry = HealthRegistry()
    registry.report(HealthReport("localization", RuntimeHealth.NOMINAL, 1.0))
    stale = registry.mark_stale(1.5, {"localization": 0.2})

    assert stale[0].state is RuntimeHealth.STALE
    assert registry.aggregate() is RuntimeHealth.STALE


def test_lineage_traces_output_to_input() -> None:
    clock = ManualClock(5.0)
    inputs = MessageFactory[dict]("sensors", clock, namespace="test")
    outputs = MessageFactory[dict]("localization", clock, namespace="test")
    input_message = inputs.create({}, source_timestamp=5.0, lifespan_s=1.0)
    output_message = outputs.create(
        {},
        source_timestamp=5.0,
        lifespan_s=1.0,
        parents=(input_message.message_id,),
    )
    lineage = LineageStore(10)
    lineage.record(input_message)
    lineage.record(output_message)

    trace = lineage.trace(output_message.message_id)
    assert [record.message_id for record in trace] == [
        output_message.message_id,
        input_message.message_id,
    ]
