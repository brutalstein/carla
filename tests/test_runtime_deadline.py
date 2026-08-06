from __future__ import annotations

from l4stack.runtime import DeadlineMonitor, ManualClock, MessageFactory
from l4stack.runtime.channel import OverflowPolicy
from l4stack.runtime.contracts import ComponentContract, ExecutionCriticality
from l4stack.runtime.deadline import DeadlineViolation


def _contract() -> ComponentContract:
    return ComponentContract(
        name="localization",
        criticality=ExecutionCriticality.FIRM_REALTIME,
        priority=20,
        max_input_age_s=0.10,
        execution_budget_s=0.02,
        expected_output_period_s=0.06,
        output_lifespan_s=0.15,
        channel_capacity=2,
        overflow_policy=OverflowPolicy.DROP_OLDEST,
    )


def test_deadline_monitor_detects_stale_input_execution_and_period_miss() -> None:
    clock = ManualClock(0.0)
    processing = ManualClock(0.0)
    monitor = DeadlineMonitor(clock, processing)
    monitor.register(_contract())
    factory = MessageFactory[dict]("sensor", clock)
    message = factory.create({}, source_timestamp=0.0, lifespan_s=0.5)

    clock.set(0.11)
    input_events = monitor.validate_input("localization", message)
    assert input_events[0].violation is DeadlineViolation.INPUT_STALE

    started = monitor.start_execution()
    processing.advance(0.03)
    execution_events = monitor.finish_execution(
        "localization", started, output_timestamp=0.11, message_id="out-1"
    )
    assert execution_events[0].violation is DeadlineViolation.EXECUTION_BUDGET

    started = monitor.start_execution()
    processing.advance(0.01)
    period_events = monitor.finish_execution(
        "localization", started, output_timestamp=0.20, message_id="out-2"
    )
    assert period_events[0].violation is DeadlineViolation.OUTPUT_PERIOD
    assert monitor.stats("localization").violations == 3
