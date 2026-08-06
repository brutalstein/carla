# Validation Report

Date: 2026-08-06

## Completed

- YAML configuration load and schema checks: passed
- Unique sensor name validation: passed
- Required sensor declaration: passed
- 360-degree horizontal camera coverage check: passed, no angular gaps
- Exact-frame sensor synchronizer unit test: passed
- Deterministic semantic LiDAR instance detector test: passed
- Deterministic velocity tracker test: passed
- Nominal ODD monitor test: passed
- Python bytecode compilation: passed

Pytest result: **6 passed**.

## Not executed here

A CARLA server/runtime is not installed in the artifact execution environment, so
an end-to-end spawn and sensor callback run was not executed. Use the following
with CARLA running:

```bash
python -m l4stack.cli --config-dir config run --frames 20
```

The runtime intentionally fails if a required sensor blueprint, required sensor
attribute, CARLA connection, or exact-frame sensor sample is unavailable.
