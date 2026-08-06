from __future__ import annotations

import argparse
import logging
from pathlib import Path

from l4stack.app.runner import run_stack
from l4stack.config.loader import load_stack_config
from l4stack.perception.manifest import verify_installation
from l4stack.sensors.coverage import camera_azimuth_gaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic CARLA L4 foundation")
    parser.add_argument("--config-dir", default="config", help="YAML configuration directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run CARLA stack")
    run_parser.add_argument("--frames", type=int, default=None)

    subparsers.add_parser("validate", help="Validate configuration without CARLA")
    subparsers.add_parser("coverage", help="Report horizontal camera coverage gaps")
    subparsers.add_parser(
        "perception-doctor",
        help="Check perception model files and configured backend prerequisites",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_stack_config(Path(args.config_dir))
    level = config.logging["logging"].get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(level).upper()),
        format="%(levelname)s %(message)s",
    )

    if args.command == "validate":
        contracts = ",".join(sorted(config.runtime.components))
        models = ",".join(sorted(config.perception.models))
        print(
            f"OK: {len(config.sensors)} sensors, required={config.required_sensor_names}, "
            f"runtime_components={contracts}, perception_models={models}, "
            f"perception_enabled={config.perception.enabled}"
        )
        return 0
    if args.command == "coverage":
        gaps = camera_azimuth_gaps(config.sensors)
        if gaps:
            print("Camera azimuth gaps:")
            for start, end in gaps:
                print(f"  {start:.1f}° .. {end:.1f}°")
            return 2
        print("OK: configured cameras cover 360° azimuth with overlap")
        return 0
    if args.command == "perception-doctor":
        failed_enabled = False
        for report in verify_installation(config.perception):
            state = "READY" if report.ready else "NOT_READY"
            enabled = "enabled" if report.enabled else "disabled"
            print(f"{report.model_name}: {state} ({enabled})")
            backend_marker = "OK" if report.backend.ok else "FAIL"
            print(f"  [{backend_marker}] backend: {report.backend.detail}")
            for artifact in report.artifacts:
                marker = "OK" if artifact.ok else "FAIL"
                print(f"  [{marker}] {artifact.path}: {artifact.detail}")
            if report.enabled and not report.ready:
                failed_enabled = True
        return 2 if failed_enabled else 0
    if args.command == "run":
        output = run_stack(config, args.frames)
        print(f"Output: {output}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
