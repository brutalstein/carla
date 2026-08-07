#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from l4stack.config.loader import load_stack_config
from l4stack.perception.cuda_doctor import run_cuda_doctor


def main() -> int:
    parser = argparse.ArgumentParser(description="RTX 5090 CUDA perception host doğrulaması")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()
    config = load_stack_config(Path(args.config_dir))
    report = run_cuda_doctor(config.perception)
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
