#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gerçek CARLA/perception JSONL çıktısına performans kapısı uygular"
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--minimum-frames", type=int, default=400)
    parser.add_argument("--maximum-failure-rate", type=float, default=0.01)
    parser.add_argument("--maximum-gpu-skip-rate", type=float, default=0.25)
    parser.add_argument("--model-p95-ms", action="append", default=[])
    parser.add_argument(
        "--critical-model",
        action="append",
        default=["bevfusion_detection", "tld_ready"],
    )
    args = parser.parse_args()

    limits: dict[str, float] = {}
    for item in args.model_p95_ms:
        try:
            name, raw = item.split("=", 1)
            limits[name] = float(raw)
        except ValueError as exc:
            raise SystemExit(f"Geçersiz --model-p95-ms: {item}") from exc

    if not args.jsonl.is_file():
        raise SystemExit(f"JSONL bulunamadı: {args.jsonl}")

    frame_count = 0
    observed_outputs: dict[str, int] = defaultdict(int)
    final_metrics: dict = {}
    p95_samples: dict[str, list[float]] = defaultdict(list)
    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Satır {line_number} JSON değil: {exc}") from exc
            perception = record.get("perception")
            if not isinstance(perception, dict):
                continue
            frame_count += 1
            for output in perception.get("snapshot", {}).get("outputs", []):
                name = str(output.get("model_name", ""))
                if name:
                    observed_outputs[name] += 1
            metrics = perception.get("metrics")
            if isinstance(metrics, dict):
                final_metrics = metrics
                for name, values in metrics.get("gpu", {}).items():
                    value = values.get("p95_ms")
                    if value is not None:
                        p95_samples[name].append(float(value))

    if frame_count < args.minimum_frames:
        raise SystemExit(
            f"Yetersiz gerçek frame: {frame_count} < {args.minimum_frames}. "
            "Sentetik veya boş log kabul edilmez."
        )
    if not observed_outputs:
        raise SystemExit("Hiçbir gerçek perception model çıktısı gözlenmedi")

    failed = final_metrics.get("failed", {})
    submitted = final_metrics.get("submitted", {})
    skipped = final_metrics.get("skipped_by_gpu_budget", {})
    errors: list[str] = []
    for name, submitted_count in submitted.items():
        submitted_count = int(submitted_count)
        if submitted_count <= 0:
            continue
        failure_rate = int(failed.get(name, 0)) / submitted_count
        skip_rate = int(skipped.get(name, 0)) / (submitted_count + int(skipped.get(name, 0)))
        if failure_rate > args.maximum_failure_rate:
            errors.append(
                f"{name} failure_rate={failure_rate:.3%} > {args.maximum_failure_rate:.3%}"
            )
        if name in args.critical_model and int(skipped.get(name, 0)) > 0:
            errors.append(f"{name} critical model GPU budget tarafından atlandı")
        elif skip_rate > args.maximum_gpu_skip_rate:
            errors.append(
                f"{name} gpu_skip_rate={skip_rate:.3%} > {args.maximum_gpu_skip_rate:.3%}"
            )

    for name, limit in limits.items():
        values = p95_samples.get(name, [])
        if not values:
            errors.append(f"{name} için p95 ölçümü yok")
            continue
        observed = percentile(values, 0.95)
        if observed > limit:
            errors.append(f"{name} observed_p95={observed:.2f}ms > {limit:.2f}ms")

    print(f"frames={frame_count}")
    for name in sorted(observed_outputs):
        print(f"{name}: outputs={observed_outputs[name]}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 2
    print("PASS: gerçek perception performans kapıları sağlandı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
