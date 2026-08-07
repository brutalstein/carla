from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from runtime import CitySemTensorRTRunner

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = (
    ROOT / "models/perception/citysemsegformer/model/citysemsegformer-sm120-fp32.engine"
)


@dataclass(frozen=True)
class BreakdownTiming:
    preprocess_ms: float
    h2d_ms: float
    inference_ms: float
    d2h_ms: float
    gpu_event_total_ms: float
    gpu_pipeline_wall_ms: float
    postprocess_ms: float
    total_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile CitySem TensorRT H2D / execute_async_v3 / D2H with persistent CUDA events."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Real video source.")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=200)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(name: str, values: list[float]) -> None:
    print(
        f"{name:22s} mean={statistics.fmean(values):8.3f} "
        f"p50={percentile(values, 50):8.3f} "
        f"p95={percentile(values, 95):8.3f} "
        f"p99={percentile(values, 99):8.3f} ms"
    )


class CudaEventTimer:
    def __init__(self, runner: CitySemTensorRTRunner) -> None:
        self.runner = runner
        self.cudart = runner._cudart
        self.stream = runner._stream
        self.events: list[Any] = []

        for name in ("start", "after_h2d", "after_infer", "after_d2h"):
            values = self._check(self.cudart.cudaEventCreate(), f"cudaEventCreate({name})")
            self.events.append(values[1])

    def _check(self, result: Any, name: str) -> tuple[Any, ...]:
        values = result if isinstance(result, tuple) else (result,)
        if not values or int(values[0]) != 0:
            status = values[0] if values else "no status"
            raise RuntimeError(f"{name} failed: {status}")
        return values

    def record(self, event_index: int) -> None:
        self._check(
            self.cudart.cudaEventRecord(self.events[event_index], self.stream),
            f"cudaEventRecord({event_index})",
        )

    def synchronize_final(self) -> None:
        self._check(
            self.cudart.cudaEventSynchronize(self.events[3]),
            "cudaEventSynchronize(after_d2h)",
        )

    def elapsed_ms(self, start_index: int, end_index: int) -> float:
        values = self._check(
            self.cudart.cudaEventElapsedTime(
                self.events[start_index],
                self.events[end_index],
            ),
            f"cudaEventElapsedTime({start_index},{end_index})",
        )
        return float(values[1])

    def close(self) -> None:
        for event in self.events:
            try:
                self.cudart.cudaEventDestroy(event)
            except Exception:
                pass
        self.events.clear()

    def __enter__(self) -> "CudaEventTimer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def infer_profiled(
    runner: CitySemTensorRTRunner,
    timer: CudaEventTimer,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, BreakdownTiming]:
    total_start = perf_counter()

    preprocess_start = total_start
    runner.preprocess(frame_bgr)
    preprocess_end = perf_counter()

    gpu_wall_start = perf_counter()

    timer.record(0)
    runner._cuda_check(
        runner._cudart.cudaMemcpyAsync(
            runner._device_input,
            runner._host_input.ctypes.data,
            runner._host_input.nbytes,
            runner._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            runner._stream,
        ),
        "cudaMemcpyAsync(H2D)",
    )
    timer.record(1)

    if not runner._context.execute_async_v3(runner._stream):
        raise RuntimeError("TensorRT execute_async_v3 failed")
    timer.record(2)

    runner._cuda_check(
        runner._cudart.cudaMemcpyAsync(
            runner._host_output.ctypes.data,
            runner._device_output,
            runner._host_output.nbytes,
            runner._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            runner._stream,
        ),
        "cudaMemcpyAsync(D2H)",
    )
    timer.record(3)
    timer.synchronize_final()

    gpu_wall_end = perf_counter()

    h2d_ms = timer.elapsed_ms(0, 1)
    inference_ms = timer.elapsed_ms(1, 2)
    d2h_ms = timer.elapsed_ms(2, 3)
    gpu_event_total_ms = timer.elapsed_ms(0, 3)

    post_start = gpu_wall_end
    mask = runner._decode_output()
    post_end = perf_counter()

    timing = BreakdownTiming(
        preprocess_ms=(preprocess_end - preprocess_start) * 1000.0,
        h2d_ms=h2d_ms,
        inference_ms=inference_ms,
        d2h_ms=d2h_ms,
        gpu_event_total_ms=gpu_event_total_ms,
        gpu_pipeline_wall_ms=(gpu_wall_end - gpu_wall_start) * 1000.0,
        postprocess_ms=(post_end - post_start) * 1000.0,
        total_ms=(post_end - total_start) * 1000.0,
    )
    return mask, timing


def print_summary(
    timings: list[BreakdownTiming],
    input_bytes: int,
    output_bytes: int,
) -> None:
    if not timings:
        raise RuntimeError("No measured frames")

    print("\n=== CitySem CUDA event breakdown ===")
    print(f"frames={len(timings)}")
    print(f"input_transfer={input_bytes / 1024**2:.2f} MiB/frame")
    print(f"output_transfer={output_bytes / 1024**2:.2f} MiB/frame")

    fields = (
        ("preprocess_ms", [x.preprocess_ms for x in timings]),
        ("h2d_ms", [x.h2d_ms for x in timings]),
        ("inference_ms", [x.inference_ms for x in timings]),
        ("d2h_ms", [x.d2h_ms for x in timings]),
        ("gpu_event_total_ms", [x.gpu_event_total_ms for x in timings]),
        ("gpu_pipeline_wall_ms", [x.gpu_pipeline_wall_ms for x in timings]),
        ("postprocess_ms", [x.postprocess_ms for x in timings]),
        ("total_ms", [x.total_ms for x in timings]),
    )
    for name, values in fields:
        summarize(name, values)

    mean_h2d = statistics.fmean(x.h2d_ms for x in timings)
    mean_d2h = statistics.fmean(x.d2h_ms for x in timings)
    h2d_gbps = (input_bytes / 1e9) / (mean_h2d / 1000.0)
    d2h_gbps = (output_bytes / 1e9) / (mean_d2h / 1000.0)

    print("\n=== Effective transfer throughput ===")
    print(f"H2D mean={h2d_gbps:.2f} GB/s")
    print(f"D2H mean={d2h_gbps:.2f} GB/s")


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    timings: list[BreakdownTiming] = []

    try:
        with CitySemTensorRTRunner(args.engine, device_index=args.device) as runner:
            info = runner.info
            print("=== CitySem TensorRT runtime ===")
            for key, value in vars(info).items():
                print(f"{key}: {value}")

            with CudaEventTimer(runner) as timer:
                warmup_left = args.warmup
                while len(timings) < args.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    mask, timing = infer_profiled(runner, timer, frame)

                    # Fail closed if the class-id contract is violated.
                    mask_min = int(mask.min())
                    mask_max = int(mask.max())
                    if mask_min < 0 or mask_max >= 19:
                        raise RuntimeError(
                            f"Invalid class-id range: min={mask_min}, max={mask_max}"
                        )

                    if warmup_left:
                        warmup_left -= 1
                        continue
                    timings.append(timing)

            print_summary(
                timings,
                input_bytes=runner._host_input.nbytes,
                output_bytes=runner._host_output.nbytes,
            )
    finally:
        capture.release()

    if len(timings) < args.max_frames:
        print(
            f"\nNOTE: requested {args.max_frames} measured frames, "
            f"video provided only {len(timings)} after warm-up."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
