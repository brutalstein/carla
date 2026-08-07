from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from runtime import (
    CITYSCAPES_LABELS,
    CitySemTensorRTRunner,
    FrameTiming,
    colorize_mask,
    load_labels,
    overlay_mask,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = (
    ROOT / "models/perception/citysemsegformer/model/citysemsegformer-sm120-fp32.engine"
)
DEFAULT_LABELS = ROOT / "models/perception/citysemsegformer/model/labels.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real CitySemSegFormer TensorRT 11.x smoke/latency/OpenCV test."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Real RGB/BGR image file to segment.")
    source.add_argument("--video", type=Path, help="Real video file to process frame by frame.")
    source.add_argument("--camera-index", type=int, help="OpenCV camera index for a live source.")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all/until quit.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--no-display", action="store_true", help="Benchmark without cv2.imshow.")
    parser.add_argument("--save", type=Path, help="Save the composed panel for --image only.")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def print_summary(timings: list[FrameTiming]) -> None:
    if not timings:
        print("No measured frames.")
        return
    fields = {
        "preprocess_ms": [x.preprocess_ms for x in timings],
        "gpu_pipeline_ms": [x.gpu_pipeline_ms for x in timings],
        "postprocess_ms": [x.postprocess_ms for x in timings],
        "total_ms": [x.total_ms for x in timings],
    }
    print("\n=== CitySem TensorRT timing ===")
    print(f"frames={len(timings)}")
    for name, values in fields.items():
        print(
            f"{name:16s} mean={statistics.fmean(values):7.2f} "
            f"p50={percentile(values, 50):7.2f} "
            f"p95={percentile(values, 95):7.2f} "
            f"p99={percentile(values, 99):7.2f} ms"
        )


def draw_text(panel: np.ndarray, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24


def class_summary(mask: np.ndarray, labels: tuple[str, ...]) -> str:
    ids = mask.astype(np.int64, copy=False).ravel()
    nonnegative = ids[ids >= 0]
    if nonnegative.size == 0:
        return "no valid class ids"
    counts = np.bincount(nonnegative)
    top = np.argsort(counts)[-4:][::-1]
    total = mask.size
    parts: list[str] = []
    for class_id in top:
        count = int(counts[class_id])
        if count == 0:
            continue
        label = labels[class_id] if class_id < len(labels) else f"id={class_id}"
        parts.append(f"{label}:{100.0 * count / total:.1f}%")
    return " | ".join(parts)


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compose_panel(
    frame: np.ndarray,
    mask: np.ndarray,
    timing: FrameTiming,
    labels: tuple[str, ...],
    alpha: float,
    panel_width: int,
) -> np.ndarray:
    mask_bgr = colorize_mask(mask)
    mask_bgr = cv2.resize(
        mask_bgr,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay = overlay_mask(frame, mask, alpha=alpha)

    original_view = fit_width(frame, panel_width)
    mask_view = fit_width(mask_bgr, panel_width)
    overlay_view = fit_width(overlay, panel_width)

    target_h = min(original_view.shape[0], mask_view.shape[0], overlay_view.shape[0])
    original_view = cv2.resize(original_view, (panel_width, target_h))
    mask_view = cv2.resize(mask_view, (panel_width, target_h), interpolation=cv2.INTER_NEAREST)
    overlay_view = cv2.resize(overlay_view, (panel_width, target_h))

    panel = np.hstack((original_view, mask_view, overlay_view))
    draw_text(
        panel,
        [
            "RGB                     SEGMENTATION MASK                  OVERLAY",
            (
                f"pre={timing.preprocess_ms:.1f} ms  gpu-pipeline={timing.gpu_pipeline_ms:.1f} ms  "
                f"post={timing.postprocess_ms:.1f} ms  total={timing.total_ms:.1f} ms"
            ),
            class_summary(mask, labels),
            "ESC/q: quit",
        ],
    )
    return panel


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source: str | int
    if args.video is not None:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        source = str(args.video)
    else:
        source = args.camera_index
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return capture


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    if labels != CITYSCAPES_LABELS:
        print("NOTE: labels.txt order differs from built-in Cityscapes names; file order is used.")

    timings: list[FrameTiming] = []
    with CitySemTensorRTRunner(args.engine, device_index=args.device) as runner:
        info = runner.info
        print("=== CitySem TensorRT runtime ===")
        for key, value in vars(info).items():
            print(f"{key}: {value}")

        if args.image is not None:
            if not args.image.is_file():
                raise FileNotFoundError(args.image)
            frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"OpenCV could not decode image: {args.image}")
            for _ in range(max(0, args.warmup)):
                runner.infer(frame)
            mask, timing = runner.infer(frame)
            timings.append(timing)
            panel = compose_panel(frame, mask, timing, labels, args.alpha, args.panel_width)
            if args.save is not None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), panel):
                    raise RuntimeError(f"Could not write {args.save}")
                print(f"Saved: {args.save}")
            if not args.no_display:
                cv2.namedWindow("CitySemSegFormer TensorRT", cv2.WINDOW_NORMAL)
                cv2.imshow("CitySemSegFormer TensorRT", panel)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            capture = open_capture(args)
            try:
                warmup_left = max(0, args.warmup)
                measured = 0
                if not args.no_display:
                    cv2.namedWindow("CitySemSegFormer TensorRT", cv2.WINDOW_NORMAL)
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    mask, timing = runner.infer(frame)
                    if warmup_left:
                        warmup_left -= 1
                    else:
                        timings.append(timing)
                        measured += 1
                    if not args.no_display:
                        panel = compose_panel(
                            frame, mask, timing, labels, args.alpha, args.panel_width
                        )
                        cv2.imshow("CitySemSegFormer TensorRT", panel)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                    if args.max_frames > 0 and measured >= args.max_frames:
                        break
            finally:
                capture.release()
                if not args.no_display:
                    cv2.destroyAllWindows()

    print_summary(timings)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
