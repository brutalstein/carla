from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

CITYSCAPES_LABELS: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Cityscapes-compatible RGB palette for the 19 train IDs.
CITYSCAPES_PALETTE_RGB = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)

# NVIDIA TAO SegFormer preprocessing defaults. Input to this runner is OpenCV BGR.
TAO_MEAN_RGB = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
TAO_STD_RGB = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)


@dataclass(frozen=True)
class FrameTiming:
    preprocess_ms: float
    gpu_pipeline_ms: float
    postprocess_ms: float
    total_ms: float


@dataclass(frozen=True)
class RuntimeInfo:
    tensorrt_version: str
    gpu_name: str
    compute_capability: str
    input_name: str
    input_shape: tuple[int, ...]
    input_dtype: str
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str


def load_labels(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    labels = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(labels) != len(CITYSCAPES_LABELS):
        raise ValueError(f"Expected 19 labels, got {len(labels)} from {path}")
    return labels


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected HxW class-id mask, got shape={mask.shape}")
    ids = mask.astype(np.int64, copy=False)
    valid = (ids >= 0) & (ids < len(CITYSCAPES_PALETTE_RGB))
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    rgb[valid] = CITYSCAPES_PALETTE_RGB[ids[valid]]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mask_bgr = colorize_mask(mask)
    if mask_bgr.shape[:2] != frame_bgr.shape[:2]:
        mask_bgr = cv2.resize(
            mask_bgr,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, mask_bgr, alpha, 0.0)


class CitySemTensorRTRunner:
    """Persistent TensorRT 11.x runner with preallocated pinned host/device buffers."""

    def __init__(
        self,
        engine_path: Path,
        *,
        device_index: int = 0,
        expected_cc: tuple[int, int] = (12, 0),
        input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1820),
    ) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._trt = trt
        self._cudart = cudart
        self._engine_path = Path(engine_path)
        self._closed = False
        self._pinned_input = False
        self._pinned_output = False
        self._device_input: Any | None = None
        self._device_output: Any | None = None
        self._stream: Any | None = None

        if not self._engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self._engine_path}")

        self._cuda_check(cudart.cudaSetDevice(device_index), "cudaSetDevice")
        _, props = self._cuda_check(
            cudart.cudaGetDeviceProperties(device_index), "cudaGetDeviceProperties"
        )
        gpu_name = bytes(props.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        cc = (int(props.major), int(props.minor))
        if cc < expected_cc:
            raise RuntimeError(
                f"Compute capability {cc[0]}.{cc[1]} < required "
                f"{expected_cc[0]}.{expected_cc[1]}"
            )

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        engine_bytes = self._engine_path.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Could not deserialize engine: {self._engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT could not create execution context")

        inputs: list[str] = []
        outputs: list[str] = []
        for idx in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one input and one output, got inputs={inputs}, outputs={outputs}"
            )

        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_shape = tuple(int(x) for x in input_shape)
        if not self._context.set_input_shape(self.input_name, self.input_shape):
            raise RuntimeError(f"Engine rejected input shape {self.input_shape}")
        missing_shapes = self._context.infer_shapes()
        if missing_shapes:
            raise RuntimeError(f"TensorRT shape inference missing tensors: {missing_shapes}")

        self.output_shape = tuple(int(x) for x in self._context.get_tensor_shape(self.output_name))
        if any(dim <= 0 for dim in self.output_shape):
            raise RuntimeError(f"Unresolved output shape: {self.output_shape}")

        input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self.input_name)))
        output_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(self.output_name)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError(f"Expected FP32 input for current engine, got {input_dtype}")
        if output_dtype.kind not in {"i", "u"}:
            raise RuntimeError(f"Expected integer class-id output, got {output_dtype}")

        _, channels, height, width = self.input_shape
        if channels != 3:
            raise RuntimeError(f"Expected 3-channel input, got {self.input_shape}")
        self.height = height
        self.width = width

        # Reused CPU staging buffers: no per-frame large allocations.
        self._resize_bgr = np.empty((height, width, 3), dtype=np.uint8)
        self._rgb_u8 = np.empty((height, width, 3), dtype=np.uint8)
        self._normalized_hwc = np.empty((height, width, 3), dtype=np.float32)
        self._host_input = np.empty(self.input_shape, dtype=input_dtype)
        self._host_output = np.empty(self.output_shape, dtype=output_dtype)

        try:
            self._pin_host(self._host_input, "input")
            self._pinned_input = True
            self._pin_host(self._host_output, "output")
            self._pinned_output = True

            _, self._device_input = self._cuda_check(
                cudart.cudaMalloc(self._host_input.nbytes), "cudaMalloc(input)"
            )
            _, self._device_output = self._cuda_check(
                cudart.cudaMalloc(self._host_output.nbytes), "cudaMalloc(output)"
            )
            _, self._stream = self._cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")

            if not self._context.set_tensor_address(self.input_name, int(self._device_input)):
                raise RuntimeError("set_tensor_address(input) failed")
            if not self._context.set_tensor_address(self.output_name, int(self._device_output)):
                raise RuntimeError("set_tensor_address(output) failed")
        except Exception:
            self.close()
            raise

        self.info = RuntimeInfo(
            tensorrt_version=trt.__version__,
            gpu_name=gpu_name,
            compute_capability=f"{cc[0]}.{cc[1]}",
            input_name=self.input_name,
            input_shape=self.input_shape,
            input_dtype=str(input_dtype),
            output_name=self.output_name,
            output_shape=self.output_shape,
            output_dtype=str(output_dtype),
        )

    def _cuda_check(self, result: Any, name: str) -> tuple[Any, ...]:
        values = result if isinstance(result, tuple) else (result,)
        if not values:
            raise RuntimeError(f"{name} returned no status")
        status = values[0]
        if int(status) != 0:
            raise RuntimeError(f"{name} failed: {status}")
        return values

    def _pin_host(self, array: np.ndarray, name: str) -> None:
        self._cuda_check(
            self._cudart.cudaHostRegister(array.ctypes.data, array.nbytes, 0),
            f"cudaHostRegister({name})",
        )

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty input frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR HxWx3 frame, got {frame_bgr.shape}")

        cv2.resize(
            frame_bgr,
            (self.width, self.height),
            dst=self._resize_bgr,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.cvtColor(self._resize_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_u8)
        np.copyto(self._normalized_hwc, self._rgb_u8, casting="unsafe")
        np.subtract(self._normalized_hwc, TAO_MEAN_RGB, out=self._normalized_hwc)
        np.divide(self._normalized_hwc, TAO_STD_RGB, out=self._normalized_hwc)
        self._host_input[0] = self._normalized_hwc.transpose(2, 0, 1)
        return self._host_input

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FrameTiming]:
        if self._closed:
            raise RuntimeError("Runner is closed")

        total_start = perf_counter()
        preprocess_start = total_start
        self.preprocess(frame_bgr)
        preprocess_end = perf_counter()

        gpu_start = perf_counter()
        self._cuda_check(
            self._cudart.cudaMemcpyAsync(
                self._device_input,
                self._host_input.ctypes.data,
                self._host_input.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self._cuda_check(
            self._cudart.cudaMemcpyAsync(
                self._host_output.ctypes.data,
                self._device_output,
                self._host_output.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )
        self._cuda_check(self._cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
        gpu_end = perf_counter()

        post_start = gpu_end
        mask = self._decode_output()
        post_end = perf_counter()

        timing = FrameTiming(
            preprocess_ms=(preprocess_end - preprocess_start) * 1000.0,
            gpu_pipeline_ms=(gpu_end - gpu_start) * 1000.0,
            postprocess_ms=(post_end - post_start) * 1000.0,
            total_ms=(post_end - total_start) * 1000.0,
        )
        return mask, timing

    def _decode_output(self) -> np.ndarray:
        value = self._host_output
        if value.shape[0] == 1:
            value = value[0]
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim != 2:
            raise RuntimeError(f"Expected class-id HxW output, got shape={value.shape}")
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cudart = getattr(self, "_cudart", None)
        if cudart is None:
            return

        stream = getattr(self, "_stream", None)
        if stream is not None:
            try:
                cudart.cudaStreamSynchronize(stream)
            except Exception:
                pass

        for ptr_name in ("_device_input", "_device_output"):
            ptr = getattr(self, ptr_name, None)
            if ptr is not None:
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
                setattr(self, ptr_name, None)

        if stream is not None:
            try:
                cudart.cudaStreamDestroy(stream)
            except Exception:
                pass
            self._stream = None

        if getattr(self, "_pinned_input", False):
            try:
                cudart.cudaHostUnregister(self._host_input.ctypes.data)
            except Exception:
                pass
            self._pinned_input = False
        if getattr(self, "_pinned_output", False):
            try:
                cudart.cudaHostUnregister(self._host_output.ctypes.data)
            except Exception:
                pass
            self._pinned_output = False

    def __enter__(self) -> "CitySemTensorRTRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
