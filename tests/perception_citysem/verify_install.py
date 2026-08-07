from __future__ import annotations

from pathlib import Path


def main() -> int:
    import cv2
    import numpy as np
    import onnx
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models/perception/citysemsegformer/model"
    engine = model_dir / "citysemsegformer-sm120-fp32.engine"
    onnx_path = model_dir / "citysemsegformer.onnx"
    labels = model_dir / "labels.txt"

    err, count = cudart.cudaGetDeviceCount()
    if int(err) != 0 or count < 1:
        raise RuntimeError(f"CUDA device check failed: error={err}, count={count}")
    err, props = cudart.cudaGetDeviceProperties(0)
    if int(err) != 0:
        raise RuntimeError(f"cudaGetDeviceProperties failed: {err}")
    gpu = bytes(props.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    cc = f"{props.major}.{props.minor}"

    required = (onnx_path, labels, engine)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    print("=== CitySem test environment ===")
    print(f"GPU: {gpu}")
    print(f"Compute capability: {cc}")
    print(f"TensorRT: {trt.__version__}")
    print(f"CUDA devices: {count}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"Engine: {engine} ({engine.stat().st_size / 1024**2:.2f} MiB)")

    from runtime import CitySemTensorRTRunner

    with CitySemTensorRTRunner(engine) as runner:
        print(
            f"Engine input: {runner.info.input_name} "
            f"{runner.info.input_shape} {runner.info.input_dtype}"
        )
        print(
            f"Engine output: {runner.info.output_name} "
            f"{runner.info.output_shape} {runner.info.output_dtype}"
        )
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
