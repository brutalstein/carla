from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models/perception/citysemsegformer/model"
    parser = argparse.ArgumentParser(
        description="Build the CitySemSegFormer FP32 TensorRT engine on RTX 5090."
    )
    parser.add_argument("--onnx", type=Path, default=model_dir / "citysemsegformer.onnx")
    parser.add_argument(
        "--engine",
        type=Path,
        default=model_dir / "citysemsegformer-sm120-fp32.engine",
    )
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument("--optimization-level", type=int, default=5, choices=range(0, 6))
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def cuda_check(result, name: str):
    values = result if isinstance(result, tuple) else (result,)
    if int(values[0]) != 0:
        raise RuntimeError(f"{name} failed: {values[0]}")
    return values


def main() -> int:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    cuda_check(cudart.cudaSetDevice(args.device), "cudaSetDevice")
    _, props = cuda_check(cudart.cudaGetDeviceProperties(args.device), "cudaGetDeviceProperties")
    name = bytes(props.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    cc = (int(props.major), int(props.minor))
    if cc < (12, 0):
        raise RuntimeError(f"RTX 5090 / SM 12.0 expected, found {name} cc={cc[0]}.{cc[1]}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"Expected 1 input/1 output, got {network.num_inputs}/{network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    shape = tuple(int(v) for v in input_tensor.shape)
    if len(shape) != 4 or shape[1:] != (3, 1024, 1820):
        raise RuntimeError(f"Unexpected CitySem input shape: {shape}")

    config = builder.create_builder_config()
    config.builder_optimization_level = args.optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * 1024**3)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (1, 3, 1024, 1820),
        (1, 3, 1024, 1820),
        (1, 3, 1024, 1820),
    )
    config.add_optimization_profile(profile)

    print(f"GPU: {name} cc={cc[0]}.{cc[1]}")
    print(f"TensorRT: {trt.__version__}")
    print(f"Input: {input_tensor.name} {shape} {input_tensor.dtype}")
    print(f"Output: {output_tensor.name} {tuple(output_tensor.shape)} {output_tensor.dtype}")
    print("Building FP32 engine with a fixed batch=1 profile...")

    start = perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    elapsed = perf_counter() - start
    print(f"ENGINE OK: {args.engine}")
    print(f"Size: {args.engine.stat().st_size / 1024**2:.2f} MiB")
    print(f"Build: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
