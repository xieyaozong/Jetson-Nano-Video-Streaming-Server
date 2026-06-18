from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_dir: Path,
        input_shape: tuple[int, int, int, int],
        cache_path: Path,
    ) -> None:
        super().__init__()
        self.image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not self.image_paths:
            raise RuntimeError(f"No calibration images found in {image_dir}")

        self.batch_size, self.channels, self.height, self.width = input_shape
        if self.batch_size != 1:
            raise RuntimeError("This calibrator currently supports static batch size 1.")
        if self.channels != 3:
            raise RuntimeError(f"Expected 3 input channels, got {self.channels}.")

        self.cache_path = cache_path
        self.index = 0
        self.device_input = torch.empty(input_shape, dtype=torch.float32, device="cuda")
        print(f"INT8 calibration images: {len(self.image_paths)} from {image_dir}", flush=True)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index >= len(self.image_paths):
            return None

        image = cv2.imread(str(self.image_paths[self.index]))
        self.index += 1
        if image is None:
            return self.get_batch(names)

        tensor = preprocess_bgr(image, width=self.width, height=self.height)
        self.device_input.copy_(torch.from_numpy(tensor).to("cuda", non_blocking=True))
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.exists():
            print(f"Using calibration cache: {self.cache_path}", flush=True)
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        print(f"Wrote calibration cache: {self.cache_path}", flush=True)


def preprocess_bgr(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = letterbox(image, width=width, height=height)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(np.ascontiguousarray(chw), axis=0)


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    int8: bool,
    calibration_images: Path | None,
    calibration_cache: Path | None,
    workspace_mb: int,
    optimization_level: int,
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}", flush=True)
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index), flush=True)
        raise RuntimeError(f"Could not parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.builder_optimization_level = optimization_level

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: platform_has_fast_fp16 is false, but FP16 flag will still be requested.", flush=True)
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            print("Warning: platform_has_fast_int8 is false, but INT8 flag will still be requested.", flush=True)
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_images is not None:
            input_tensor = network.get_input(0)
            input_shape = tuple(int(dim) for dim in input_tensor.shape)
            cache_path = calibration_cache or engine_path.with_suffix(".calib")
            config.int8_calibrator = ImageEntropyCalibrator(
                image_dir=calibration_images,
                input_shape=input_shape,
                cache_path=cache_path,
            )

    print("Network inputs:", flush=True)
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        print(f"  {tensor.name}: shape={tensor.shape} dtype={tensor.dtype}", flush=True)

    print("Network outputs:", flush=True)
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        print(f"  {tensor.name}: shape={tensor.shape} dtype={tensor.dtype}", flush=True)

    print(
        f"Building TensorRT engine: fp16={fp16} workspace={workspace_mb}MB "
        f"int8={int8} optimization_level={optimization_level}",
        flush=True,
    )
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT returned an empty engine.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"Saved engine: {engine_path} ({engine_path.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument("--optimization-level", type=int, default=0, choices=range(0, 6))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        calibration_images=args.calibration_images,
        calibration_cache=args.calibration_cache,
        workspace_mb=args.workspace_mb,
        optimization_level=args.optimization_level,
    )


if __name__ == "__main__":
    main()
