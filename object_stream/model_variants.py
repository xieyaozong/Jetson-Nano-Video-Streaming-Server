from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from object_stream.live_view import (
    build_jetson_csi_pipeline,
    choose_device,
)


DEFAULT_VARIANTS = (
    "onnx-fp32",
    "onnx-fp16",
    "onnx-int8",
    "trt-fp16",
    "trt-int8",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    format: str
    half: bool = False
    int8: bool = False


@dataclass(frozen=True)
class Prediction:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    frames: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    avg_objects: float
    avg_confidence: float
    baseline_count_agreement: float | None = None
    baseline_mean_iou: float | None = None
    file_size_mb: float | None = None


def variant_specs(names: Iterable[str]) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for name in names:
        normalized = name.lower()
        if normalized == "onnx-fp32":
            specs.append(VariantSpec(name=normalized, format="onnx"))
        elif normalized == "onnx-fp16":
            specs.append(VariantSpec(name=normalized, format="onnx", half=True))
        elif normalized == "onnx-int8":
            specs.append(VariantSpec(name=normalized, format="onnx", int8=True))
        elif normalized == "trt-fp32":
            specs.append(VariantSpec(name=normalized, format="engine"))
        elif normalized == "trt-fp16":
            specs.append(VariantSpec(name=normalized, format="engine", half=True))
        elif normalized == "trt-int8":
            specs.append(VariantSpec(name=normalized, format="engine", int8=True))
        else:
            raise ValueError(f"Unknown variant '{name}'. Valid: {', '.join(DEFAULT_VARIANTS)}, trt-fp32")
    return specs


def export_variants(
    model_path: str,
    variants: Iterable[str],
    imgsz: int,
    device: str,
    data: str,
    fraction: float,
    batch: int,
    output_dir: Path,
    workspace: float | None,
    simplify: bool,
    continue_on_error: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    selected_device = choose_device(device)

    for spec in variant_specs(variants):
        print(f"\n=== Exporting {spec.name} ===", flush=True)
        try:
            model = YOLO(model_path, task="detect")
            kwargs = {
                "format": spec.format,
                "imgsz": imgsz,
                "half": spec.half,
                "int8": spec.int8,
                "batch": batch,
                "simplify": simplify,
            }
            if spec.format == "engine":
                kwargs["device"] = selected_device
            elif spec.half:
                kwargs["device"] = selected_device
            if spec.int8:
                kwargs["data"] = data
                kwargs["fraction"] = fraction
            if workspace is not None and spec.format == "engine":
                kwargs["workspace"] = workspace

            path = Path(model.export(**kwargs))
            destination = output_dir / _variant_filename(Path(model_path), spec, path, imgsz)
            if destination.exists():
                destination.unlink()
            path.replace(destination)
            exported.append(destination)
            print(f"saved: {destination}", flush=True)
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"failed: {spec.name}: {exc}", flush=True)

    return exported


def benchmark_models(
    model_paths: Iterable[str],
    source: int | str | None,
    imgsz: int,
    device: str,
    frames: int,
    warmup: int,
    width: int,
    height: int,
    fps: int,
    conf: float,
    iou: float,
    class_filter: list[int] | None,
    baseline_path: str | None,
    output_csv: Path,
) -> list[BenchmarkResult]:
    sample_frames = load_frames(source=source, count=frames + warmup, width=width, height=height, fps=fps)
    if len(sample_frames) <= warmup:
        raise RuntimeError("Not enough frames captured for benchmarking.")

    selected_device = choose_device(device)
    baseline_predictions = None
    if baseline_path:
        print(f"Running baseline for detection agreement: {baseline_path}", flush=True)
        baseline_predictions, _ = predict_frames(
            model_path=baseline_path,
            sample_frames=sample_frames,
            imgsz=imgsz,
            device=selected_device,
            warmup=warmup,
            conf=conf,
            iou=iou,
            class_filter=class_filter,
        )

    results: list[BenchmarkResult] = []
    for model_path in model_paths:
        print(f"\n=== Benchmarking {model_path} ===", flush=True)
        predictions, elapsed_ms = predict_frames(
            model_path=model_path,
            sample_frames=sample_frames,
            imgsz=imgsz,
            device=selected_device,
            warmup=warmup,
            conf=conf,
            iou=iou,
            class_filter=class_filter,
        )
        result = summarize_predictions(
            model_path=Path(model_path),
            predictions=predictions,
            elapsed_ms=elapsed_ms,
            baseline_predictions=baseline_predictions,
        )
        results.append(result)
        print(json.dumps(asdict(result), indent=2), flush=True)

    write_results(output_csv, results)
    print(f"\nCSV saved: {output_csv}", flush=True)
    return results


def load_frames(source: int | str | None, count: int, width: int, height: int, fps: int) -> list[np.ndarray]:
    if source is None:
        return synthetic_frames(count=count, width=width, height=height)

    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER) if isinstance(source, str) and "!" in source else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def synthetic_frames(count: int, width: int, height: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40 + idx % 120, 60), (220 + idx % 120, height - 40), (180, 180, 180), -1)
        cv2.circle(frame, (130 + idx % 120, 70), 45, (220, 220, 220), -1)
        frames.append(frame)
    return frames


def predict_frames(
    model_path: str,
    sample_frames: list[np.ndarray],
    imgsz: int,
    device: str,
    warmup: int,
    conf: float,
    iou: float,
    class_filter: list[int] | None,
) -> tuple[list[list[Prediction]], list[float]]:
    model = YOLO(model_path, task="detect")
    runtime_device = runtime_device_for_model(model_path=model_path, requested_device=device)
    predictions: list[list[Prediction]] = []
    elapsed_ms: list[float] = []

    for index, frame in enumerate(sample_frames):
        start = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            classes=class_filter,
            conf=conf,
            iou=iou,
            device=runtime_device,
            verbose=False,
        )[0]
        duration_ms = (time.perf_counter() - start) * 1000.0
        if index < warmup:
            continue
        predictions.append(_predictions_from_result(result))
        elapsed_ms.append(duration_ms)

    return predictions, elapsed_ms


def runtime_device_for_model(model_path: str, requested_device: str) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx" and requested_device in {"auto", "cuda", "cuda:0", "0"}:
        return "cpu"
    return requested_device


def summarize_predictions(
    model_path: Path,
    predictions: list[list[Prediction]],
    elapsed_ms: list[float],
    baseline_predictions: list[list[Prediction]] | None,
) -> BenchmarkResult:
    objects_counts = [len(frame_predictions) for frame_predictions in predictions]
    confidences = [prediction.confidence for frame_predictions in predictions for prediction in frame_predictions]
    baseline_count_agreement = None
    baseline_mean_iou = None

    if baseline_predictions is not None:
        compared = zip(predictions, baseline_predictions)
        count_matches = 0
        mean_ious: list[float] = []
        total = 0
        for current, baseline in compared:
            total += 1
            if len(current) == len(baseline):
                count_matches += 1
            mean_ious.append(mean_best_iou(current, baseline))
        baseline_count_agreement = count_matches / total if total else 0.0
        baseline_mean_iou = statistics.mean(mean_ious) if mean_ious else 0.0

    return BenchmarkResult(
        model=str(model_path),
        frames=len(predictions),
        avg_ms=statistics.mean(elapsed_ms),
        p50_ms=statistics.median(elapsed_ms),
        p95_ms=percentile(elapsed_ms, 95),
        fps=1000.0 / statistics.mean(elapsed_ms),
        avg_objects=statistics.mean(objects_counts) if objects_counts else 0.0,
        avg_confidence=statistics.mean(confidences) if confidences else 0.0,
        baseline_count_agreement=baseline_count_agreement,
        baseline_mean_iou=baseline_mean_iou,
        file_size_mb=model_path.stat().st_size / (1024 * 1024) if model_path.exists() else None,
    )


def _predictions_from_result(result: object) -> list[Prediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    predictions: list[Prediction] = []
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        predictions.append(Prediction(xyxy=(x1, y1, x2, y2), confidence=float(box.conf[0].item())))
    return predictions


def mean_best_iou(current: list[Prediction], baseline: list[Prediction]) -> float:
    if not current and not baseline:
        return 1.0
    if not current or not baseline:
        return 0.0
    return statistics.mean(max(box_iou(pred.xyxy, base.xyxy) for base in baseline) for pred in current)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    intersection = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((pct / 100) * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _variant_filename(model_path: Path, spec: VariantSpec, exported_path: Path, imgsz: int) -> str:
    return f"{model_path.stem}-{imgsz}-{spec.name}{exported_path.suffix}"


def parse_source(value: str, jetson_csi: bool, sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> int | str | None:
    if value == "synthetic":
        return None
    if jetson_csi:
        return build_jetson_csi_pipeline(
            sensor_id=sensor_id,
            width=width,
            height=height,
            fps=fps,
            flip_method=flip_method,
        )
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and benchmark YOLO ONNX/TensorRT variants.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model", default="yolo11n.pt")
    export_parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    export_parser.add_argument("--imgsz", type=int, default=640)
    export_parser.add_argument("--device", default="auto")
    export_parser.add_argument("--data", default="coco8.yaml")
    export_parser.add_argument("--fraction", type=float, default=1.0)
    export_parser.add_argument("--batch", type=int, default=1)
    export_parser.add_argument("--output-dir", type=Path, default=Path("models"))
    export_parser.add_argument("--workspace", type=float)
    export_parser.add_argument("--no-simplify", action="store_true")
    export_parser.add_argument("--continue-on-error", action="store_true")

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("models", nargs="+")
    bench_parser.add_argument("--baseline", default="yolo11n.pt")
    bench_parser.add_argument("--source", default="synthetic")
    bench_parser.add_argument("--jetson-csi", action="store_true")
    bench_parser.add_argument("--sensor-id", type=int, default=0)
    bench_parser.add_argument("--flip-method", type=int, default=0)
    bench_parser.add_argument("--imgsz", type=int, default=640)
    bench_parser.add_argument("--device", default="auto")
    bench_parser.add_argument("--frames", type=int, default=120)
    bench_parser.add_argument("--warmup", type=int, default=10)
    bench_parser.add_argument("--width", type=int, default=1280)
    bench_parser.add_argument("--height", type=int, default=720)
    bench_parser.add_argument("--fps", type=int, default=30)
    bench_parser.add_argument("--conf", type=float, default=0.35)
    bench_parser.add_argument("--iou", type=float, default=0.45)
    bench_parser.add_argument("--classes", type=int, nargs="+", help="Optional YOLO class IDs to keep.")
    bench_parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/benchmark_results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_variants(
            model_path=args.model,
            variants=args.variants,
            imgsz=args.imgsz,
            device=args.device,
            data=args.data,
            fraction=args.fraction,
            batch=args.batch,
            output_dir=args.output_dir,
            workspace=args.workspace,
            simplify=not args.no_simplify,
            continue_on_error=args.continue_on_error,
        )
    elif args.command == "benchmark":
        source = parse_source(
            value=args.source,
            jetson_csi=args.jetson_csi,
            sensor_id=args.sensor_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip_method,
        )
        benchmark_models(
            model_paths=args.models,
            source=source,
            imgsz=args.imgsz,
            device=args.device,
            frames=args.frames,
            warmup=args.warmup,
            width=args.width,
            height=args.height,
            fps=args.fps,
            conf=args.conf,
            iou=args.iou,
            class_filter=args.classes,
            baseline_path=args.baseline,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
