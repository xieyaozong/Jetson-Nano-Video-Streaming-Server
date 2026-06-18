from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from object_stream.model_variants import Prediction, box_iou, percentile
from object_stream.live_view import choose_device


DEFAULT_MODELS = (
    "yolo11n.pt",
    "models/yolo11n-640-trt-fp32.engine",
    "models/yolo11n-640-trt-fp16.engine",
    "models/yolo11n-640-trt-int8.engine",
    "models/yolo11n-416-trt-fp32.engine",
    "models/yolo11n-416-trt-fp16.engine",
    "models/yolo11n-416-trt-int8.engine",
    "models/yolo11n-320-trt-fp32.engine",
    "models/yolo11n-320-trt-fp16.engine",
    "models/yolo11n-320-trt-int8.engine",
)


@dataclass(frozen=True)
class CameraProbe:
    requested_width: int
    requested_height: int
    requested_fps: int
    reported_width: float
    reported_height: float
    reported_fps: float
    measured_capture_fps: float
    fourcc: str
    exposure: float
    auto_exposure: float


@dataclass(frozen=True)
class ModelBenchmark:
    model: str
    imgsz: int
    frames: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    model_fps: float
    avg_objects: float
    detection_rate: float
    avg_confidence: float
    baseline_count_agreement: float | None
    baseline_mean_iou: float | None
    file_size_mb: float | None
    annotated_image: str


def capture_camera_frames(
    source: int,
    width: int,
    height: int,
    fps: int,
    frames: int,
    warmup: int,
    fourcc: str,
    auto_exposure: float | None,
    exposure: float | None,
    gain: float | None,
) -> tuple[list[object], CameraProbe]:
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open /dev/video{source}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if auto_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, gain)

    for _ in range(warmup):
        cap.read()

    captured: list[object] = []
    started = time.perf_counter()
    while len(captured) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        captured.append(frame)
    elapsed = time.perf_counter() - started

    raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((raw_fourcc >> 8 * i) & 0xFF) for i in range(4))
    probe = CameraProbe(
        requested_width=width,
        requested_height=height,
        requested_fps=fps,
        reported_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        reported_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        reported_fps=cap.get(cv2.CAP_PROP_FPS),
        measured_capture_fps=len(captured) / elapsed if elapsed else 0.0,
        fourcc=actual_fourcc,
        exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
        auto_exposure=cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
    )
    cap.release()
    return captured, probe


def benchmark_model(
    model_path: str,
    frames: list[object],
    baseline_predictions: list[list[Prediction]] | None,
    device: str,
    conf: float,
    class_filter: list[int] | None,
    model_warmup: int,
    output_dir: Path,
) -> ModelBenchmark:
    imgsz = infer_imgsz(model_path)
    model = YOLO(model_path, task="detect")
    predictions: list[list[Prediction]] = []
    elapsed_ms: list[float] = []

    for frame in frames[:model_warmup]:
        model.predict(
            frame,
            imgsz=imgsz,
            classes=class_filter,
            conf=conf,
            iou=0.45,
            device=device,
            verbose=False,
        )

    for frame in frames:
        started = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            classes=class_filter,
            conf=conf,
            iou=0.45,
            device=device,
            verbose=False,
        )[0]
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        predictions.append(predictions_from_result(result))

    annotated_path = output_dir / f"benchmark_{safe_name(Path(model_path).stem)}.jpg"
    annotated = draw_predictions(frames[-1], predictions[-1])
    cv2.imwrite(str(annotated_path), annotated)

    object_counts = [len(frame_predictions) for frame_predictions in predictions]
    confidences = [prediction.confidence for frame_predictions in predictions for prediction in frame_predictions]
    count_agreement = None
    mean_iou = None
    if baseline_predictions is not None:
        count_matches = 0
        ious: list[float] = []
        for current, baseline in zip(predictions, baseline_predictions):
            count_matches += int(len(current) == len(baseline))
            ious.append(mean_best_iou(current, baseline))
        count_agreement = count_matches / len(predictions) if predictions else 0.0
        mean_iou = statistics.mean(ious) if ious else 0.0

    path = Path(model_path)
    return ModelBenchmark(
        model=model_path,
        imgsz=imgsz,
        frames=len(frames),
        avg_ms=statistics.mean(elapsed_ms),
        p50_ms=statistics.median(elapsed_ms),
        p95_ms=percentile(elapsed_ms, 95),
        model_fps=1000.0 / statistics.mean(elapsed_ms),
        avg_objects=statistics.mean(object_counts) if object_counts else 0.0,
        detection_rate=sum(1 for count in object_counts if count > 0) / len(object_counts) if object_counts else 0.0,
        avg_confidence=statistics.mean(confidences) if confidences else 0.0,
        baseline_count_agreement=count_agreement,
        baseline_mean_iou=mean_iou,
        file_size_mb=path.stat().st_size / (1024 * 1024) if path.exists() else None,
        annotated_image=str(annotated_path),
    )


def predictions_from_result(result: object) -> list[Prediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    predictions: list[Prediction] = []
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        predictions.append(Prediction(xyxy=(x1, y1, x2, y2), confidence=float(box.conf[0].item())))
    return predictions


def draw_predictions(frame: object, predictions: list[Prediction]) -> object:
    annotated = frame.copy()
    for prediction in predictions:
        x1, y1, x2, y2 = (int(value) for value in prediction.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 220, 30), 2)
        cv2.putText(
            annotated,
            f"object {prediction.confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"objects: {len(predictions)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def mean_best_iou(current: list[Prediction], baseline: list[Prediction]) -> float:
    if not current and not baseline:
        return 1.0
    if not current or not baseline:
        return 0.0
    return statistics.mean(max(box_iou(pred.xyxy, base.xyxy) for base in baseline) for pred in current)


def infer_imgsz(model_path: str) -> int:
    match = re.search(r"-(320|416|640)-", Path(model_path).name)
    if match:
        return int(match.group(1))
    return 640


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def write_csv(path: Path, rows: list[ModelBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark existing PT/TRT models on one shared USB camera frame set.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model-warmup", type=int, default=5)
    parser.add_argument("--fourcc", choices=("YUYV", "MJPG"), default="YUYV")
    parser.add_argument("--auto-exposure", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=300.0)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", help="Optional YOLO class IDs to keep.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--baseline", default="yolo11n.pt")
    parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/camera_models_benchmark.csv"))
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device("auto")

    frames, probe = capture_camera_frames(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        warmup=args.warmup,
        fourcc=args.fourcc,
        auto_exposure=args.auto_exposure,
        exposure=args.exposure,
        gain=args.gain,
    )
    if not frames:
        raise RuntimeError("No camera frames captured.")

    source_image = args.capture_dir / "benchmark_camera_source.jpg"
    cv2.imwrite(str(source_image), frames[-1])
    print("Camera probe:", json.dumps(asdict(probe), indent=2), flush=True)
    print(f"Saved source image: {source_image}", flush=True)

    print(f"Running baseline: {args.baseline}", flush=True)
    baseline_result = benchmark_model(
        model_path=args.baseline,
        frames=frames,
        baseline_predictions=None,
        device=device,
        conf=args.conf,
        class_filter=args.classes,
        model_warmup=args.model_warmup,
        output_dir=args.capture_dir,
    )
    baseline_predictions = load_predictions(
        args.baseline,
        frames,
        device=device,
        conf=args.conf,
        class_filter=args.classes,
    )

    rows: list[ModelBenchmark] = []
    for model_path in args.models:
        print(f"\n=== Benchmarking {model_path} ===", flush=True)
        if model_path == args.baseline:
            row = ModelBenchmark(
                **{
                    **asdict(baseline_result),
                    "baseline_count_agreement": 1.0,
                    "baseline_mean_iou": 1.0,
                }
            )
        else:
            row = benchmark_model(
                model_path=model_path,
                frames=frames,
                baseline_predictions=baseline_predictions,
                device=device,
                conf=args.conf,
                class_filter=args.classes,
                model_warmup=args.model_warmup,
                output_dir=args.capture_dir,
            )
        rows.append(row)
        print(json.dumps(asdict(row), indent=2), flush=True)

    write_csv(args.output_csv, rows)
    print(f"\nCSV saved: {args.output_csv}", flush=True)


def load_predictions(
    model_path: str,
    frames: list[object],
    device: str,
    conf: float,
    class_filter: list[int] | None = None,
) -> list[list[Prediction]]:
    model = YOLO(model_path, task="detect")
    imgsz = infer_imgsz(model_path)
    predictions: list[list[Prediction]] = []
    for frame in frames:
        result = model.predict(
            frame,
            imgsz=imgsz,
            classes=class_filter,
            conf=conf,
            iou=0.45,
            device=device,
            verbose=False,
        )[0]
        predictions.append(predictions_from_result(result))
    return predictions


if __name__ == "__main__":
    main()
