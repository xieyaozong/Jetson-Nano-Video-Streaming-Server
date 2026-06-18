from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from object_stream.live_view import iter_object_detections


DEFAULT_MODELS = (
    ("pt-cuda", "yolo11n.pt"),
    ("trt-fp16", "models/yolo11n-640-trt-fp16.engine"),
    ("trt-int8", "models/yolo11n-640-trt-int8.engine"),
)


def probe_camera(source: int, backend: int, width: int, height: int, fps: int, seconds: float) -> dict[str, float | str]:
    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    for _ in range(10):
        cap.read()

    frames = 0
    shape = ""
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ok, frame = cap.read()
        if ok:
            frames += 1
            shape = "x".join(str(value) for value in frame.shape)

    elapsed = time.perf_counter() - start
    result = {
        "source": source,
        "requested": f"{width}x{height}@{fps}",
        "reported_width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "reported_height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_shape": shape,
        "measured_capture_fps": frames / elapsed if elapsed else 0.0,
    }
    cap.release()
    return result


def run_model_test(
    name: str,
    model_path: str,
    source: int,
    backend: int,
    width: int,
    height: int,
    fps: int,
    imgsz: int,
    conf: float,
    class_filter: list[int] | None,
    sample_frames: int,
    warmup_frames: int,
    output_dir: Path,
) -> dict[str, float | int | str]:
    generator = iter_object_detections(
        source=source,
        model_path=model_path,
        device="auto",
        backend=backend,
        width=width,
        height=height,
        target_fps=fps,
        imgsz=imgsz,
        conf=conf,
        class_filter=class_filter,
        iou=0.45,
    )

    counts: list[int] = []
    inference_ms: list[float] = []
    last_frame = None
    total_started = time.perf_counter()
    sample_started = None
    try:
        for index, item in zip(range(sample_frames + warmup_frames), generator):
            if index >= warmup_frames:
                if sample_started is None:
                    sample_started = time.perf_counter()
                counts.append(len(item.detections))
                inference_ms.append(item.inference_ms)
                last_frame = item.frame
    finally:
        generator.close()

    total_elapsed = time.perf_counter() - total_started
    sample_elapsed = time.perf_counter() - sample_started if sample_started is not None else 0.0
    annotated = output_dir / f"camera_detect_{name}.jpg"
    if last_frame is not None:
        cv2.imwrite(str(annotated), last_frame)

    detected_frames = sum(1 for count in counts if count > 0)
    avg_inference = sum(inference_ms) / len(inference_ms)
    return {
        "model_name": name,
        "model_path": model_path,
        "frames": len(counts),
        "warmup_frames": warmup_frames,
        "sample_elapsed_s": sample_elapsed,
        "total_elapsed_s": total_elapsed,
        "sample_throughput_fps": len(counts) / sample_elapsed if sample_elapsed else 0.0,
        "avg_inference_ms": avg_inference,
        "min_inference_ms": min(inference_ms),
        "max_inference_ms": max(inference_ms),
        "avg_objects": sum(counts) / len(counts),
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / len(counts),
        "annotated_image": str(annotated),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe USB camera and benchmark object detection models.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", help="Optional YOLO class IDs to keep.")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/camera_test_results.csv"))
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = cv2.CAP_V4L2
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_camera(args.source, backend, args.width, args.height, args.fps, seconds=3.0)
    print("Camera probe:", probe, flush=True)

    rows = [
        run_model_test(
            name=name,
            model_path=model_path,
            source=args.source,
            backend=backend,
            width=args.width,
            height=args.height,
            fps=args.fps,
            imgsz=args.imgsz,
            conf=args.conf,
            class_filter=args.classes,
            sample_frames=args.frames,
            warmup_frames=args.warmup,
            output_dir=args.capture_dir,
        )
        for name, model_path in DEFAULT_MODELS
    ]
    write_csv(args.output_csv, rows)

    for row in rows:
        print(row, flush=True)
    print(f"Saved CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
