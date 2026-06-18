from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2

from tools.benchmark_models import DEFAULT_MODELS, benchmark_model, capture_camera_frames, load_predictions
from object_stream.live_view import choose_device


CAMERA_PROBE_SIZES = (
    "160x120",
    "320x240",
    "424x240",
    "640x360",
    "640x480",
    "800x600",
    "960x540",
    "1024x768",
    "1280x720",
    "1280x960",
    "1920x1080",
)

REPORT_COLUMNS = (
    "camera_requested",
    "camera_reported",
    "model",
    "model_imgsz",
    "avg_ms",
    "p95_ms",
    "model_fps",
    "camera_capture_fps",
    "detection_rate",
    "avg_confidence",
    "baseline_mean_iou",
)


@dataclass(frozen=True)
class MatrixRow:
    camera_requested: str
    camera_reported: str
    camera_frame_shape: str
    camera_capture_fps: float
    camera_fourcc: str
    camera_exposure: float
    model: str
    model_imgsz: int
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


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = text.lower().split("x", 1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {text!r}") from exc


def camera_sort_key(camera_name: str) -> tuple[int, int]:
    size = camera_name.split("@", 1)[0]
    return parse_size(size)


def set_camera_mode(cap: cv2.VideoCapture, width: int, height: int, fps: int, fourcc: str) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def probe_camera_sizes(source: int, fps: int, fourcc: str) -> list[str]:
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {source!r}.")

    sizes: list[str] = []
    seen: set[tuple[int, int]] = set()
    try:
        for candidate in CAMERA_PROBE_SIZES:
            width, height = parse_size(candidate)
            set_camera_mode(cap, width, height, fps, fourcc)

            ok, frame = cap.read()
            if not ok:
                continue

            reported = (int(frame.shape[1]), int(frame.shape[0]))
            if reported in seen:
                continue

            seen.add(reported)
            sizes.append(f"{reported[0]}x{reported[1]}")
    finally:
        cap.release()

    if not sizes:
        raise RuntimeError("No working camera sizes were detected.")
    return sizes


def camera_sizes_from_args(args: argparse.Namespace) -> list[str]:
    if args.camera_sizes == ["auto"]:
        sizes = probe_camera_sizes(args.source, args.fps, args.fourcc)
        print(f"Detected camera sizes: {', '.join(sizes)}", flush=True)
        return sizes
    return args.camera_sizes


def save_source_frame(frame: object, output_dir: Path, width: int, height: int) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"source_{width}x{height}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def build_row(args: argparse.Namespace, model_path: str, frame_shape: str, probe: object, benchmark: object) -> MatrixRow:
    is_baseline = model_path == args.baseline
    return MatrixRow(
        camera_requested=f"{probe.requested_width:.0f}x{probe.requested_height:.0f}@{args.fps}",
        camera_reported=f"{probe.reported_width:.0f}x{probe.reported_height:.0f}@{probe.reported_fps:.1f}",
        camera_frame_shape=frame_shape,
        camera_capture_fps=probe.measured_capture_fps,
        camera_fourcc=probe.fourcc,
        camera_exposure=probe.exposure,
        model=benchmark.model,
        model_imgsz=benchmark.imgsz,
        frames=benchmark.frames,
        avg_ms=benchmark.avg_ms,
        p50_ms=benchmark.p50_ms,
        p95_ms=benchmark.p95_ms,
        model_fps=benchmark.model_fps,
        avg_objects=benchmark.avg_objects,
        detection_rate=benchmark.detection_rate,
        avg_confidence=benchmark.avg_confidence,
        baseline_count_agreement=1.0 if is_baseline else benchmark.baseline_count_agreement,
        baseline_mean_iou=1.0 if is_baseline else benchmark.baseline_mean_iou,
        file_size_mb=benchmark.file_size_mb,
        annotated_image=benchmark.annotated_image,
    )


def benchmark_camera_size(args: argparse.Namespace, size_text: str, device: str) -> list[MatrixRow]:
    width, height = parse_size(size_text)
    print(f"\n== {width}x{height} ==", flush=True)

    frames, probe = capture_camera_frames(
        source=args.source,
        width=width,
        height=height,
        fps=args.fps,
        frames=args.frames,
        warmup=args.warmup,
        fourcc=args.fourcc,
        auto_exposure=args.auto_exposure,
        exposure=args.exposure,
        gain=args.gain,
    )
    if not frames:
        print(f"Skipping {width}x{height}: no frames captured", flush=True)
        return []

    output_dir = args.capture_dir / f"matrix_{width}x{height}"
    source_image = save_source_frame(frames[-1], output_dir, width, height)
    frame_shape = "x".join(str(value) for value in frames[-1].shape)

    print(
        f"Camera reported {probe.reported_width:.0f}x{probe.reported_height:.0f}"
        f"@{probe.reported_fps:.1f}; measured {probe.measured_capture_fps:.2f} FPS",
        flush=True,
    )
    print(f"Saved source frame: {source_image}", flush=True)

    baseline = load_predictions(
        args.baseline,
        frames,
        device=device,
        conf=args.conf,
        class_filter=args.classes,
    )
    rows: list[MatrixRow] = []
    for model_path in args.models:
        print(f"Benchmarking {model_path}", flush=True)
        result = benchmark_model(
            model_path=model_path,
            frames=frames,
            baseline_predictions=baseline if model_path != args.baseline else None,
            device=device,
            conf=args.conf,
            class_filter=args.classes,
            model_warmup=args.model_warmup,
            output_dir=output_dir,
        )
        row = build_row(args, model_path, frame_shape, probe, result)
        rows.append(row)
        print(
            f"  {row.avg_ms:.2f} ms, {row.model_fps:.1f} model FPS,"
            f" detection {row.detection_rate:.2f}, confidence {row.avg_confidence:.3f}",
            flush=True,
        )
    return rows


def run_matrix(args: argparse.Namespace) -> list[MatrixRow]:
    device = choose_device("auto")
    all_rows: list[MatrixRow] = []
    args.capture_dir.mkdir(parents=True, exist_ok=True)

    for size_text in camera_sizes_from_args(args):
        try:
            all_rows.extend(benchmark_camera_size(args, size_text, device))
        except Exception as exc:
            print(f"Skipping {size_text}: {exc}", flush=True)

    return all_rows


def write_csv(path: Path, rows: list[MatrixRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(MatrixRow)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def best_detection(rows: list[MatrixRow]) -> MatrixRow:
    return max(rows, key=lambda row: (row.detection_rate, row.avg_confidence, -row.avg_ms))


def write_markdown(path: Path, rows: list[MatrixRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = {row.camera_requested: [item for item in rows if item.camera_requested == row.camera_requested] for row in rows}

    with path.open("w") as file:
        file.write("# Camera Size x Model Benchmark\n\n")
        file.write("One fresh frame set is captured per camera size, then reused for every model in that group.\n\n")
        file.write("| " + " | ".join(REPORT_COLUMNS) + " |\n")
        file.write("| " + " | ".join(["---"] * len(REPORT_COLUMNS)) + " |\n")

        ordered_rows = sorted(rows, key=lambda row: (camera_sort_key(row.camera_requested), row.avg_ms))
        for row in ordered_rows:
            data = asdict(row)
            file.write("| " + " | ".join(fmt(data[column]) for column in REPORT_COLUMNS) + " |\n")

        file.write("\n## Picks By Camera Size\n\n")
        for camera_name in sorted(grouped, key=camera_sort_key):
            camera_rows = grouped[camera_name]
            fastest = min(camera_rows, key=lambda row: row.avg_ms)
            detected = best_detection(camera_rows)
            file.write(
                f"- `{camera_name}`: fastest `{fastest.model}` "
                f"({fastest.avg_ms:.2f} ms, detection {fastest.detection_rate:.2f}); "
                f"best detection `{detected.model}` "
                f"(detection {detected.detection_rate:.2f}, confidence {detected.avg_confidence:.3f})\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark saved YOLO models across USB camera input sizes.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--camera-sizes", nargs="+", default=["auto"], help="Use WIDTHxHEIGHT values, or 'auto'.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--fourcc", choices=("YUYV", "MJPG"), default="YUYV")
    parser.add_argument("--auto-exposure", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=300.0)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--classes", type=int, nargs="+", help="Optional YOLO class IDs to keep.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--baseline", default="yolo11n.pt")
    parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/camera_size_model_matrix.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/camera_size_model_matrix.md"))
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_matrix(args)
    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"\nCSV saved: {args.output_csv}", flush=True)
    print(f"Markdown saved: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
