from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from object_stream.live_view import (
    boxes_to_detections,
    choose_device,
    draw_detection_overlay,
    parse_backend,
    parse_source,
    select_usb_fourcc,
)


def capture_frame(
    source: int | str,
    backend: int | None,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None,
    warmup: int,
) -> object:
    cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/source: {source}")

    try:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        frame = None
        for _ in range(max(warmup, 1)):
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

        if frame is None:
            raise RuntimeError("Camera opened, but no frame was captured.")

        return frame
    finally:
        cap.release()


def annotate_frame(
    frame: object,
    model_path: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    class_filter: list[int] | None,
) -> object:
    model = YOLO(model_path, task="detect")
    started = time.perf_counter()
    result = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=class_filter,
        device=choose_device(device),
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - started) * 1000.0
    detections = boxes_to_detections(result.boxes, result.names)
    return draw_detection_overlay(frame, detections, inference_ms=inference_ms)


def write_image(path: Path, frame: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not write image: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one public-safe annotated camera image for README/GitHub pages.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--backend", choices=("auto", "default", "gstreamer", "v4l2"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", choices=("auto", "MJPG", "YUYV"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", type=int, nargs="+", help="Optional YOLO class IDs to keep.")
    parser.add_argument("--raw", action="store_true", help="Save the raw camera frame instead of an annotated image.")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/desk-detections.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    backend = parse_backend(args.backend, source)
    fourcc = select_usb_fourcc(args.width, args.height, args.fourcc, source)
    frame = capture_frame(
        source=source,
        backend=backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=fourcc,
        warmup=args.warmup,
    )
    if not args.raw:
        frame = annotate_frame(
            frame=frame,
            model_path=args.model,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            class_filter=args.classes,
        )
    output = write_image(args.output, frame)
    print(f"Saved image: {output}")


if __name__ == "__main__":
    main()
