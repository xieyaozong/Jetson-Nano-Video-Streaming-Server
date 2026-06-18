from __future__ import annotations

from dataclasses import dataclass

import argparse
import os


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    source_type: str
    source: str
    backend: str
    width: int
    height: int
    fps: int
    fourcc: str
    jpeg_quality: int
    show_overlay: bool
    enable_inference: bool
    model: str
    device: str
    imgsz: int
    conf: float
    iou: float
    classes: list[int] | None


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Jetson Nano real-time video streaming server.")
    parser.add_argument("--host", default=os.getenv("STREAM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("STREAM_PORT", "8080")))
    parser.add_argument("--source-type", choices=("usb", "rtsp", "file"), default=os.getenv("SOURCE_TYPE", "usb"))
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"))
    parser.add_argument("--backend", choices=("auto", "default", "gstreamer", "v4l2"), default=os.getenv("CAMERA_BACKEND", "auto"))
    parser.add_argument("--width", type=int, default=int(os.getenv("FRAME_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.getenv("FRAME_HEIGHT", "720")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FRAME_FPS", "30")))
    parser.add_argument("--fourcc", choices=("auto", "MJPG", "YUYV"), default=os.getenv("CAMERA_FOURCC", "auto"))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.getenv("JPEG_QUALITY", "80")))
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--enable-inference", action="store_true")
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "auto"))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("YOLO_IMGSZ", "640")))
    parser.add_argument("--conf", type=float, default=float(os.getenv("YOLO_CONF", "0.25")))
    parser.add_argument("--iou", type=float, default=float(os.getenv("YOLO_IOU", "0.45")))
    parser.add_argument("--classes", type=int, nargs="+", help="Optional model class IDs to keep.")
    args = parser.parse_args()

    return ServerConfig(
        host=args.host,
        port=args.port,
        source_type=args.source_type,
        source=args.source,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        jpeg_quality=args.jpeg_quality,
        show_overlay=not args.no_overlay,
        enable_inference=args.enable_inference,
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
    )
