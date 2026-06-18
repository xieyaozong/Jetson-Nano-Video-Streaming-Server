from __future__ import annotations

from pathlib import Path

import argparse
import sys
import time

# Direct script execution starts from scripts/, so add the project root first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.camera.usb_camera import UsbCamera, parse_backend, parse_source, select_fourcc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure raw camera capture FPS.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--backend", choices=("auto", "default", "gstreamer", "v4l2"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", choices=("auto", "MJPG", "YUYV"), default="auto")
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    camera = UsbCamera(
        source=source,
        backend=parse_backend(args.backend, source),
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=select_fourcc(args.width, args.height, args.fourcc, source),
    )
    camera.open()
    frames = 0
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.seconds:
            camera.read()
            frames += 1
    finally:
        camera.close()

    elapsed = time.perf_counter() - started
    print(f"frames={frames}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"capture_fps={frames / elapsed if elapsed else 0.0:.2f}")


if __name__ == "__main__":
    main()
