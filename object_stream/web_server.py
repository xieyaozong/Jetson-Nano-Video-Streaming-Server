from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from object_stream.live_view import (
    DetectionState,
    LatestFrameCamera,
    SharedDetections,
    draw_detection_overlay,
    parse_backend,
    parse_source,
    boxes_to_detections,
    build_jetson_csi_pipeline,
    choose_device,
    select_usb_fourcc,
)


@dataclass(frozen=True)
class StreamStats:
    sequence: int
    encoded_fps: float
    detection_fps: float
    inference_ms: float
    objects: int
    clients: int
    frame_age_ms: float
    width: int
    height: int
    camera_width: float
    camera_height: float
    camera_fps: float
    camera_fourcc: str


class FrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._stats = StreamStats(0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0, "")
        self._clients = 0

    def publish(self, jpeg: bytes, stats: StreamStats) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._stats = stats
            self._condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 5.0) -> tuple[int, bytes] | None:
        with self._condition:
            ok = self._condition.wait_for(
                lambda: self._jpeg is not None and self._stats.sequence != last_sequence,
                timeout=timeout,
            )
            if not ok or self._jpeg is None:
                return None
            return self._stats.sequence, self._jpeg

    def snapshot(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def stats(self) -> StreamStats:
        with self._condition:
            return self._stats

    def add_client(self) -> None:
        with self._condition:
            self._clients += 1

    def remove_client(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)

    def client_count(self) -> int:
        with self._condition:
            return self._clients


class StreamRunner:
    def __init__(self, args: argparse.Namespace, store: FrameStore) -> None:
        self.args = args
        self.store = store
        self.stopped = threading.Event()
        self.latest_frame: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.detections = SharedDetections()
        self.camera: LatestFrameCamera | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        source = parse_source(self.args.source)
        backend = parse_backend(self.args.backend, source)
        if self.args.jetson_csi:
            source = build_jetson_csi_pipeline(
                sensor_id=self.args.sensor_id,
                width=self.args.width,
                height=self.args.height,
                fps=self.args.fps,
                flip_method=self.args.flip_method,
            )
            backend = cv2.CAP_GSTREAMER

        camera_fourcc = select_usb_fourcc(self.args.width, self.args.height, self.args.fourcc, source)
        self.camera = LatestFrameCamera(
            source=source,
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
            backend=backend,
            fourcc=camera_fourcc,
            auto_exposure=self.args.auto_exposure,
            exposure=self.args.exposure,
            gain=self.args.gain,
            buffer_size=self.args.buffer_size,
            suppress_jpeg_warning=not self.args.show_jpeg_warning,
        ).start()
        if self.camera.info is not None:
            print(self.camera.info.summary(), flush=True)
            if self.camera.info.reported_fps + 0.5 < self.args.fps:
                print(
                    "Warning: actual camera FPS is below target. For 720p USB input, use --fourcc MJPG.",
                    flush=True,
                )

        self._threads = [
            threading.Thread(target=self._inference_loop, name="inference", daemon=True),
            threading.Thread(target=self._encode_loop, name="encoder", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.camera is not None:
            self.camera.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)

    def _put_latest_frame(self, frame: object) -> None:
        if self.latest_frame.full():
            try:
                self.latest_frame.get_nowait()
            except queue.Empty:
                pass
        self.latest_frame.put(frame)

    def _inference_loop(self) -> None:
        model = YOLO(self.args.model, task="detect")
        device = choose_device(self.args.device)
        last_done = time.perf_counter()

        while not self.stopped.is_set():
            try:
                frame = self.latest_frame.get(timeout=0.2)
            except queue.Empty:
                continue

            started = time.perf_counter()
            result = model.predict(
                frame,
                classes=self.args.classes,
                conf=self.args.conf,
                iou=self.args.iou,
                imgsz=self.args.imgsz,
                device=device,
                verbose=False,
            )[0]
            finished = time.perf_counter()
            self.detections.update(
                DetectionState(
                    detections=boxes_to_detections(result.boxes, result.names),
                    inference_ms=(finished - started) * 1000.0,
                    detection_fps=1.0 / max(finished - last_done, 1e-6),
                    updated_at=finished,
                )
            )
            last_done = finished

    def _encode_loop(self) -> None:
        assert self.camera is not None
        sequence = 0
        last_publish = time.perf_counter()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]

        while not self.stopped.is_set():
            try:
                frame = self.camera.read(timeout=2.0)
            except queue.Empty:
                continue

            self._put_latest_frame(frame.copy())
            state = self.detections.get()
            now = time.perf_counter()
            encoded_fps = 1.0 / max(now - last_publish, 1e-6)
            last_publish = now

            annotated = draw_detection_overlay(
                frame,
                state.detections,
                state.inference_ms,
                display_fps=encoded_fps,
                detection_fps=state.detection_fps,
            )
            ok, encoded = cv2.imencode(".jpg", annotated, encode_params)
            if not ok:
                continue

            sequence += 1
            height, width = annotated.shape[:2]
            camera_info = self.camera.info
            stats = StreamStats(
                sequence=sequence,
                encoded_fps=encoded_fps,
                detection_fps=state.detection_fps,
                inference_ms=state.inference_ms,
                objects=len(state.detections),
                clients=self.store.client_count(),
                frame_age_ms=(time.perf_counter() - now) * 1000.0,
                width=width,
                height=height,
                camera_width=camera_info.reported_width if camera_info is not None else 0.0,
                camera_height=camera_info.reported_height if camera_info is not None else 0.0,
                camera_fps=camera_info.reported_fps if camera_info is not None else 0.0,
                camera_fourcc=camera_info.reported_fourcc if camera_info is not None else "",
            )
            self.store.publish(encoded.tobytes(), stats)


class StreamHandler(BaseHTTPRequestHandler):
    server: "StreamServer"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/video.mjpg":
            self._send_mjpeg()
        elif self.path == "/snapshot.jpg":
            self._send_snapshot()
        elif self.path == "/status.json":
            self._send_status()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        if not self.server.quiet:
            super().log_message(fmt, *args)

    def _send_html(self) -> None:
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Object Stream</title>
  <style>
    html, body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    main {{ min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }}
    header {{ padding: 10px 14px; background: #1d1d1d; display: flex; justify-content: space-between; gap: 12px; }}
    img {{ width: 100%; height: calc(100vh - 44px); object-fit: contain; background: #000; }}
    code {{ color: #9cdcfe; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>YOLO Object Stream</div>
    <div><code>/video.mjpg</code></div>
  </header>
  <img src="/video.mjpg" alt="YOLO object stream">
</main>
</body>
</html>
"""
        self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

    def _send_status(self) -> None:
        data = json.dumps(asdict(self.server.store.stats()), indent=2).encode("utf-8")
        self._send_bytes(data + b"\n", "application/json")

    def _send_snapshot(self) -> None:
        jpeg = self.server.store.snapshot()
        if jpeg is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame is ready yet.")
            return
        self._send_bytes(jpeg, "image/jpeg")

    def _send_mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_sequence = 0
        self.server.store.add_client()
        try:
            while not self.server.stopped.is_set():
                frame = self.server.store.wait_for_frame(last_sequence)
                if frame is None:
                    continue
                last_sequence, jpeg = frame
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.store.remove_client()

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StreamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: FrameStore, quiet: bool) -> None:
        super().__init__(address, StreamHandler)
        self.store = store
        self.quiet = quiet
        self.stopped = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the YOLO object livestream to browser clients.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=80)

    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/yolo11n-320-trt-fp16.engine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("auto", "default", "gstreamer", "v4l2"), default="auto")
    parser.add_argument("--jetson-csi", action="store_true")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        choices=("auto", "MJPG", "YUYV"),
        default="auto",
        help="USB camera pixel format. auto uses MJPG for 720p or larger, YUYV below 720p.",
    )
    parser.add_argument("--auto-exposure", type=float)
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--show-jpeg-warning", action="store_true", help="Show libjpeg warnings from MJPG USB camera decode.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Optional YOLO class IDs to keep. Omit this to detect every class the model supports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FrameStore()
    runner = StreamRunner(args, store)
    server = StreamServer((args.host, args.port), store, quiet=args.quiet)

    def shutdown_server() -> None:
        server.shutdown()

    def stop(_signum: int, _frame: object) -> None:
        server.stopped.set()
        runner.stop()
        threading.Thread(target=shutdown_server, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    runner.start()
    print(f"Open http://{args.host}:{args.port}/ from another device on the same network.")
    try:
        server.serve_forever()
    finally:
        server.stopped.set()
        runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
