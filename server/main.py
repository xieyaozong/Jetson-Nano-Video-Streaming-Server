from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from server.camera.camera_source import CameraSource
from server.camera.rtsp_source import RtspSource
from server.camera.usb_camera import UsbCamera, parse_backend, parse_source, select_fourcc
from server.camera.video_file_source import VideoFileSource
from server.config import ServerConfig, parse_args
from server.processing.inference_hook import build_inference_hook
from server.processing.overlay import draw_overlay
from server.streaming.frame_buffer import FrameBuffer, StreamStats
from server.streaming.mjpeg_streamer import encode_jpeg
from server.utils.device_monitor import system_snapshot
from server.utils.fps_counter import FpsCounter
from server.utils.logger import configure_logging

import cv2
import json
import logging
import signal
import threading
import time

LOGGER = logging.getLogger(__name__)


class StreamRunner:
    def __init__(self, config: ServerConfig, buffer: FrameBuffer) -> None:
        self.config = config
        self.buffer = buffer
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None
        self.source = build_source(config)
        self.inference = build_inference_hook(
            enabled=config.enable_inference,
            model_path=config.model,
            device=config.device,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            classes=config.classes,
        )

    def start(self) -> None:
        self.source.open()
        info = self.source.info()
        LOGGER.info("source opened: %.0fx%.0f@%.1f fourcc=%s", info.width, info.height, info.fps, info.fourcc)
        self.thread = threading.Thread(target=self._run, name="stream-runner", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.source.close()

    def _run(self) -> None:
        sequence = 0
        source_fps = FpsCounter()
        encoded_fps = FpsCounter()

        while not self.stopped.is_set():
            try:
                frame = self.source.read()
            except RuntimeError as exc:
                LOGGER.warning("frame read failed: %s", exc)
                time.sleep(0.05)
                continue

            current_source_fps = source_fps.tick()
            detections = self.inference.process(frame)
            current_encoded_fps = encoded_fps.tick()
            output = (
                draw_overlay(
                    frame,
                    source_fps=current_source_fps,
                    encoded_fps=current_encoded_fps,
                    detections=detections,
                    inference_enabled=self.config.enable_inference,
                )
                if self.config.show_overlay
                else frame
            )

            try:
                jpeg = encode_jpeg(output, quality=self.config.jpeg_quality)
            except RuntimeError as exc:
                LOGGER.warning("jpeg encode failed: %s", exc)
                continue

            sequence += 1
            height, width = output.shape[:2]
            self.buffer.publish(
                jpeg,
                StreamStats(
                    sequence=sequence,
                    encoded_fps=current_encoded_fps,
                    source_fps=current_source_fps,
                    width=width,
                    height=height,
                    clients=self.buffer.client_count(),
                    inference_enabled=self.config.enable_inference,
                    detections=len(detections),
                ),
            )


class StreamHandler(BaseHTTPRequestHandler):
    server: "StreamingHttpServer"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_index()
        elif self.path == "/video.mjpg":
            self._send_mjpeg()
        elif self.path == "/snapshot.jpg":
            self._send_snapshot()
        elif self.path == "/status.json":
            self._send_status()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.debug(fmt, *args)

    def _send_index(self) -> None:
        body = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jetson Streaming Server</title>
  <style>
    html, body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    main { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { padding: 10px 14px; background: #1d1d1d; display: flex; justify-content: space-between; gap: 12px; }
    img { width: 100%; height: calc(100vh - 44px); object-fit: contain; background: #000; }
    code { color: #9cdcfe; }
  </style>
</head>
<body>
<main>
  <header>
    <div>Jetson Streaming Server</div>
    <div><code>/video.mjpg</code></div>
  </header>
  <img src="/video.mjpg" alt="Jetson video stream">
</main>
</body>
</html>
"""
        self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

    def _send_status(self) -> None:
        data = {
            "stream": self.server.buffer.stats().to_dict(),
            "device": system_snapshot(),
        }
        self._send_bytes(json.dumps(data, indent=2).encode("utf-8") + b"\n", "application/json")

    def _send_snapshot(self) -> None:
        jpeg = self.server.buffer.snapshot()
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
        self.server.buffer.add_client()
        try:
            while not self.server.stopped.is_set():
                frame = self.server.buffer.wait_for_frame(last_sequence)
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
            self.server.buffer.remove_client()

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StreamingHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], buffer: FrameBuffer) -> None:
        super().__init__(address, StreamHandler)
        self.buffer = buffer
        self.stopped = threading.Event()


def build_source(config: ServerConfig) -> CameraSource:
    source = parse_source(config.source)
    backend = parse_backend(config.backend, source)
    fourcc = select_fourcc(config.width, config.height, config.fourcc, source)

    if config.source_type == "rtsp":
        return RtspSource(source=str(config.source))
    if config.source_type == "file":
        return VideoFileSource(source=str(config.source))
    return UsbCamera(
        source=source,
        backend=backend if backend is not None else cv2.CAP_V4L2,
        width=config.width,
        height=config.height,
        fps=config.fps,
        fourcc=fourcc,
    )


def main() -> None:
    configure_logging()
    config = parse_args()
    buffer = FrameBuffer()
    runner = StreamRunner(config, buffer)
    server = StreamingHttpServer((config.host, config.port), buffer)

    def stop(_signum: int, _frame: object) -> None:
        LOGGER.info("shutting down")
        server.stopped.set()
        runner.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    runner.start()
    LOGGER.info("open http://%s:%s/", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        server.stopped.set()
        runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
