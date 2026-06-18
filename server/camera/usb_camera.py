from __future__ import annotations

from dataclasses import dataclass
from server.camera.camera_source import CameraInfo

import contextlib
import os

import cv2


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_backend(value: str, source: int | str) -> int | None:
    if value == "gstreamer":
        return cv2.CAP_GSTREAMER
    if value == "v4l2":
        return cv2.CAP_V4L2
    if value == "default":
        return None
    if isinstance(source, str) and "!" in source:
        return cv2.CAP_GSTREAMER
    return None


def select_fourcc(width: int, height: int, requested: str | None, source: int | str) -> str | None:
    if requested in (None, "", "auto"):
        if not isinstance(source, int):
            return None
        return "MJPG" if width >= 1280 and height >= 720 else "YUYV"
    return requested


def fourcc_to_string(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    chars = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return "".join(char if 32 <= ord(char) <= 126 else "?" for char in chars).strip()


@dataclass
class UsbCamera:
    source: int | str = 0
    backend: int | None = cv2.CAP_V4L2
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str | None = "MJPG"
    suppress_jpeg_warning: bool = True

    def __post_init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source, self.backend) if self.backend is not None else cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> object:
        if self._cap is None:
            raise RuntimeError("Camera source is not open.")
        read_context = suppress_stderr() if self.suppress_jpeg_warning and self.fourcc == "MJPG" else contextlib.nullcontext()
        with read_context:
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Could not read frame from camera.")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def info(self) -> CameraInfo:
        if self._cap is None:
            return CameraInfo(0.0, 0.0, 0.0, "")
        return CameraInfo(
            width=self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            fourcc=fourcc_to_string(self._cap.get(cv2.CAP_PROP_FOURCC)),
        )


@contextlib.contextmanager
def suppress_stderr() -> object:
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
