from __future__ import annotations

from dataclasses import dataclass
from server.camera.usb_camera import UsbCamera

import cv2


@dataclass
class RtspSource(UsbCamera):
    source: str = "rtsp://example.local/stream"
    backend: int | None = None
    fourcc: str | None = None
