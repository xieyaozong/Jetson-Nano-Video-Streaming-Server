from __future__ import annotations

from dataclasses import dataclass
from server.camera.usb_camera import UsbCamera


@dataclass
class VideoFileSource(UsbCamera):
    source: str = "sample_data/sample.mp4"
    backend: int | None = None
    fourcc: str | None = None
