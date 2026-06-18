from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CameraInfo:
    width: float
    height: float
    fps: float
    fourcc: str


class CameraSource(Protocol):
    def open(self) -> None:
        """Open the underlying camera or stream."""

    def read(self) -> object:
        """Return the next frame or raise RuntimeError."""

    def close(self) -> None:
        """Release the source."""

    def info(self) -> CameraInfo:
        """Return source metadata reported by OpenCV."""

