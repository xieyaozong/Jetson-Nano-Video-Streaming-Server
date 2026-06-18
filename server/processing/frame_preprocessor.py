from __future__ import annotations

import cv2


def resize_frame(frame: object, width: int | None = None, height: int | None = None) -> object:
    if width is None or height is None:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

