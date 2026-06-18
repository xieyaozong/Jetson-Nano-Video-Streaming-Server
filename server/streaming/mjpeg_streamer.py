from __future__ import annotations

import cv2


def encode_jpeg(frame: object, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    return encoded.tobytes()

