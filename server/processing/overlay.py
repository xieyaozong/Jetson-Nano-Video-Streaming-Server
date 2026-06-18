from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


def draw_overlay(
    frame: object,
    source_fps: float,
    encoded_fps: float,
    detections: Iterable[Detection] = (),
    inference_enabled: bool = False,
) -> object:
    annotated = frame.copy()
    count = 0
    for detection in detections:
        count += 1
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 220, 30), 2)
        cv2.putText(
            annotated,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )

    mode = "inference:on" if inference_enabled else "inference:off"
    status = f"source {source_fps:.1f} fps | stream {encoded_fps:.1f} fps | {mode} | detections {count}"
    cv2.putText(
        annotated,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
