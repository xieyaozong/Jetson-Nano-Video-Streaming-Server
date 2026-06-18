from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import re
import signal
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, NamedTuple

import cv2
from ultralytics import YOLO


DEFAULT_CLASSES: list[int] | None = None


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionFrame:
    frame: object
    detections: list[Detection]
    fps: float
    inference_ms: float
    capture_wait_ms: float
    postprocess_ms: float


class DetectionState(NamedTuple):
    detections: list[Detection]
    inference_ms: float
    detection_fps: float
    updated_at: float


class CameraInfo(NamedTuple):
    requested_width: int
    requested_height: int
    requested_fps: int
    requested_fourcc: str | None
    reported_width: float
    reported_height: float
    reported_fps: float
    reported_fourcc: str

    def summary(self) -> str:
        requested_format = self.requested_fourcc or "camera default"
        return (
            f"camera requested={self.requested_width}x{self.requested_height}@{self.requested_fps} "
            f"fourcc={requested_format}; actual={self.reported_width:.0f}x{self.reported_height:.0f}"
            f"@{self.reported_fps:.1f} fourcc={self.reported_fourcc or 'unknown'}"
        )


class RunLogger:
    def __init__(
        self,
        log_dir: Path,
        source: int | str,
        model_path: str,
        width: int,
        height: int,
        target_fps: int,
        imgsz: int,
        conf: float,
        run_mode: str = "sync",
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = slugify(Path(model_path).stem)
        if run_mode != "sync":
            model_name = f"{model_name}-{slugify(run_mode)}"
        source_name = slugify(str(source))
        self.base_path = log_dir / f"{timestamp}_{model_name}_src-{source_name}"
        self.csv_path = self.base_path.with_suffix(".csv")
        self.summary_path = self.base_path.with_suffix(".summary.json")
        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "frame",
                "elapsed_s",
                "objects",
                "fps",
                "inference_ms",
                "capture_wait_ms",
                "postprocess_ms",
                "display_ms",
                "detections",
            ],
        )
        self._writer.writeheader()
        self._started = time.perf_counter()
        self._frames = 0
        self._objects_counts: list[int] = []
        self._fps_values: list[float] = []
        self._inference_values: list[float] = []
        self._capture_wait_values: list[float] = []
        self._display_values: list[float] = []
        self._summary_context = {
            "source": str(source),
            "model_path": model_path,
            "width": width,
            "height": height,
            "target_fps": target_fps,
            "imgsz": imgsz,
            "conf": conf,
            "run_mode": run_mode,
        }

    def log_frame(self, item: DetectionFrame, display_ms: float) -> None:
        self._frames += 1
        objects = len(item.detections)
        self._objects_counts.append(objects)
        self._fps_values.append(item.fps)
        self._inference_values.append(item.inference_ms)
        self._capture_wait_values.append(item.capture_wait_ms)
        self._display_values.append(display_ms)
        self._writer.writerow(
            {
                "frame": self._frames,
                "elapsed_s": f"{time.perf_counter() - self._started:.6f}",
                "objects": objects,
                "fps": f"{item.fps:.3f}",
                "inference_ms": f"{item.inference_ms:.3f}",
                "capture_wait_ms": f"{item.capture_wait_ms:.3f}",
                "postprocess_ms": f"{item.postprocess_ms:.3f}",
                "display_ms": f"{display_ms:.3f}",
                "detections": json.dumps([detection_to_dict(d) for d in item.detections]),
            }
        )
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        elapsed = time.perf_counter() - self._started
        steady_skip = min(5, len(self._inference_values))
        steady_inference = self._inference_values[steady_skip:]
        steady_fps = self._fps_values[steady_skip:]
        steady_capture_wait = self._capture_wait_values[steady_skip:]
        summary = {
            **self._summary_context,
            "frames": self._frames,
            "elapsed_s": elapsed,
            "overall_fps": self._frames / elapsed if elapsed else 0.0,
            "avg_objects": mean_value(self._objects_counts),
            "detected_frames": sum(1 for count in self._objects_counts if count > 0),
            "avg_fps": mean_value(self._fps_values),
            "avg_inference_ms": mean_value(self._inference_values),
            "p95_inference_ms": percentile_value(self._inference_values, 95),
            "avg_capture_wait_ms": mean_value(self._capture_wait_values),
            "avg_display_ms": mean_value(self._display_values),
            "steady_state_skip_frames": steady_skip,
            "steady_avg_fps": mean_value(steady_fps),
            "steady_avg_inference_ms": mean_value(steady_inference),
            "steady_p95_inference_ms": percentile_value(steady_inference, 95),
            "steady_avg_capture_wait_ms": mean_value(steady_capture_wait),
            "csv_path": str(self.csv_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


class LatestFrameCamera:
    """Camera reader that drops stale frames instead of building latency."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int | None = None,
        fourcc: str | None = None,
        auto_exposure: float | None = None,
        exposure: float | None = None,
        gain: float | None = None,
        buffer_size: int | None = None,
        suppress_jpeg_warning: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.fourcc = fourcc
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self.gain = gain
        self.buffer_size = buffer_size
        self.suppress_jpeg_warning = suppress_jpeg_warning
        self._frames: queue.Queue[object] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.info: CameraInfo | None = None

    def start(self) -> "LatestFrameCamera":
        self._cap = (
            cv2.VideoCapture(self.source, self.backend)
            if self.backend is not None
            else cv2.VideoCapture(self.source)
        )
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera/source: {self.source}")

        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.auto_exposure is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
        if self.exposure is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if self.gain is not None:
            self._cap.set(cv2.CAP_PROP_GAIN, self.gain)
        if self.buffer_size is not None:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self.info = CameraInfo(
            requested_width=self.width,
            requested_height=self.height,
            requested_fps=self.fps,
            requested_fourcc=self.fourcc,
            reported_width=self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            reported_height=self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            reported_fps=self._cap.get(cv2.CAP_PROP_FPS),
            reported_fourcc=_fourcc_to_string(self._cap.get(cv2.CAP_PROP_FOURCC)),
        )

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def read(self, timeout: float = 1.0) -> object:
        return self._frames.get(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def _read_loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            read_context = suppress_stderr() if self.suppress_jpeg_warning and self.fourcc == "MJPG" else contextlib.nullcontext()
            with read_context:
                ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self._frames.full():
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            self._frames.put(frame)


class SharedDetections:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = DetectionState([], 0.0, 0.0, 0.0)

    def update(self, state: DetectionState) -> None:
        with self._lock:
            self._state = state

    def get(self) -> DetectionState:
        with self._lock:
            return self._state


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def select_usb_fourcc(width: int, height: int, requested: str | None, source: int | str) -> str | None:
    if requested in (None, "", "auto"):
        if not isinstance(source, int):
            return None
        return "MJPG" if width >= 1280 and height >= 720 else "YUYV"
    return requested


def _fourcc_to_string(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    chars = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return "".join(char if 32 <= ord(char) <= 126 else "?" for char in chars).strip()


def iter_object_detections(
    source: int | str = 0,
    model_path: str = "yolo11n.pt",
    device: str = "auto",
    backend: int | None = None,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    imgsz: int = 640,
    conf: float = 0.35,
    iou: float = 0.45,
    class_filter: list[int] | None = DEFAULT_CLASSES,
    fourcc: str | None = None,
    auto_exposure: float | None = None,
    exposure: float | None = None,
    gain: float | None = None,
    buffer_size: int | None = None,
    suppress_jpeg_warning: bool = True,
) -> Generator[DetectionFrame, None, None]:
    model = YOLO(model_path, task="detect")
    selected_device = choose_device(device)
    camera_fourcc = select_usb_fourcc(width, height, fourcc, source)
    camera = LatestFrameCamera(
        source=source,
        width=width,
        height=height,
        fps=target_fps,
        backend=backend,
        fourcc=camera_fourcc,
        auto_exposure=auto_exposure,
        exposure=exposure,
        gain=gain,
        buffer_size=buffer_size,
        suppress_jpeg_warning=suppress_jpeg_warning,
    ).start()
    if camera.info is not None:
        print(camera.info.summary(), flush=True)
        if camera.info.reported_fps + 0.5 < target_fps:
            print(
                "Warning: actual camera FPS is below target. For 720p USB input, use --fourcc MJPG.",
                flush=True,
            )
    last_yield = time.perf_counter()
    first_frame = True

    try:
        while True:
            read_start = time.perf_counter()
            frame = camera.read(timeout=5.0 if first_frame else 2.0)
            capture_wait_ms = (time.perf_counter() - read_start) * 1000.0
            first_frame = False
            start = time.perf_counter()
            results = model.predict(
                frame,
                classes=class_filter,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=selected_device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0

            now = time.perf_counter()
            fps = 1.0 / max(now - last_yield, 1e-6)
            last_yield = now
            postprocess_start = time.perf_counter()
            detections = boxes_to_detections(results[0].boxes, results[0].names)
            annotated = draw_detection_overlay(frame, detections, inference_ms, display_fps=fps)
            postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

            yield DetectionFrame(
                frame=annotated,
                detections=detections,
                fps=fps,
                inference_ms=inference_ms,
                capture_wait_ms=capture_wait_ms,
                postprocess_ms=postprocess_ms,
            )
    finally:
        camera.stop()


def run_display(
    source: int | str = 0,
    model_path: str = "yolo11n.pt",
    device: str = "auto",
    backend: int | None = None,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    imgsz: int = 640,
    conf: float = 0.35,
    iou: float = 0.45,
    class_filter: list[int] | None = DEFAULT_CLASSES,
    fourcc: str | None = None,
    auto_exposure: float | None = None,
    exposure: float | None = None,
    gain: float | None = None,
    buffer_size: int | None = None,
    window_name: str = "YOLO object detector",
    fullscreen: bool = False,
    log_dir: Path | None = Path("logs"),
    suppress_gtk_warning: bool = True,
    suppress_jpeg_warning: bool = True,
) -> None:
    stopped = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    window_context = suppress_stderr() if suppress_gtk_warning else contextlib.nullcontext()
    with window_context:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger = (
        RunLogger(
            log_dir=log_dir,
            source=source,
            model_path=model_path,
            width=width,
            height=height,
            target_fps=target_fps,
            imgsz=imgsz,
            conf=conf,
            run_mode="sync",
        )
        if log_dir is not None
        else None
    )
    if logger is not None:
        print(f"Logging run to {logger.csv_path}")

    try:
        for item in iter_object_detections(
            source=source,
            model_path=model_path,
            device=device,
            backend=backend,
            width=width,
            height=height,
            target_fps=target_fps,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            class_filter=class_filter,
            fourcc=fourcc,
            auto_exposure=auto_exposure,
            exposure=exposure,
            gain=gain,
            buffer_size=buffer_size,
            suppress_jpeg_warning=suppress_jpeg_warning,
        ):
            print(
                f"objects={len(item.detections)} fps={item.fps:.1f} "
                f"inference={item.inference_ms:.1f}ms "
                f"wait={item.capture_wait_ms:.1f}ms",
                end="\r",
                flush=True,
            )
            display_start = time.perf_counter()
            cv2.imshow(window_name, item.frame)
            key = cv2.waitKey(1) & 0xFF
            display_ms = (time.perf_counter() - display_start) * 1000.0
            if logger is not None:
                logger.log_frame(item, display_ms=display_ms)
            if stopped or key in (ord("q"), 27):
                break
    finally:
        if logger is not None:
            logger.close()
            print(f"\nSaved log: {logger.csv_path}")
            print(f"Saved summary: {logger.summary_path}")

    cv2.destroyAllWindows()
    print()


def run_async_display(
    source: int | str = 0,
    model_path: str = "yolo11n.pt",
    device: str = "auto",
    backend: int | None = None,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    imgsz: int = 640,
    conf: float = 0.35,
    iou: float = 0.45,
    class_filter: list[int] | None = DEFAULT_CLASSES,
    fourcc: str | None = None,
    auto_exposure: float | None = None,
    exposure: float | None = None,
    gain: float | None = None,
    buffer_size: int | None = None,
    window_name: str = "YOLO object detector",
    fullscreen: bool = False,
    log_dir: Path | None = Path("logs"),
    suppress_gtk_warning: bool = True,
    suppress_jpeg_warning: bool = True,
) -> None:
    stopped = threading.Event()
    shared_detections = SharedDetections()
    latest_frame: queue.Queue[object] = queue.Queue(maxsize=1)
    selected_device = choose_device(device)
    camera_fourcc = select_usb_fourcc(width, height, fourcc, source)
    camera = LatestFrameCamera(
        source=source,
        width=width,
        height=height,
        fps=target_fps,
        backend=backend,
        fourcc=camera_fourcc,
        auto_exposure=auto_exposure,
        exposure=exposure,
        gain=gain,
        buffer_size=buffer_size,
        suppress_jpeg_warning=suppress_jpeg_warning,
    ).start()
    if camera.info is not None:
        print(camera.info.summary(), flush=True)
        if camera.info.reported_fps + 0.5 < target_fps:
            print(
                "Warning: actual camera FPS is below target. For 720p USB input, use --fourcc MJPG.",
                flush=True,
            )

    def _stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def inference_loop() -> None:
        model = YOLO(model_path, task="detect")
        last_done = time.perf_counter()
        while not stopped.is_set():
            try:
                frame = latest_frame.get(timeout=0.2)
            except queue.Empty:
                continue
            start = time.perf_counter()
            results = model.predict(
                frame,
                classes=class_filter,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=selected_device,
                verbose=False,
            )
            finished = time.perf_counter()
            detections = boxes_to_detections(results[0].boxes, results[0].names)
            shared_detections.update(
                DetectionState(
                    detections=detections,
                    inference_ms=(finished - start) * 1000.0,
                    detection_fps=1.0 / max(finished - last_done, 1e-6),
                    updated_at=finished,
                )
            )
            last_done = finished

    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    logger = (
        RunLogger(
            log_dir=log_dir,
            source=source,
            model_path=model_path,
            width=width,
            height=height,
            target_fps=target_fps,
            imgsz=imgsz,
            conf=conf,
            run_mode="async",
        )
        if log_dir is not None
        else None
    )
    if logger is not None:
        print(f"Logging run to {logger.csv_path}")

    window_context = suppress_stderr() if suppress_gtk_warning else contextlib.nullcontext()
    with window_context:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    display_last = time.perf_counter()
    first_frame = True
    frame_number = 0
    try:
        while not stopped.is_set():
            read_start = time.perf_counter()
            frame = camera.read(timeout=5.0 if first_frame else 2.0)
            capture_wait_ms = (time.perf_counter() - read_start) * 1000.0
            first_frame = False
            frame_number += 1

            if latest_frame.full():
                try:
                    latest_frame.get_nowait()
                except queue.Empty:
                    pass
            latest_frame.put(frame.copy())

            now = time.perf_counter()
            display_fps = 1.0 / max(now - display_last, 1e-6)
            display_last = now
            state = shared_detections.get()
            annotated = draw_detection_overlay(
                frame,
                state.detections,
                state.inference_ms,
                display_fps=display_fps,
                detection_fps=state.detection_fps,
            )

            print(
                f"objects={len(state.detections)} display_fps={display_fps:.1f} "
                f"detect_fps={state.detection_fps:.1f} inference={state.inference_ms:.1f}ms "
                f"wait={capture_wait_ms:.1f}ms",
                end="\r",
                flush=True,
            )
            display_start = time.perf_counter()
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            display_ms = (time.perf_counter() - display_start) * 1000.0
            if logger is not None:
                logger.log_frame(
                    DetectionFrame(
                        frame=annotated,
                        detections=state.detections,
                        fps=display_fps,
                        inference_ms=state.inference_ms,
                        capture_wait_ms=capture_wait_ms,
                        postprocess_ms=0.0,
                    ),
                    display_ms=display_ms,
                )
            if key in (ord("q"), 27):
                stopped.set()
    finally:
        stopped.set()
        camera.stop()
        worker.join(timeout=2.0)
        if logger is not None:
            logger.close()
            print(f"\nSaved log: {logger.csv_path}")
            print(f"Saved summary: {logger.summary_path}")
        cv2.destroyAllWindows()
        print()


def boxes_to_detections(boxes: object, names: dict[int, str]) -> list[Detection]:
    detections: list[Detection] = []
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                class_id=class_id,
                label=names.get(class_id, str(class_id)),
                confidence=float(box.conf[0].item()),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return detections


def draw_detection_overlay(
    frame: object,
    detections: Iterable[Detection],
    inference_ms: float,
    display_fps: float | None = None,
    detection_fps: float | None = None,
) -> object:
    annotated = frame.copy()
    objects = 0
    for detection in detections:
        objects += 1
        x1, y1, x2, y2 = detection.xyxy
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 220, 30), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )

    status = f"objects: {objects}  inference: {inference_ms:.1f} ms"
    if display_fps is not None:
        status += f"  display: {display_fps:.1f} fps"
    if detection_fps is not None:
        status += f"  detect: {detection_fps:.1f} fps"
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


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


@contextlib.contextmanager
def suppress_stderr() -> Generator[None, None, None]:
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def detection_to_dict(detection: Detection) -> dict[str, object]:
    return {
        "class_id": detection.class_id,
        "label": detection.label,
        "confidence": detection.confidence,
        "xyxy": detection.xyxy,
    }


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "source"


def mean_value(values: list[float] | list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def percentile_value(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((pct / 100) * (len(ordered) - 1)), len(ordered) - 1)
    return float(ordered[index])


def build_jetson_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1 sync=false"
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="30 FPS YOLO object detection from a camera.")
    parser.add_argument("--source", default="0", help="Camera index, video file, RTSP URL, or GStreamer pipeline.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path or Ultralytics model name.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or another Ultralytics device string.")
    parser.add_argument("--backend", choices=("auto", "default", "gstreamer", "v4l2"), default="auto")
    parser.add_argument("--jetson-csi", action="store_true", help="Use a Jetson CSI camera through nvarguscamerasrc.")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        choices=("auto", "MJPG", "YUYV"),
        default="auto",
        help="USB camera pixel format for V4L2. auto uses MJPG for 720p or larger, YUYV below 720p.",
    )
    parser.add_argument("--auto-exposure", type=float, help="OpenCV CAP_PROP_AUTO_EXPOSURE value. V4L2 usually uses 1=manual, 3=auto.")
    parser.add_argument("--exposure", type=float, help="OpenCV CAP_PROP_EXPOSURE value.")
    parser.add_argument("--gain", type=float, help="OpenCV CAP_PROP_GAIN value.")
    parser.add_argument("--buffer-size", type=int, help="Optional OpenCV CAP_PROP_BUFFERSIZE. Leave unset for this USB camera; buffer-size 1 halves FPS here.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Optional YOLO class IDs to keep. Omit this to detect every class the model supports.",
    )
    parser.add_argument("--window-name", default="YOLO object detector")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--show-gtk-warning", action="store_true")
    parser.add_argument("--show-jpeg-warning", action="store_true", help="Show libjpeg warnings from MJPG USB camera decode.")
    parser.add_argument("--async-display", action="store_true", help="Display camera frames continuously while YOLO runs on a worker thread.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    backend = parse_backend(args.backend, source)
    if args.jetson_csi:
        source = build_jetson_csi_pipeline(
            sensor_id=args.sensor_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip_method,
        )
        backend = cv2.CAP_GSTREAMER

    runner = run_async_display if args.async_display else run_display
    runner(
        source=source,
        model_path=args.model,
        device=args.device,
        backend=backend,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_filter=args.classes,
        fourcc=args.fourcc,
        auto_exposure=args.auto_exposure,
        exposure=args.exposure,
        gain=args.gain,
        buffer_size=args.buffer_size,
        window_name=args.window_name,
        fullscreen=args.fullscreen,
        log_dir=None if args.no_log else args.log_dir,
        suppress_gtk_warning=not args.show_gtk_warning,
        suppress_jpeg_warning=not args.show_jpeg_warning,
    )


if __name__ == "__main__":
    main()
