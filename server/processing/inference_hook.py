from __future__ import annotations

from server.processing.overlay import Detection


class InferenceHook:
    def process(self, frame: object) -> list[Detection]:
        return []


class YoloInferenceHook(InferenceHook):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.device = _select_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes

    def process(self, frame: object) -> list[Detection]:
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=result.names.get(class_id, str(class_id)),
                    confidence=float(box.conf[0].item()),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections


def build_inference_hook(
    enabled: bool,
    model_path: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    classes: list[int] | None,
) -> InferenceHook:
    if not enabled:
        return InferenceHook()
    return YoloInferenceHook(model_path, device=device, imgsz=imgsz, conf=conf, iou=iou, classes=classes)


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

