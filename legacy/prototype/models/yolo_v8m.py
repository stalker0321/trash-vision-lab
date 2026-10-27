"""YOLOv8m wrapper."""

from __future__ import annotations

from typing import Any, Dict

from legacy.prototype.datasets.yolo_datayaml import build_data_yaml
from .base import Model, TrainRequest
from .registry import register


@register("yolo_v8m")
class YoloV8m(Model):
    def train(self, req: TrainRequest) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise SystemExit("Ultralytics not installed. Run: pip install ultralytics") from exc

        data_yaml = build_data_yaml(req.version)

        cfg: Dict[str, Any] = req.config or {}
        yolo_cfg = cfg.get("yolo", {}) if isinstance(cfg.get("yolo"), dict) else {}

        weights = yolo_cfg.get("weights", "yolov8m.pt")
        imgsz = yolo_cfg.get("imgsz", 640)
        batch = yolo_cfg.get("batch", 8)
        epochs = yolo_cfg.get("epochs", 1)
        device = yolo_cfg.get("device", None)
        project = yolo_cfg.get("project", "runs/yolo")
        name = yolo_cfg.get("name", f"{self.name}-{req.version}")
        cache = yolo_cfg.get("cache", False)
        workers = yolo_cfg.get("workers", None)

        model = YOLO(weights)

        train_args: Dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": imgsz,
            "batch": batch,
            "epochs": epochs,
            "project": project,
            "name": name,
            "cache": cache,
        }
        if device is not None:
            train_args["device"] = device
        if workers is not None:
            train_args["workers"] = workers

        model.train(**train_args)
