"""RT-DETRv2-m wrapper."""

from __future__ import annotations

from typing import Any, Dict

from legacy.prototype.datasets.yolo_datayaml import build_data_yaml
from .base import Model, TrainRequest
from .registry import register


@register("rtdetrv2_m")
class RTDETRv2M(Model):
    def train(self, req: TrainRequest) -> None:
        try:
            from ultralytics import RTDETR  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise SystemExit("Ultralytics not installed. Run: pip install ultralytics") from exc

        data_yaml = build_data_yaml(req.version)

        cfg: Dict[str, Any] = req.config or {}
        rtdetr_cfg = cfg.get("rtdetr", {}) if isinstance(cfg.get("rtdetr"), dict) else {}

        weights = rtdetr_cfg.get("weights", "rtdetrv2-m.pt")
        imgsz = rtdetr_cfg.get("imgsz", 640)
        batch = rtdetr_cfg.get("batch", 8)
        epochs = rtdetr_cfg.get("epochs", 1)
        device = rtdetr_cfg.get("device", None)
        project = rtdetr_cfg.get("project", "runs/rtdetr")
        name = rtdetr_cfg.get("name", f"{self.name}-{req.version}")
        cache = rtdetr_cfg.get("cache", False)
        workers = rtdetr_cfg.get("workers", None)

        model = RTDETR(weights)

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
