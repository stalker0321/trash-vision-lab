"""YOLOv8m wrapper."""

from .base import ModelBase
from .registry import register


@register("yolo_v8m")
class YoloV8m(ModelBase):
    def train(self) -> None:
        raise NotImplementedError
