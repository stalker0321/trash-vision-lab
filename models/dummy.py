from .base import Model, TrainRequest
from .registry import register

@register("dummy")
class Dummy(Model):
    def train(self, req: TrainRequest) -> None:
        print("OK: dummy train")
        print("version:", req.version)
        print("config keys:", list(req.config.keys()))
