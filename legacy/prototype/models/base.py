from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TrainRequest:
    version: str
    config: Dict[str, Any]

class Model:
    name: str = "base"
    def train(self, req: TrainRequest) -> None:
        raise NotImplementedError
