"""Model registry."""


MODEL_REGISTRY = {}


def register(name: str, cls) -> None:
    MODEL_REGISTRY[name] = cls
