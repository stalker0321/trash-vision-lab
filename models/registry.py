# models/registry.py
from __future__ import annotations

from typing import Dict, Type
from .base import Model

_REGISTRY: Dict[str, Type[Model]] = {}


def register(name: str):
    """
    Decorator: @register("dummy") registers the class in the global registry.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Model registry name must be a non-empty string")

    def _wrap(cls: Type[Model]) -> Type[Model]:
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' already registered: {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        cls.name = name  # optional: attach key on class
        return cls

    return _wrap


def get(name: str) -> Model:
    """
    Returns an instance of a registered model by its registry key.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name]()  # create instance


def available() -> list[str]:
    """
    Returns list of registered model keys.
    """
    return sorted(_REGISTRY.keys())
