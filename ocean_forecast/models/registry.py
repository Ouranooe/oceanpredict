from __future__ import annotations

from typing import Dict, Type

from .base import BaseForecaster
from .convlstm import ConvLSTMForecaster


MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {
    "convlstm": ConvLSTMForecaster,
}


def build_model(model_name: str, **kwargs) -> BaseForecaster:
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)

