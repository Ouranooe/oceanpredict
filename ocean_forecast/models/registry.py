from __future__ import annotations

import inspect
from typing import Dict, Type

from .base import BaseForecaster
from .cnn_transformer import CNNTransformerForecaster
from .convlstm import ConvLSTMForecaster
from .tau import TAUForecaster


MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {
    "convlstm": ConvLSTMForecaster,
    "tau": TAUForecaster,
    "cnn_transformer": CNNTransformerForecaster,
    "cnn_transformer_phys": CNNTransformerForecaster,
    "cnn_transformer_physics": CNNTransformerForecaster,
}


def build_model(model_name: str, **kwargs) -> BaseForecaster:
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    model_cls = MODEL_REGISTRY[name]
    sig = inspect.signature(model_cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return model_cls(**kwargs)

    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return model_cls(**filtered)
