from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    graph_mode: str


class ModelRegistry:
    _models: dict[str, ModelSpec] = {
        "tgn": ModelSpec("tgn", "tgn", "ctdg"),
        "dyrep": ModelSpec("dyrep", "dyrep", "ctdg"),
        "jodie": ModelSpec("jodie", "jodie", "ctdg"),
        "tgat": ModelSpec("tgat", "tgat", "ctdg"),
        "apan": ModelSpec("apan", "apan", "ctdg"),
        "evolvegcn": ModelSpec("evolvegcn", "evolvegcn", "dtdg"),
        "tgcn": ModelSpec("tgcn", "tgcn", "dtdg"),
        "mpnn_lstm": ModelSpec("mpnn_lstm", "mpnn_lstm", "dtdg"),
        "gcn": ModelSpec("gcn", "gcn", "dtdg"),
    }

    @classmethod
    def resolve(cls, model_name: str | None = None, family: str | None = None) -> ModelSpec:
        key = (family or model_name or "").lower()
        if key not in cls._models:
            raise KeyError(f"Unknown model/family: {key}")
        return cls._models[key]
