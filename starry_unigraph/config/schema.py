from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")

MODEL_GRAPH_MODES = {
    "tgn": "ctdg",
    "dyrep": "ctdg",
    "jodie": "ctdg",
    "tgat": "ctdg",
    "apan": "ctdg",
    "evolvegcn": "dtdg",
    "tgcn": "dtdg",
    "mpnn_lstm": "dtdg",
}

REQUIRED_PATHS = (
    "model.name",
    "model.family",
    "model.task",
    "data.root",
    "data.name",
    "train.epochs",
    "train.batch_size",
    "runtime.backend",
    "runtime.device",
)

CTDG_ACTIVE_PREFIXES = (
    "sampling.",
    "model.memory.",
    "runtime.state_sync",
)

DTDG_ACTIVE_PREFIXES = (
    "train.snaps",
    "graph.storage",
    "model.window.size",
    "runtime.cache",
)


class ConfigError(ValueError):
    pass


def load_default_config() -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_config(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    return _deep_merge(config, overrides or {})


def load_config(config_or_path: str | Path | dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(config_or_path, dict):
        config = deepcopy(config_or_path)
    else:
        path = Path(config_or_path)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    return merge_config(load_default_config(), merge_config(config, overrides))


def get_by_path(config: dict[str, Any], path: str) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Missing required config path: {path}")
        cur = cur[part]
    return cur


def set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    cur = config
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def detect_graph_mode(config: dict[str, Any]) -> str:
    explicit = config.get("data", {}).get("graph_mode")
    if explicit:
        return explicit
    family = str(get_by_path(config, "model.family")).lower()
    if family in MODEL_GRAPH_MODES:
        mode = MODEL_GRAPH_MODES[family]
        set_by_path(config, "data.graph_mode", mode)
        return mode
    raise ConfigError(f"Unable to infer graph mode for model.family={family!r}")


def validate_config(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for path in REQUIRED_PATHS:
        get_by_path(config, path)

    mode = detect_graph_mode(config)
    graph_storage = config.get("graph", {}).get("storage")
    if mode == "dtdg" and graph_storage != "snapshots":
        warnings.append("DTDG mode usually expects graph.storage=snapshots; continuing with configured value.")
    if mode == "ctdg" and graph_storage == "snapshots":
        warnings.append("CTDG mode received graph.storage=snapshots; sampling pipeline will ignore snapshot storage.")

    active = CTDG_ACTIVE_PREFIXES if mode == "ctdg" else DTDG_ACTIVE_PREFIXES
    inactive = DTDG_ACTIVE_PREFIXES if mode == "ctdg" else CTDG_ACTIVE_PREFIXES
    warnings.extend(_warn_inactive(config, active_prefixes=active, inactive_prefixes=inactive))
    return warnings


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _warn_inactive(config: dict[str, Any], active_prefixes: tuple[str, ...], inactive_prefixes: tuple[str, ...]) -> list[str]:
    flat = _flatten(config)
    warnings: list[str] = []
    for path, value in flat.items():
        if value is None:
            continue
        if any(path.startswith(prefix) for prefix in inactive_prefixes):
            if any(path.startswith(prefix) for prefix in active_prefixes):
                continue
            warnings.append(f"Inactive config field for selected graph mode: {path}")
    return warnings
