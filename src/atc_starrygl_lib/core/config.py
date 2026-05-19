from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .types import RuntimeContext


def load_config(config_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Load a plain dict config from mapping, JSON, or YAML.

    YAML is supported when PyYAML is installed. Core keeps the dependency optional
    so the library can be imported in minimal runtime environments.
    """

    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)

    path = Path(config_or_path).expanduser()
    if not path.exists():
        raise ConfigError(f"config file does not exist: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return _ensure_mapping(json.loads(text), path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("YAML config requires PyYAML to be installed") from exc
        return _ensure_mapping(yaml.safe_load(text), path)
    raise ConfigError(f"unsupported config extension: {path.suffix}")


def normalize_config(config_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    config = load_config(config_or_path)
    graph = _section(config, "graph")
    task = _section(config, "task")
    runtime = dict(config.get("runtime", {}))

    graph["mode"] = _required_str(graph, "mode", "graph")
    task["name"] = _required_str(task, "name", "task")
    runtime.setdefault("device", "cpu")

    config["graph"] = graph
    config["task"] = task
    config["runtime"] = runtime
    return config


def build_context(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    artifact_root: str | Path,
    rank: int = 0,
    world_size: int = 1,
    device: str | None = None,
) -> RuntimeContext:
    config = normalize_config(config_or_path)
    runtime_device = str(device or config["runtime"].get("device", "cpu"))
    return RuntimeContext(
        config=config,
        artifact_root=Path(artifact_root).expanduser().resolve(),
        rank=int(rank),
        world_size=int(world_size),
        device=runtime_device,
    )


def _ensure_mapping(value: Any, source: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"config must be a mapping: {source}")
    return dict(value)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing or invalid config section: {name}")
    return dict(value)


def _required_str(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"missing required config field: {section_name}.{key}")
    return value.strip().lower()
