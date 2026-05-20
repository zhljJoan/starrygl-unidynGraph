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
    gnn = dict(config.get("gnn", {})) if isinstance(config.get("gnn"), Mapping) else {}
    model = _section(config, "model") if isinstance(config.get("model"), Mapping) else {}
    model = _merge_model_gnn(model, gnn)
    sampling = _merge_sampling_config(config.get("sampling"), gnn.get("sampling"))
    runtime = dict(config.get("runtime", {}))
    runtime = _merge_runtime_gnn(runtime, gnn=gnn, sampling=sampling, model=model)
    preprocess = dict(config.get("preprocess", {})) if isinstance(config.get("preprocess"), Mapping) else {}

    task["name"] = _required_str(task, "name", "task")
    if "mode" not in graph or not str(graph.get("mode", "")).strip():
        plan = infer_execution_plan(model=model, sampling=sampling, task=task, runtime=runtime)
        graph["mode"] = _graph_mode_for_plan(plan)
        runtime.setdefault("execution_plan", plan)
    else:
        graph["mode"] = _required_str(graph, "mode", "graph")
        runtime.setdefault("execution_plan", _execution_plan_for_graph_mode(graph["mode"]))
    runtime.setdefault("device", "cpu")

    config["graph"] = graph
    config["task"] = task
    config["model"] = model
    config["gnn"] = gnn
    config["sampling"] = sampling
    config["runtime"] = runtime
    config["preprocess"] = preprocess
    return config


def infer_execution_plan(
    *,
    model: Mapping[str, Any],
    sampling: Mapping[str, Any] | None = None,
    task: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> str:
    """Infer the user-facing execution plan without exposing CTDG/DTDG names."""

    sampling = {} if sampling is None else sampling
    runtime = {} if runtime is None else runtime
    model_name = _model_name(model)
    if _sampling_enabled(sampling) or _sampling_enabled(runtime):
        return "temporal_sampling"
    if model_name in {"general", "tgn", "tgat", "jodie", "dyrep", "identity_ctdg", "ctdg_general"}:
        return "temporal_sampling"
    if model_name in {"tgcn", "gcn", "mpnn_lstm", "mpnn-lstm", "evolvegcn", "evolve_gcn"}:
        return "snapshot_full_graph"
    raise ConfigError(
        "cannot infer execution path; set model.name to a known CTDG/DTDG model "
        "or enable sampling.fanouts for the CTDG sampling path"
    )


def infer_graph_mode(
    *,
    model: Mapping[str, Any],
    sampling: Mapping[str, Any] | None = None,
    task: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> str:
    return _graph_mode_for_plan(infer_execution_plan(model=model, sampling=sampling, task=task, runtime=runtime))


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


def _model_name(model: Mapping[str, Any]) -> str:
    value = model.get("name", model.get("type", model.get("arch", "")))
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("missing required config field: model.name")
    return value.strip().lower()


def _sampling_enabled(config: Mapping[str, Any]) -> bool:
    fanouts = config.get("fanouts")
    if fanouts is not None:
        if isinstance(fanouts, str):
            return bool(fanouts.strip())
        try:
            return len(fanouts) > 0  # type: ignore[arg-type]
        except TypeError:
            return bool(fanouts)
    for key in ("neighbor_sampling", "sample_neighbors", "build_sampler", "sampler"):
        value = config.get(key)
        if value not in (None, False, "false", "False", "none", "None"):
            return True
    return False


def _merge_model_gnn(model: Mapping[str, Any], gnn: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(model)
    for key in (
        "history",
        "layers",
        "num_layers",
        "gcn_layers",
        "gnn_arch",
        "memory_update",
        "memory_history",
        "hidden_dim",
        "hidden_size",
        "dim_time",
        "att_head",
        "dropout",
        "att_dropout",
    ):
        if key in gnn and key not in out:
            out[key] = gnn[key]
    return out


def _merge_sampling_config(raw_sampling: Any, raw_gnn_sampling: Any) -> dict[str, Any]:
    sampling = dict(raw_sampling) if isinstance(raw_sampling, Mapping) else {}
    if isinstance(raw_gnn_sampling, Mapping):
        merged = dict(raw_gnn_sampling)
        merged.update(sampling)
        sampling = merged
    return sampling


def _merge_runtime_gnn(runtime: Mapping[str, Any], *, gnn: Mapping[str, Any], sampling: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(runtime)
    if _sampling_enabled(sampling):
        out.setdefault("build_sampler", True)
        if "fanouts" in sampling:
            out.setdefault("fanouts", sampling["fanouts"])
            try:
                out.setdefault("num_layers", len(sampling["fanouts"]))  # type: ignore[arg-type]
            except TypeError:
                pass
        if "policy" in sampling:
            out.setdefault("policy", _normalize_sampling_policy(str(sampling["policy"])))
        for src, dst in (
            ("probability", "sample_probability"),
            ("sample_probability", "sample_probability"),
            ("boundary_probability", "boundary_probability"),
            ("boundery_probability", "boundary_probability"),
            ("sampler_workers", "sampler_workers"),
            ("workers", "workers"),
        ):
            if src in sampling:
                out.setdefault(dst, sampling[src])
    full_graph = {}
    if isinstance(gnn.get("full_graph"), Mapping):
        full_graph.update(gnn["full_graph"])
    if isinstance(gnn.get("slice_config"), Mapping):
        full_graph.update(gnn["slice_config"])
    if full_graph:
        aliases = {
            "chunk_order": "chunk_order",
            "chunk_decay": "chunk_decay",
            "decay": "chunk_decay",
            "num_full_snapshots": "num_full_snapshots",
            "disable_states": "disable_states",
            "disable_routes": "disable_routes",
        }
        for src, dst in aliases.items():
            if src in full_graph:
                out.setdefault(dst, full_graph[src])
    if "history" in model:
        out.setdefault("num_full_snapshots", int(model["history"]))
    if "memory_history" in model:
        out.setdefault("mailbox_size", int(model["memory_history"]))
    return out


def _graph_mode_for_plan(plan: str) -> str:
    plan = str(plan).strip().lower()
    if plan in {"temporal_sampling", "sampling", "neighbor_sampling", "sampled"}:
        return "ctdg"
    if plan in {"snapshot_full_graph", "full_graph", "snapshot", "stgraph"}:
        return "dtdg"
    raise ConfigError(f"unknown execution plan: {plan!r}")


def _execution_plan_for_graph_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "ctdg":
        return "temporal_sampling"
    if mode == "dtdg":
        return "snapshot_full_graph"
    raise ConfigError(f"unknown graph mode: {mode!r}")


def _normalize_sampling_policy(policy: str) -> str:
    policy = str(policy).strip().lower()
    if policy.startswith("boundery_"):
        policy = "boundary_" + policy[len("boundery_"):]
    aliases = {
        "boundary_recent_sample": "boundary_recent_uniform",
    }
    return aliases.get(policy, policy)


def _required_str(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"missing required config field: {section_name}.{key}")
    return value.strip().lower()
