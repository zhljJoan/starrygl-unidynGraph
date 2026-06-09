from __future__ import annotations

from typing import Any

import torch

from atc_starrygl_lib.comm.dist_index import dist_index_is_shared, dist_index_loc
from atc_starrygl_lib.core.errors import RegistryError
from atc_starrygl_lib.core.registry import BackendRegistry
from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.ctdg.runtime import MemShareTemporalSamplingBackend
from atc_starrygl_lib.dtdg.runtime import FlareDTDGBackend
from atc_starrygl_lib.tasks import register_builtin_tasks


def register_builtin_backends() -> None:
    for name, backend in {
        "ctdg": MemShareTemporalSamplingBackend,
        "dtdg": FlareDTDGBackend,
    }.items():
        try:
            BackendRegistry.register(name, backend)
        except RegistryError:
            pass
    register_builtin_tasks()


def artifact_bundle(root, *, graph_mode: str, world_size: int) -> ArtifactBundle:
    root = root.expanduser().resolve()
    files = {
        "graph": root / "graph.pt",
        "dist": root / "dist.pt",
        "meta": root / "meta.json",
    }
    for rank in range(int(world_size)):
        files[f"rank_{rank:03d}"] = root / f"rank_{rank:03d}.pt"
        pd = root / f"partition_data_{rank:03d}.pt"
        if pd.exists():
            files[f"partition_data_{rank:03d}"] = pd
        feature = root / f"feature_{rank:03d}.pt"
        if feature.exists():
            files[f"feature_{rank:03d}"] = feature
    return ArtifactBundle(root=root, graph_mode=graph_mode, files=files)


def build_model_and_head(ctx: RuntimeContext, backend: Any) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    graph = _load_graph(ctx)
    model_cfg = dict(ctx.config.get("model", {}))
    task_name = str(ctx.config.get("task", {}).get("name", "")).lower()
    mode = str(ctx.config.get("graph", {}).get("mode", "")).lower()
    name = str(model_cfg.get("name", model_cfg.get("type", model_cfg.get("arch", "")))).lower()
    if mode == "dtdg":
        return _build_dtdg_model(name=name, model_cfg=model_cfg, task_name=task_name, graph=graph, device=ctx.device), None
    return _build_temporal_sampling_model(name=name, model_cfg=model_cfg, task_name=task_name, graph=graph, backend=backend, ctx=ctx)


def _build_dtdg_model(*, name: str, model_cfg: dict[str, Any], task_name: str, graph: dict[str, Any], device: str) -> torch.nn.Module:
    from atc_starrygl_lib.models.dtdg import EvolveGCN, GCN, MPNN_LSTM, TGCN

    in_dim = _node_feature_dim(graph, model_cfg)
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_size", 16)))
    out_dim = _task_output_dim(graph, model_cfg, task_name)
    layers = int(model_cfg.get("layers", model_cfg.get("num_layers", model_cfg.get("gcn_layers", 1))))
    if name == "tgcn":
        model = TGCN(input_size=in_dim, hidden_size=hidden_dim, output_size=out_dim, num_gcn_layers=layers)
    elif name == "gcn":
        model = GCN(in_dim, out_dim, num_layers=layers)
    elif name in {"mpnn_lstm", "mpnn-lstm"}:
        model = MPNN_LSTM(input_size=in_dim, hidden_size=hidden_dim, output_size=out_dim)
    elif name in {"evolvegcn", "evolve_gcn"}:
        model = EvolveGCN(input_size=in_dim, hidden_size=hidden_dim, output_size=out_dim, num_layers=layers)
    else:
        raise ValueError(f"unsupported DTDG model: {name!r}")
    return model.to(torch.device(device))


def _build_temporal_sampling_model(
    *,
    name: str,
    model_cfg: dict[str, Any],
    task_name: str,
    graph: dict[str, Any],
    backend: Any,
    ctx: RuntimeContext,
) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    from atc_starrygl_lib.memory import (
        AsyncMemoryCommitter,
        HistoricalBlend,
        HistoricalDeltaFilter,
        RuntimeAsyncMemoryUpdater,
        SharedHistoricalCache,
    )
    from atc_starrygl_lib.memory.sync_mode import resolve_memory_sync_mode
    from atc_starrygl_lib.models.ctdg import GeneralModel
    from atc_starrygl_lib.models.shared import EdgePredictHead, EdgeRegressHead, NodeClassifyHead, NodeRegressHead

    if name in {"gcn", "tgcn", "mpnn_lstm", "mpnn-lstm", "evolvegcn", "evolve_gcn"}:
        return _build_sampled_block_model(name=name, model_cfg=model_cfg, task_name=task_name, graph=graph, ctx=ctx)
    if name not in {"general", "ctdg_general"}:
        raise ValueError(f"unsupported temporal_sampling model: {name!r}")
    runtime = getattr(backend, "_runtime", None)
    if runtime is None or runtime.memory_runtime is None or runtime.mailbox_runtime is None:
        raise RuntimeError("CTDG GeneralModel requires build_memory_runtime and build_mailbox_runtime")
    dim_node = _node_feature_dim(graph, model_cfg)
    dim_edge = _edge_feature_dim(graph, model_cfg)
    if dim_edge <= 0:
        dim_edge = int(getattr(runtime, "edge_feat_dim", 0))
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_size", 16)))
    runtime_cfg = dict(ctx.config.get("runtime", {}))
    mailbox_cfg = dict(runtime_cfg.get("mailbox", {}))
    async_memory_cfg = (
        dict(runtime_cfg.get("async_memory", {}))
        if isinstance(runtime_cfg.get("async_memory", {}), dict)
        else {}
    )
    historical_cfg = (
        dict(runtime_cfg.get("historical", {}))
        if isinstance(runtime_cfg.get("historical", {}), dict)
        else {}
    )
    sync_cfg = resolve_memory_sync_mode(runtime_cfg)
    mailbox_msg_dim = int(
        runtime_cfg.get(
            "mailbox_msg_dim",
            mailbox_cfg.get("msg_dim", 2 * hidden_dim + max(int(dim_edge), 0)),
        )
    )
    shared_memory_ssim = float(
        runtime_cfg.get(
            "shared_memory_ssim",
            historical_cfg.get("shared_memory_ssim", historical_cfg.get("alpha", 0.0)),
        )
    )
    memory_dim_time = int(model_cfg.get("memory_dim_time", model_cfg.get("dim_time", hidden_dim)))
    gnn_dim_time = int(model_cfg.get("gnn_dim_time", model_cfg.get("dim_time", hidden_dim)))
    model_config = {
        "sample": {"history": int(model_cfg.get("history", 1))},
        "memory": {
            "type": "node",
            "dim_out": hidden_dim,
            "dim_time": memory_dim_time,
            "input_dim": mailbox_msg_dim,
            "memory_update": str(model_cfg.get("memory_update", "gru")),
            "mailbox_size": int(model_cfg.get("memory_history", ctx.config.get("runtime", {}).get("mailbox_size", 1))),
            "combine_node_feature": bool(model_cfg.get("combine_node_feature", False)),
        },
        "gnn": {
            "arch": str(model_cfg.get("gnn_arch", "identity")),
            "dim_time": gnn_dim_time,
            "att_head": int(model_cfg.get("att_head", 1)),
            "dim_out": hidden_dim,
            "layer": int(model_cfg.get("layers", 1)),
        },
        "train": {
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "att_dropout": float(model_cfg.get("att_dropout", 0.0)),
        },
    }
    committer = AsyncMemoryCommitter(runtime.memory_runtime, runtime.mailbox_runtime)
    shared_filter_enabled = bool(sync_cfg["shared_filter"])
    historical_enabled = bool(sync_cfg["historical_enabled"])
    historical_cache = None
    historical_filter = None
    historical_blend = None
    if historical_enabled:
        filter_index_scope = str(historical_cfg.get("filter_index_scope", "shared_local")).strip().lower()
        filter_node_id_to_index = None
        filter_num_nodes = int(graph.get("num_nodes", 0))
        if filter_index_scope in {"shared_local", "shared-local", "shared"}:
            memory_index = getattr(runtime.memory_runtime, "index", None)
            read_dist_index = getattr(memory_index, "read_dist_index", None)
            if read_dist_index is not None:
                read_index = read_dist_index.long().cpu().contiguous()
                shared_mask = dist_index_is_shared(read_index)
                filter_node_id_to_index = torch.full_like(read_index, -1)
                if bool(shared_mask.any().item()):
                    filter_rows = dist_index_loc(read_index[shared_mask])
                    filter_node_id_to_index[shared_mask] = filter_rows
                    filter_num_nodes = int(filter_rows.max().item()) + 1
                else:
                    filter_num_nodes = 0
        historical_cache = SharedHistoricalCache(
            memory_dim=hidden_dim,
            num_nodes=int(graph.get("num_nodes", 0)),
            alpha=shared_memory_ssim,
            times_threshold=int(historical_cfg.get("times_threshold", 10)),
            time_threshold=historical_cfg.get("time_threshold"),
            preload_candidate_delta=bool(sync_cfg.get("preload_candidate_delta", False)),
        )
        historical_filter = HistoricalDeltaFilter(
            memory_dim=hidden_dim,
            num_nodes=filter_num_nodes,
            preload_candidate_delta=bool(sync_cfg.get("preload_candidate_delta", False)),
            node_id_to_index=filter_node_id_to_index,
        )
        if bool(sync_cfg.get("historical_blend", False)):
            historical_blend = HistoricalBlend(memory_dim=hidden_dim, learnable_gamma=True)
    updater = RuntimeAsyncMemoryUpdater(
        torch.nn.Identity(),
        committer=committer,
        historical_blend=historical_blend,
        historical_cache=historical_cache,
        historical_filter=historical_filter,
        use_staged_commit=bool(sync_cfg["staged_commit"]),
        use_shared_filter=bool(shared_filter_enabled),
        enable_delta_compensation=bool(sync_cfg["delta_compensation"]),
        delta_compensation_gamma=float(async_memory_cfg.get("delta_compensation_gamma", 0.5)),
    )
    setattr(runtime, "historical_cache_provider", historical_cache)
    setattr(runtime, "historical_filter_provider", historical_filter)
    setattr(runtime, "historical_updater_provider", updater)
    model = GeneralModel.from_config(
        dim_node=dim_node,
        dim_edge=dim_edge,
        num_nodes=int(graph.get("num_nodes", 0)),
        config=model_config,
        runtime_memory_updater=updater,
    ).to(torch.device(ctx.device))
    head = _build_head(task_name=task_name, dim=hidden_dim, graph=graph, model_cfg=model_cfg).to(torch.device(ctx.device))
    return model, head


def _build_sampled_block_model(
    *,
    name: str,
    model_cfg: dict[str, Any],
    task_name: str,
    graph: dict[str, Any],
    ctx: RuntimeContext,
) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    if name != "gcn":
        raise ValueError(
            f"{name!r} with neighbor sampling is routed to temporal_sampling, "
            "but only sampled-block gcn is implemented today"
        )
    in_dim = _node_feature_dim(graph, model_cfg)
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("hidden_size", 16)))
    layers = int(model_cfg.get("layers", model_cfg.get("num_layers", 1)))
    model = SampledBlockGCNEncoder(in_dim, hidden_dim, num_layers=layers).to(torch.device(ctx.device))
    head = _build_head(task_name=task_name, dim=hidden_dim, graph=graph, model_cfg=model_cfg).to(torch.device(ctx.device))
    return model, head


class SampledBlockGCNEncoder(torch.nn.Module):
    """GCN encoder for the temporal_sampling path using sampled DGL blocks."""

    def __init__(self, in_dim: int, hidden_dim: int, *, num_layers: int = 1) -> None:
        super().__init__()
        from atc_starrygl_lib.models.dtdg import GCNConv

        self.convs = torch.nn.ModuleList()
        for layer in range(int(num_layers)):
            self.convs.append(GCNConv(in_dim if layer == 0 else hidden_dim, hidden_dim))

    def encode(self, mfgs: Any) -> torch.Tensor:
        layers = _sampled_layers(mfgs)
        h = None
        for layer, conv in enumerate(self.convs):
            block = layers[min(layer, len(layers) - 1)]
            if layer == 0:
                h = block.srcdata.get("h", block.srcdata.get("x"))
                if h is None:
                    raise RuntimeError("sampled block is missing srcdata['h'] or srcdata['x']")
            else:
                block.srcdata["h"] = h
            h = conv(block, h)
        return h


def _sampled_layers(mfgs: Any) -> list[Any]:
    if isinstance(mfgs, (list, tuple)):
        if mfgs and isinstance(mfgs[0], (list, tuple)):
            return [layer[0] for layer in mfgs if layer]
        return list(mfgs)
    return [mfgs]


def _build_head(*, task_name: str, dim: int, graph: dict[str, Any], model_cfg: dict[str, Any]) -> torch.nn.Module:
    from atc_starrygl_lib.models.shared import EdgeLabelHead, EdgePredictHead, EdgeRegressHead, NodeClassifyHead, NodeRegressHead

    if task_name in {"edge_prediction", "edge_predict", "link_prediction"}:
        return EdgePredictHead(dim)
    if task_name in {"edge_label_prediction", "edge_classification"}:
        return EdgeLabelHead(dim, _num_edge_classes(graph, model_cfg))
    if task_name == "edge_regression":
        return EdgeRegressHead(dim, int(model_cfg.get("out_dim", 1)))
    if task_name in {"node_prediction", "node_classification"}:
        return NodeClassifyHead(dim, _num_classes(graph, model_cfg))
    return NodeRegressHead(dim, int(model_cfg.get("out_dim", 1)))


def _load_graph(ctx: RuntimeContext) -> dict[str, Any]:
    path = ctx.artifact_root / "graph.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return {}


def _node_feature_dim(graph: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    if "input_dim" in model_cfg:
        return int(model_cfg["input_dim"])
    feat = graph.get("node_feat")
    if feat is None:
        return int(model_cfg.get("dim_node", model_cfg.get("node_dim", 0)))
    return int(torch.as_tensor(feat).size(-1))


def _edge_feature_dim(graph: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    if "edge_dim" in model_cfg:
        return int(model_cfg["edge_dim"])
    feat = graph.get("edge_feat")
    if feat is None:
        return 0
    return int(torch.as_tensor(feat).size(-1))


def _task_output_dim(graph: dict[str, Any], model_cfg: dict[str, Any], task_name: str) -> int:
    if "output_dim" in model_cfg:
        return int(model_cfg["output_dim"])
    if task_name in {"node_prediction", "node_classification"}:
        return _num_classes(graph, model_cfg)
    labels = graph.get("node_label")
    if labels is not None:
        label_t = torch.as_tensor(labels)
        if bool(graph.get("node_label_time_varying", False)):
            return int(label_t.size(-1)) if label_t.dim() > 2 else 1
        if label_t.dim() > 1:
            return int(label_t.size(-1))
    return int(model_cfg.get("out_dim", 1))


def _num_classes(graph: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    labels = graph.get("node_label")
    if labels is None or torch.as_tensor(labels).numel() == 0:
        return 2
    return int(torch.as_tensor(labels).max().item()) + 1


def _num_edge_classes(graph: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    if "num_classes" in model_cfg:
        return int(model_cfg["num_classes"])
    labels = graph.get("edge_label")
    if labels is None or torch.as_tensor(labels).numel() == 0:
        return 2
    labels_t = torch.as_tensor(labels)
    if labels_t.dtype.is_floating_point and int(labels_t.max().item()) <= 1:
        return 1
    return int(labels_t.max().item()) + 1
