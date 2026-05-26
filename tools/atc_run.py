from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from atc_starrygl_lib.core.config import build_context
from atc_starrygl_lib.core.config import normalize_config
from atc_starrygl_lib.core.session import TrainingSession
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook
from atc_starrygl_lib.ctdg.runtime.backend import build_memory_replica_index
from atc_starrygl_lib.dtdg.runtime.stgraph_loader import STGraphWindow
from atc_starrygl_lib.runtime.grad_sync import AsyncGradientSyncOptimizer
from atc_starrygl_lib.runtime.unified import artifact_bundle, build_model_and_head, register_builtin_backends


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_dist(args.device)
    device = _device(args.device, local_rank)
    register_builtin_backends()
    config = _runtime_config(args.config)
    _apply_runtime_threading(config)
    ctx = build_context(
        config,
        artifact_root=args.artifact_root,
        rank=rank,
        world_size=world_size,
        device=str(device),
    )
    session = TrainingSession.from_config(ctx)
    action = str(args.action)

    if action == "prepare":
        if rank == 0:
            session.prepare()
        _barrier()
        _shutdown_dist()
        return

    if action == "run":
        if rank == 0 and bool(args.prepare):
            session.prepare()
        _barrier()
    artifacts = artifact_bundle(ctx.artifact_root, graph_mode=str(ctx.config["graph"]["mode"]), world_size=world_size)
    session.build_runtime(artifacts)

    if action in {"train", "run"}:
        summary = _train_eval(ctx, session, epochs=_epochs(args, ctx.config))
    elif action == "eval":
        summary = {"eval": _eval(ctx, session, split=args.split)}
    elif action == "predict":
        summary = {"predict": _predict(ctx, session, split=args.split)}
    else:
        raise ValueError(f"unknown action: {action!r}")

    if rank == 0:
        print(json.dumps({"rank": rank, "execution_plan": ctx.config["runtime"].get("execution_plan"), **summary}, sort_keys=True), flush=True)
    _barrier()
    _shutdown_dist()


def _train_eval(ctx: Any, session: TrainingSession, *, epochs: int) -> dict[str, Any]:
    model, head = build_model_and_head(ctx, session.backend)
    sync_mode = _gradient_sync_mode(ctx)
    if sync_mode == "ddp":
        model = _wrap_ddp(model, ctx)
        if head is not None:
            head = _wrap_ddp(head, ctx)
    _sync_module_state(model)
    if head is not None:
        _sync_module_state(head)
    task = session.task
    train_cfg = dict(ctx.config.get("train", {}))
    lr = float(train_cfg.get("lr", ctx.config.get("optimizer", {}).get("lr", 0.01) if isinstance(ctx.config.get("optimizer"), dict) else 0.01))
    weight_decay = float(train_cfg.get("weight_decay", ctx.config.get("optimizer", {}).get("weight_decay", 0.0) if isinstance(ctx.config.get("optimizer"), dict) else 0.0))
    params = list(model.parameters()) + ([] if head is None else list(head.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if sync_mode == "async":
        optimizer = AsyncGradientSyncOptimizer(optimizer, params)
    route_summary = _validate_routes(
        session,
        split="train",
        device=torch.device(ctx.device),
        max_batches=int(ctx.config.get("runtime", {}).get("route_check_batches", 0)),
    )
    rows = []
    for epoch in range(int(epochs)):
        t_epoch0 = time.perf_counter()
        _reset_epoch_state(ctx, session, model)
        t0 = time.perf_counter()
        train_metrics = _train_epoch(ctx, session, model, head, task, optimizer)
        train_wall = float(time.perf_counter() - t0)
        train_metrics = dict(train_metrics)
        train_metrics["seconds"] = float(train_metrics.get("stage_wall_seconds", train_wall))
        dtdg_sum_keys = {"loss"} if str(ctx.config["graph"]["mode"]) == "dtdg" else set()
        train_time_keys = {key for key in train_metrics if key.endswith("seconds")}
        train_sum_keys = set(dtdg_sum_keys)
        if "stage_batches" in train_metrics:
            train_sum_keys.add("stage_batches")
        if "backend_batches" in train_metrics:
            train_sum_keys.add("backend_batches")
        t_reduce0 = time.perf_counter()
        train_metrics = _reduce_metrics(train_metrics, device=torch.device(ctx.device), time_keys=train_time_keys, sum_keys=train_sum_keys)
        train_reduce_wall = float(time.perf_counter() - t_reduce0)
        train_metrics["train_metrics_reduce_wall_seconds"] = train_reduce_wall
        t_val0 = time.perf_counter()
        val_metrics = _eval_with_model(ctx, session, model, head, task, split="val")
        val_wall = float(time.perf_counter() - t_val0)
        val_metrics = _reduce_metrics(val_metrics, device=torch.device(ctx.device), sum_keys=dtdg_sum_keys)
        t_test0 = time.perf_counter()
        test_metrics = _eval_with_model(ctx, session, model, head, task, split="test")
        test_wall = float(time.perf_counter() - t_test0)
        test_metrics = _reduce_metrics(test_metrics, device=torch.device(ctx.device), sum_keys=dtdg_sum_keys)
        epoch_wall = float(time.perf_counter() - t_epoch0)
        train_seconds = float(train_metrics.get("seconds", train_wall))
        other_wall = max(0.0, epoch_wall - train_wall - val_wall - test_wall)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "timing": {
                "epoch_wall_seconds": epoch_wall,
                "train_call_wall_seconds": train_wall,
                "train_reported_seconds": train_seconds,
                "train_wrapper_gap_seconds": max(0.0, train_wall - train_seconds),
                "train_metrics_reduce_wall_seconds": train_reduce_wall,
                "val_wall_seconds": val_wall,
                "test_wall_seconds": test_wall,
                "other_wall_seconds": other_wall,
            },
        }
        rows.append(row)
        if int(ctx.rank) == 0:
            print(json.dumps({"rank": int(ctx.rank), **row}, sort_keys=True), flush=True)
    out: dict[str, Any] = {"epochs": rows}
    if route_summary["checked"] > 0:
        out["route"] = route_summary
    return out


def _reset_epoch_state(ctx: Any, session: TrainingSession, model: torch.nn.Module) -> None:
    runtime_cfg = dict(ctx.config.get("runtime", {}))
    if not bool(runtime_cfg.get("reset_memory_each_epoch", False)):
        return
    if hasattr(session.backend, "reset_state"):
        session.backend.reset_state()
    module = model.module if hasattr(model, "module") else model
    updater = getattr(module, "memory_updater", None)
    if updater is not None and hasattr(updater, "reset_state"):
        updater.reset_state()


def _train_epoch(ctx: Any, session: TrainingSession, model: torch.nn.Module, head: torch.nn.Module | None, task: Any, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    mode = str(ctx.config["graph"]["mode"])
    task_name = str(ctx.config["task"]["name"]).lower()
    if mode == "dtdg":
        from atc_starrygl_lib.dtdg.train_loop import train_edge_prediction_epoch, train_epoch

        if _is_edge_prediction(task_name):
            if head is None:
                raise RuntimeError("DTDG edge prediction requires a head")
            return train_edge_prediction_epoch(session, model, head, task, optimizer)
        return train_epoch(session, model, task, optimizer)

    from atc_starrygl_lib.ctdg.train_loop import train_epoch

    if head is None:
        raise RuntimeError("temporal_sampling path requires a head")
    t_commit0 = time.perf_counter()
    commit = _build_ctdg_memory_commit(session, enabled=bool(ctx.config.get("runtime", {}).get("commit_memory", True)))
    commit_build_wall = float(time.perf_counter() - t_commit0)
    t_train0 = time.perf_counter()
    out = train_epoch(session, model, head, task, optimizer, memory_commit=commit)
    out = dict(out)
    out["train_epoch_call_wall_seconds"] = float(time.perf_counter() - t_train0)
    out["memory_commit_hook_build_wall_seconds"] = commit_build_wall
    if commit is not None:
        out["memory_replica_index_build_wall_seconds"] = float(getattr(commit, "memory_replica_index_build_wall_seconds", 0.0))
        out["mailbox_replica_index_build_wall_seconds"] = float(getattr(commit, "mailbox_replica_index_build_wall_seconds", 0.0))
    return out


def _eval(ctx: Any, session: TrainingSession, *, split: str) -> dict[str, float]:
    model, head = build_model_and_head(ctx, session.backend)
    _sync_module_state(model)
    if head is not None:
        _sync_module_state(head)
    return _eval_with_model(ctx, session, model, head, session.task, split=split)


def _eval_with_model(ctx: Any, session: TrainingSession, model: torch.nn.Module, head: torch.nn.Module | None, task: Any, *, split: str) -> dict[str, float]:
    mode = str(ctx.config["graph"]["mode"])
    task_name = str(ctx.config["task"]["name"]).lower()
    if mode == "dtdg":
        from atc_starrygl_lib.dtdg.train_loop import evaluate, evaluate_edge_prediction

        if _is_edge_prediction(task_name):
            if head is None:
                raise RuntimeError("DTDG edge prediction requires a head")
            return evaluate_edge_prediction(session, model, head, task, split=split)
        return evaluate(session, model, task, split=split)

    from atc_starrygl_lib.ctdg.train_loop import evaluate

    if head is None:
        raise RuntimeError("temporal_sampling path requires a head")
    commit = _build_ctdg_memory_commit(session, enabled=bool(ctx.config.get("runtime", {}).get("eval_updates_memory", False)))
    return evaluate(session, model, head, task, split=split, memory_commit=commit)


def _predict(ctx: Any, session: TrainingSession, *, split: str) -> dict[str, int]:
    model, head = build_model_and_head(ctx, session.backend)
    _sync_module_state(model)
    if head is not None:
        _sync_module_state(head)
    if str(ctx.config["graph"]["mode"]) != "ctdg":
        count = sum(1 for _ in session.iter_batches(split))
        return {"batches": count}
    from atc_starrygl_lib.ctdg.train_loop import predict

    if head is None:
        raise RuntimeError("temporal_sampling path requires a head")
    commit = _build_ctdg_memory_commit(session, enabled=bool(ctx.config.get("runtime", {}).get("predict_updates_memory", True)))
    outputs = predict(session, model, head, split=split, memory_commit=commit)
    return {"batches": len(outputs)}


def _build_ctdg_memory_commit(session: TrainingSession, *, enabled: bool) -> CTDGMemoryCommitHook | None:
    if not enabled:
        return None
    backend = getattr(session, "backend", None)
    runtime = getattr(backend, "_runtime", None)
    replica_index = None
    mailbox_replica_index = None
    mailbox_runtime = None
    runtime_cfg = getattr(getattr(session, "ctx", None), "config", {}).get("runtime", {})
    async_memory_cfg = dict(runtime_cfg.get("async_memory", {})) if isinstance(runtime_cfg.get("async_memory", {}), dict) else {}
    build_stats: dict[str, float] = {
        "memory_replica_index_build_wall_seconds": 0.0,
        "mailbox_replica_index_build_wall_seconds": 0.0,
    }
    if runtime is not None and bool(runtime_cfg.get("memory_replica_push", False)):
        replica_index = getattr(runtime, "_cached_memory_replica_index", None)
        if replica_index is None:
            t0 = time.perf_counter()
            replica_index = build_memory_replica_index(dist=getattr(runtime, "dist", {}))
            build_stats["memory_replica_index_build_wall_seconds"] = float(time.perf_counter() - t0)
            setattr(runtime, "_cached_memory_replica_index", replica_index)
    if runtime is not None and bool(runtime_cfg.get("mailbox_replica_push", runtime_cfg.get("memory_replica_push", False))):
        mailbox_replica_index = getattr(runtime, "_cached_mailbox_replica_index", None)
        if mailbox_replica_index is None:
            t0 = time.perf_counter()
            mailbox_replica_index = build_memory_replica_index(dist=getattr(runtime, "dist", {}))
            build_stats["mailbox_replica_index_build_wall_seconds"] = float(time.perf_counter() - t0)
            setattr(runtime, "_cached_mailbox_replica_index", mailbox_replica_index)
        mailbox_runtime = getattr(runtime, "mailbox_runtime", None)
    hook = CTDGMemoryCommitHook(
        memory_replica_index=replica_index,
        mailbox_replica_index=mailbox_replica_index,
        mailbox_runtime=mailbox_runtime,
        wait_mode=str(async_memory_cfg.get("commit_order", "legacy")),
    )
    for key, value in build_stats.items():
        setattr(hook, key, float(value))
    return hook


def _is_edge_prediction(task_name: str) -> bool:
    return task_name in {"edge_prediction", "edge_predict", "link_prediction"}


def _gradient_sync_enabled(ctx: Any) -> bool:
    return _gradient_sync_mode(ctx) != "none"


def _gradient_sync_mode(ctx: Any) -> str:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return "none"
    runtime_cfg = ctx.config.get("runtime", {})
    raw = runtime_cfg.get("gradient_sync", runtime_cfg.get("sync_gradients", "async"))
    mode = str(raw).strip().lower()
    if mode in {"0", "false", "none", "off", "disabled"}:
        return "none"
    if mode == "ddp":
        return "ddp"
    return "async"


def _wrap_ddp(module: torch.nn.Module, ctx: Any) -> torch.nn.Module:
    if hasattr(module, "module"):
        return module
    device = torch.device(ctx.device)
    find_unused = bool(ctx.config.get("runtime", {}).get("ddp_find_unused_parameters", False))
    if device.type == "cuda":
        return DDP(module, device_ids=[device.index], output_device=device.index, find_unused_parameters=find_unused)
    return DDP(module, find_unused_parameters=find_unused)


def _sync_module_state(module: torch.nn.Module) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ATC runtime entrypoint.")
    parser.add_argument("action", choices=("prepare", "train", "eval", "predict", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _epochs(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    train_cfg = config.get("train", {})
    if isinstance(train_cfg, dict):
        return int(train_cfg.get("epochs", train_cfg.get("epoch", 1)))
    return 1


def _runtime_config(config_path: str) -> dict[str, Any]:
    config_dir = Path(config_path).expanduser().resolve().parent
    config = normalize_config(config_path)
    graph = dict(config.get("graph", {}))
    source = graph.get("source") or graph.get("path")
    if source is not None:
        path = _resolve_input_source(source, graph=graph, config_dir=config_dir)
        if str(graph.get("mode")) == "dtdg" and (path.is_dir() or path.suffix.lower() in {".csv", ".edges"}):
            degree_xy = bool(graph.get("degree_xy", graph.get("generate_degree_xy", True)))
            graph["source"] = _event_source_to_snapshot_graph(
                path,
                num_snapshots=int(graph.get("event_snapshot_bins", graph.get("num_snapshots", 16))),
                max_events=int(graph.get("max_events", 0)),
                snapshot_lags=int(graph.get("event_snapshot_lags", graph.get("snapshot_lags", 0))),
                compact_nodes=bool(graph.get("compact_nodes", True)),
                degree_xy=degree_xy,
                node_feat_source=str(graph.get("node_feat_source", graph.get("x_source", "snapshot_degree" if degree_xy else "file"))),
                node_label_source=str(graph.get("node_label_source", graph.get("y_source", "snapshot_next_in_degree" if degree_xy else "file"))),
            )
        else:
            graph["source"] = str(path)
    config["graph"] = graph
    return config


def _apply_runtime_threading(config: dict[str, Any]) -> None:
    runtime_cfg = dict(config.get("runtime", {}))
    thread_env = {
        "OMP_NUM_THREADS": runtime_cfg.get("omp_num_threads"),
        "MKL_NUM_THREADS": runtime_cfg.get("mkl_num_threads"),
        "OPENBLAS_NUM_THREADS": runtime_cfg.get("openblas_num_threads"),
        "NUMEXPR_NUM_THREADS": runtime_cfg.get("numexpr_num_threads"),
    }
    for key, value in thread_env.items():
        if value is None:
            continue
        os.environ[str(key)] = str(int(value))
    torch_threads = runtime_cfg.get("torch_num_threads")
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    interop_threads = runtime_cfg.get("torch_num_interop_threads")
    if interop_threads is not None:
        torch.set_num_interop_threads(int(interop_threads))


def _resolve_input_source(source: Any, *, graph: dict[str, Any], config_dir: Path) -> Path:
    path = Path(str(source)).expanduser()
    if path.exists():
        return path
    if path.is_absolute():
        return path
    roots = []
    for key in ("root", "data_root", "dataset_root"):
        if graph.get(key) is not None:
            roots.append(Path(str(graph[key])).expanduser())
    roots.append(config_dir)
    candidates = [
        root / str(source)
        for root in roots
    ]
    candidates.extend(root / "TGL-DATA" / str(source).upper() for root in roots)
    candidates.extend(root / "TGL-DATA" / str(source) for root in roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _event_source_to_snapshot_graph(
    path: Path,
    *,
    num_snapshots: int,
    max_events: int,
    snapshot_lags: int = 0,
    compact_nodes: bool = True,
    degree_xy: bool = True,
    node_feat_source: str = "snapshot_degree",
    node_label_source: str = "snapshot_next_in_degree",
) -> dict[str, Any]:
    from atc_starrygl_lib.preprocess.dataset import build_dataset

    if num_snapshots <= 0:
        raise ValueError("graph.event_snapshot_bins must be positive for event snapshot input")
    graph = build_dataset(data=path, mode="event")
    src = graph["src"].long()
    dst = graph["dst"].long()
    ts = graph["ts"].float()
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32).cpu().contiguous()
    if max_events > 0:
        keep = slice(0, int(max_events))
        src = src[keep]
        dst = dst[keep]
        ts = ts[keep]
        if edge_weight is not None:
            edge_weight = edge_weight[keep]
    if bool(compact_nodes) and src.numel() > 0:
        nodes = torch.cat([src, dst]).long()
        unique, inverse = torch.unique(nodes, sorted=True, return_inverse=True)
        src = inverse[: int(src.numel())].long()
        dst = inverse[int(src.numel()) :].long()
        num_nodes = int(unique.numel())
    else:
        num_nodes = int(torch.cat([src, dst]).max().item()) + 1 if src.numel() else 0

    snapshots = []
    snapshot_edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    boundaries = torch.linspace(0, int(src.numel()), steps=int(num_snapshots) + 1, dtype=torch.long)
    window_width = max(1, int(snapshot_lags) + 1)
    snapshot_count = max(0, int(num_snapshots) - max(0, int(snapshot_lags)))
    for sid in range(snapshot_count):
        begin = int(boundaries[sid])
        end = int(boundaries[sid + window_width])
        s = src[begin:end]
        d = dst[begin:end]
        w = None if edge_weight is None else edge_weight[begin:end]
        snapshot_edges.append((s, d, w))
        snapshots.append({"src": s, "dst": d, "ts": ts[begin:end]})

    node_feat = graph.get("node_feat")
    if bool(compact_nodes) and node_feat is not None:
        raise ValueError("compact_nodes=True is incompatible with file node features; set graph.compact_nodes=false")
    feat_source = str(node_feat_source).lower()
    label_source = str(node_label_source).lower()
    if feat_source in {"none", "off", "false"}:
        node_feat = None
    elif degree_xy and feat_source in {"snapshot_degree", "degree", "snapshot_x", "flare_degree"}:
        node_feat = torch.stack([
            _snapshot_degree_features(src=s, dst=d, edge_weight=w, num_nodes=num_nodes)
            for s, d, w in snapshot_edges
        ], dim=0).contiguous()
        feat_source = "snapshot_degree"
    elif feat_source in {"file", "label_file", "input", "input_file"}:
        if node_feat is None:
            raise ValueError("graph.degree_xy=false requires input node features for DTDG x")
        node_feat = torch.as_tensor(node_feat, dtype=torch.float32)[:num_nodes].contiguous()
        feat_source = "file_node_feat"
    elif node_feat is None or int(torch.as_tensor(node_feat).size(0)) < num_nodes:
        node_feat = _build_node_features(src=src, dst=dst, num_nodes=num_nodes)
        feat_source = "global_degree"
    else:
        node_feat = torch.as_tensor(node_feat, dtype=torch.float32)[:num_nodes].contiguous()
        feat_source = "file_node_feat"
    node_label = None
    if label_source not in {"none", "off", "false"}:
        if degree_xy and label_source in {"snapshot_next_in_degree", "next_snapshot_in_degree", "snapshot_y", "flare_y"}:
            labels = [
                _snapshot_log_in_degree(src=snapshot_edges[sid + 1][0], dst=snapshot_edges[sid + 1][1], edge_weight=snapshot_edges[sid + 1][2], num_nodes=num_nodes)
                for sid in range(max(0, len(snapshot_edges) - 1))
            ]
            node_label = torch.stack(labels, dim=0).contiguous() if labels else torch.empty((0, num_nodes, 1), dtype=torch.float32)
            snapshots = snapshots[: int(node_label.size(0))]
            if isinstance(node_feat, torch.Tensor) and node_feat.dim() >= 3:
                node_feat = node_feat[: int(node_label.size(0))].contiguous()
            label_source = "snapshot_next_in_degree"
        elif degree_xy and label_source in {"snapshot_in_degree", "snapshot_degree", "degree"}:
            node_label = torch.stack([
                _snapshot_log_in_degree(src=s, dst=d, edge_weight=w, num_nodes=num_nodes)
                for s, d, w in snapshot_edges
            ], dim=0).contiguous()
            label_source = "snapshot_in_degree"
        elif label_source in {"file", "label_file", "input", "input_file"}:
            node_label = graph.get("node_label")
            if node_label is None:
                raise ValueError("graph.degree_xy=false requires an input node label file for DTDG y")
            node_label = torch.as_tensor(node_label, dtype=torch.float32)
            if node_label.dim() == 1:
                node_label = node_label.unsqueeze(1)
            node_label = node_label[:num_nodes].contiguous()
            label_source = "file_node_label"
        else:
            indeg = torch.bincount(dst, minlength=num_nodes).float()
            node_label = torch.log1p(indeg).unsqueeze(1)
            if node_label.numel() > 0 and float(node_label.max().item()) > 0.0:
                node_label = node_label / node_label.max()
            label_source = "global_in_degree"
    edge_weight_out = None
    if edge_weight is not None and snapshots:
        edge_weight_parts = []
        for sid in range(len(snapshots)):
            begin = int(boundaries[sid])
            end = int(boundaries[sid + window_width])
            edge_weight_parts.append(edge_weight[begin:end])
        edge_weight_out = torch.cat(edge_weight_parts, dim=0).contiguous() if edge_weight_parts else None
    return {
        "snapshots": snapshots,
        "num_nodes": num_nodes,
        "node_feat": node_feat,
        "node_label": node_label,
        "edge_weight": edge_weight_out,
        "edge_label": edge_weight_out,
        "node_feat_time_varying": isinstance(node_feat, torch.Tensor) and node_feat.dim() >= 3,
        "node_label_time_varying": isinstance(node_label, torch.Tensor) and node_label.dim() >= 3,
        "node_feat_source": feat_source,
        "node_label_source": None if node_label is None else label_source,
    }


def _snapshot_degree_features(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    weight = torch.ones(int(src.numel()), dtype=torch.float32) if edge_weight is None else edge_weight.float()
    indeg = torch.zeros(int(num_nodes), dtype=torch.float32)
    outdeg = torch.zeros(int(num_nodes), dtype=torch.float32)
    if src.numel() > 0:
        indeg.scatter_add_(0, dst.cpu().long(), weight.cpu())
        outdeg.scatter_add_(0, src.cpu().long(), weight.cpu())
    return torch.stack([indeg, outdeg], dim=1).contiguous()


def _snapshot_log_in_degree(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> torch.Tensor:
    del src
    weight = torch.ones(int(dst.numel()), dtype=torch.float32) if edge_weight is None else edge_weight.float()
    indeg = torch.zeros(int(num_nodes), dtype=torch.float32)
    if dst.numel() > 0:
        indeg.scatter_add_(0, dst.cpu().long(), weight.cpu())
    return torch.log1p(indeg).unsqueeze(1).contiguous()


def _build_node_features(*, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    node_id = torch.arange(num_nodes, dtype=torch.float32)
    denom = max(num_nodes - 1, 1)
    indeg = torch.bincount(dst, minlength=num_nodes).float()
    outdeg = torch.bincount(src, minlength=num_nodes).float()
    if indeg.numel() and float(indeg.max().item()) > 0.0:
        indeg = indeg / indeg.max()
    if outdeg.numel() and float(outdeg.max().item()) > 0.0:
        outdeg = outdeg / outdeg.max()
    return torch.stack([node_id / denom, indeg, outdeg], dim=1).contiguous()


def _reduce_metrics(
    metrics: dict[str, float],
    *,
    device: torch.device,
    time_keys: set[str] | None = None,
    sum_keys: set[str] | None = None,
) -> dict[str, float]:
    if not dist.is_initialized() or not metrics:
        return metrics
    time_keys = set() if time_keys is None else set(time_keys)
    sum_keys = set() if sum_keys is None else set(sum_keys)
    out: dict[str, float] = {}
    for key in sorted(metrics):
        value = torch.tensor(float(metrics[key]), dtype=torch.float64, device=device)
        if key in time_keys:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            out[key] = float(value.item())
            continue
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if key in sum_keys:
            out[key] = float(value.item())
            continue
        out[key] = float(value.item() / float(dist.get_world_size()))
    return out


def _validate_routes(session: TrainingSession, *, split: str, device: torch.device, max_batches: int) -> dict[str, Any]:
    checked = 0
    routed = 0
    send_total = 0
    recv_total = 0
    if max_batches <= 0 or str(session.ctx.config.get("graph", {}).get("mode")) != "dtdg":
        return {"checked": checked, "routed": routed, "send_total": send_total, "recv_total": recv_total}
    for batch in session.iter_batches(split):
        for graph in _graphs_from_batch_graph(batch.graph):
            route = getattr(graph, "route", None)
            if route is None or route.send_index is None:
                continue
            rows = int(graph.num_dst_nodes()) if hasattr(graph, "num_dst_nodes") else int(graph.num_nodes())
            x = torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1)
            y = graph.flare_apply_route(x)
            expected = rows + int(route.recv_len)
            if int(y.size(0)) != expected:
                raise RuntimeError(f"route output rows mismatch: got {int(y.size(0))}, expected {expected}")
            checked += 1
            routed += int(route.send_len > 0 or route.recv_len > 0)
            send_total += int(route.send_len)
            recv_total += int(route.recv_len)
        if checked >= int(max_batches):
            break
    return {"checked": checked, "routed": routed, "send_total": send_total, "recv_total": recv_total}


def _graphs_from_batch_graph(graph: Any) -> list[Any]:
    if isinstance(graph, STGraphWindow):
        return list(graph)
    if isinstance(graph, (list, tuple)):
        out = []
        for item in graph:
            out.extend(_graphs_from_batch_graph(item))
        return out
    return [graph]


def _init_dist(device_arg: str | None) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, world_size, local_rank
    use_cuda = (device_arg or "").startswith("cuda") and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _device(device_arg: str | None, local_rank: int) -> torch.device:
    if device_arg is None:
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if str(device_arg).startswith("cuda"):
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _shutdown_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
