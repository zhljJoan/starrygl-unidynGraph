from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from atc_starrygl_lib.core.config import build_context
from atc_starrygl_lib.core.config import normalize_config
from atc_starrygl_lib.core.session import TrainingSession
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook
from atc_starrygl_lib.dtdg.runtime.stgraph_loader import STGraphWindow
from atc_starrygl_lib.runtime.grad_sync import AsyncGradientSyncOptimizer
from atc_starrygl_lib.runtime.unified import artifact_bundle, build_model_and_head, register_builtin_backends


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_dist(args.device)
    device = _device(args.device, local_rank)
    register_builtin_backends()
    config = _runtime_config(args.config)
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
    task = session.task
    train_cfg = dict(ctx.config.get("train", {}))
    lr = float(train_cfg.get("lr", ctx.config.get("optimizer", {}).get("lr", 0.01) if isinstance(ctx.config.get("optimizer"), dict) else 0.01))
    params = list(model.parameters()) + ([] if head is None else list(head.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)
    if _gradient_sync_enabled(ctx):
        optimizer = AsyncGradientSyncOptimizer(optimizer, params)
    route_summary = _validate_routes(
        session,
        split="train",
        device=torch.device(ctx.device),
        max_batches=int(ctx.config.get("runtime", {}).get("route_check_batches", 0)),
    )
    rows = []
    for epoch in range(int(epochs)):
        t0 = time.perf_counter()
        train_metrics = _train_epoch(ctx, session, model, head, task, optimizer)
        train_metrics = dict(train_metrics)
        train_metrics["seconds"] = float(time.perf_counter() - t0)
        train_metrics = _reduce_metrics(train_metrics, device=torch.device(ctx.device), time_keys={"seconds"})
        val_metrics = _eval_with_model(ctx, session, model, head, task, split="val")
        val_metrics = _reduce_metrics(val_metrics, device=torch.device(ctx.device))
        test_metrics = _eval_with_model(ctx, session, model, head, task, split="test")
        test_metrics = _reduce_metrics(test_metrics, device=torch.device(ctx.device))
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "test": test_metrics}
        rows.append(row)
        if int(ctx.rank) == 0:
            print(json.dumps({"rank": int(ctx.rank), **row}, sort_keys=True), flush=True)
    out: dict[str, Any] = {"epochs": rows}
    if route_summary["checked"] > 0:
        out["route"] = route_summary
    return out


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
    commit = CTDGMemoryCommitHook() if bool(ctx.config.get("runtime", {}).get("commit_memory", True)) else None
    return train_epoch(session, model, head, task, optimizer, memory_commit=commit)


def _eval(ctx: Any, session: TrainingSession, *, split: str) -> dict[str, float]:
    model, head = build_model_and_head(ctx, session.backend)
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
    commit = CTDGMemoryCommitHook() if bool(ctx.config.get("runtime", {}).get("eval_updates_memory", False)) else None
    return evaluate(session, model, head, task, split=split, memory_commit=commit)


def _predict(ctx: Any, session: TrainingSession, *, split: str) -> dict[str, int]:
    model, head = build_model_and_head(ctx, session.backend)
    if str(ctx.config["graph"]["mode"]) != "ctdg":
        count = sum(1 for _ in session.iter_batches(split))
        return {"batches": count}
    from atc_starrygl_lib.ctdg.train_loop import predict

    if head is None:
        raise RuntimeError("temporal_sampling path requires a head")
    commit = CTDGMemoryCommitHook() if bool(ctx.config.get("runtime", {}).get("predict_updates_memory", True)) else None
    outputs = predict(session, model, head, split=split, memory_commit=commit)
    return {"batches": len(outputs)}


def _is_edge_prediction(task_name: str) -> bool:
    return task_name in {"edge_prediction", "edge_predict", "link_prediction"}


def _gradient_sync_enabled(ctx: Any) -> bool:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return False
    runtime_cfg = ctx.config.get("runtime", {})
    raw = runtime_cfg.get("gradient_sync", runtime_cfg.get("sync_gradients", "async"))
    return str(raw).strip().lower() not in {"0", "false", "none", "off", "disabled"}


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
            graph["source"] = _event_source_to_snapshot_graph(
                path,
                num_snapshots=int(graph.get("event_snapshot_bins", graph.get("num_snapshots", 16))),
                max_events=int(graph.get("max_events", 0)),
            )
        else:
            graph["source"] = str(path)
    config["graph"] = graph
    return config


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


def _event_source_to_snapshot_graph(path: Path, *, num_snapshots: int, max_events: int) -> dict[str, Any]:
    from atc_starrygl_lib.preprocess.dataset import build_dataset

    if num_snapshots <= 0:
        raise ValueError("graph.event_snapshot_bins must be positive for event snapshot input")
    graph = build_dataset(data=path, mode="event")
    src = graph["src"].long()
    dst = graph["dst"].long()
    ts = graph["ts"].float()
    if max_events > 0:
        keep = slice(0, int(max_events))
        src = src[keep]
        dst = dst[keep]
        ts = ts[keep]
    num_nodes = int(torch.cat([src, dst]).max().item()) + 1 if src.numel() else 0

    snapshots = []
    boundaries = torch.linspace(0, int(src.numel()), steps=int(num_snapshots) + 1, dtype=torch.long)
    for sid in range(int(num_snapshots)):
        begin = int(boundaries[sid])
        end = int(boundaries[sid + 1])
        snapshots.append({"src": src[begin:end], "dst": dst[begin:end], "ts": ts[begin:end]})

    node_feat = graph.get("node_feat")
    if node_feat is None or int(torch.as_tensor(node_feat).size(0)) < num_nodes:
        node_feat = _build_node_features(src=src, dst=dst, num_nodes=num_nodes)
    else:
        node_feat = torch.as_tensor(node_feat, dtype=torch.float32)[:num_nodes].contiguous()
    indeg = torch.bincount(dst, minlength=num_nodes).float()
    node_label = torch.log1p(indeg).unsqueeze(1)
    if node_label.numel() > 0 and float(node_label.max().item()) > 0.0:
        node_label = node_label / node_label.max()
    return {
        "snapshots": snapshots,
        "num_nodes": num_nodes,
        "node_feat": node_feat,
        "node_label": node_label,
    }


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


def _reduce_metrics(metrics: dict[str, float], *, device: torch.device, time_keys: set[str] | None = None) -> dict[str, float]:
    if not dist.is_initialized() or not metrics:
        return metrics
    time_keys = set() if time_keys is None else set(time_keys)
    out: dict[str, float] = {}
    for key in sorted(metrics):
        value = torch.tensor(float(metrics[key]), dtype=torch.float64, device=device)
        if key in time_keys:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            out[key] = float(value.item())
            continue
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
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
