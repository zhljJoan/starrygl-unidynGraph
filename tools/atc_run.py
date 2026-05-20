from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from atc_starrygl_lib.core.config import build_context
from atc_starrygl_lib.core.session import TrainingSession
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook
from atc_starrygl_lib.runtime.unified import artifact_bundle, build_model_and_head, register_builtin_backends


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_dist(args.device)
    device = _device(args.device, local_rank)
    register_builtin_backends()
    ctx = build_context(
        args.config,
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
        summary = _train_eval(ctx, session, epochs=int(args.epochs))
    elif action == "eval":
        summary = {"eval": _eval(ctx, session, split=args.split)}
    elif action == "predict":
        summary = {"predict": _predict(ctx, session, split=args.split)}
    else:
        raise ValueError(f"unknown action: {action!r}")

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
    rows = []
    for epoch in range(int(epochs)):
        train_metrics = _train_epoch(ctx, session, model, head, task, optimizer)
        val_metrics = _eval_with_model(ctx, session, model, head, task, split="val")
        rows.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
    return {"epochs": rows, "test": _eval_with_model(ctx, session, model, head, task, split="test")}


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ATC runtime entrypoint.")
    parser.add_argument("action", choices=("prepare", "train", "eval", "predict", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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
