#!/usr/bin/env python3
"""Benchmark/smoke entry for chunk CTDG native-sampler training.

Examples:
    python examples/bench_chunk_native.py --dataset WIKI --mode all --device cpu
    python examples/bench_chunk_native.py --dataset WIKI --mode train --device cuda
    torchrun --nproc_per_node=4 examples/bench_chunk_native.py --dataset WikiTalk --mode all --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from starry_unigraph.distributed import finalize_distributed, initialize_distributed
from starry_unigraph.registry import ModelRegistry, TaskRegistry
from starry_unigraph.session import SchedulerSession
from starry_unigraph.types import DistributedContext, SessionContext


DATA_ROOT = "/mnt/data/zlj/starrygl-data/ctdg"


def _is_torchrun() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _build_config(args: argparse.Namespace, world_size: int, rank: int, local_rank: int) -> dict:
    device = args.device
    if device == "cuda" and world_size > 1:
        device = f"cuda:{local_rank}"
    return {
        "model": {
            "name": "tgn",
            "family": "tgn",
            "task": "temporal_link_prediction",
            "hidden_dim": int(args.hidden_dim),
        },
        "data": {
            "root": str(args.data_root),
            "name": args.dataset,
            "format": "auto",
            "graph_mode": "chunk",
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "slice_config": {
                "use_batch_split": False,
                "num_windows": int(args.num_windows),
                "window_size": 0,
                "lags": 0,
                "skip_time": 0,
            },
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "snaps": int(args.num_windows),
            "eval_interval": 1,
        },
        "runtime": {
            "device": device,
            "checkpoint": str(Path(args.artifact_root) / "checkpoints" / "latest.pt"),
        },
        "dist": {
            "backend": "nccl" if world_size > 1 else ("nccl" if args.device == "cuda" else "single"),
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
            "launcher": "torchrun" if world_size > 1 else "single_process",
        },
        "graph": {"partition": "metis", "route": "all2all"},
        "sampler": {
            "neg_strategy": "edge_predict_mixed",
            "train_remote_dst_prob": 0.15,
            "test_policy": "global_average",
            "memshare": {
                "enabled": True,
                "fanout": [int(args.fanout)],
                "num_layers": 1,
                "workers": int(args.workers),
                "policy": "recent",
            },
        },
        "memory": {
            "change_threshold": float(args.memory_change_threshold),
            "change_metric": "cos",
        },
        "chunk": {
            "time_slices": int(args.num_windows),
            "node_clusters": int(args.chunks_per_partition),
            "num_chunks_per_partition": int(args.chunks_per_partition),
            "build_mem_routes": True,
            "num_candidates": 3,
            "hot_ratio": float(args.hot_ratio),
        },
    }


def _build_session(cfg: dict, artifact_root: Path, dist_ctx: DistributedContext) -> SchedulerSession:
    model_spec = ModelRegistry.resolve(cfg["model"]["name"], cfg["model"]["family"])
    task_adapter = TaskRegistry.resolve(cfg["model"]["task"])()
    ctx = SessionContext(
        config=cfg,
        project_root=Path(__file__).resolve().parents[1],
        dataset_path=None,
        artifact_root=artifact_root,
        dist=dist_ctx,
    )
    return SchedulerSession(ctx, model_spec, task_adapter)


def _run_epoch(session: SchedulerSession, split: str) -> dict:
    t0 = time.perf_counter()
    result = session.run_epoch(split=split)
    result = dict(result)
    result["wall_s"] = time.perf_counter() - t0
    outputs = result.pop("outputs", [])
    result["native_sampling_steps"] = sum(
        1 for item in outputs if item.get("meta", {}).get("native_sampling")
    )
    result["sampled_mfg_count_head"] = [
        item.get("meta", {}).get("sampled_mfg_count")
        for item in outputs[:5]
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WIKI", choices=["WIKI", "WikiTalk"])
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--mode", default="all", choices=["prepare", "train", "all"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--chunks-per-partition", type=int, default=4)
    parser.add_argument("--hot-ratio", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--memory-change-threshold", type=float, default=0.0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = _rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    artifact_root = Path(
        args.artifact_root or f"/tmp/starry_chunk_native_{args.dataset}_{world_size}p"
    ).expanduser().resolve()

    if world_size > 1 and args.device != "cuda":
        raise RuntimeError("4-card chunk benchmark requires --device cuda so internal communication uses NCCL")
    if world_size > 1:
        torch.cuda.set_device(local_rank)

    cfg = _build_config(args, world_size=world_size, rank=rank, local_rank=local_rank)
    dist_ctx = DistributedContext(
        backend=cfg["dist"]["backend"],
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_world_size=cfg["dist"]["local_world_size"],
        launcher=cfg["dist"]["launcher"],
    )

    if world_size > 1:
        initialize_distributed(dist_ctx)

    try:
        session = _build_session(cfg, artifact_root=artifact_root, dist_ctx=dist_ctx)
        metrics: dict[str, object] = {
            "dataset": args.dataset,
            "world_size": world_size,
            "rank": rank,
            "artifact_root": str(artifact_root),
            "batch_size": int(args.batch_size),
            "fanout": int(args.fanout),
            "hidden_dim": int(args.hidden_dim),
        }

        if args.mode in {"prepare", "all"}:
            if rank == 0:
                t0 = time.perf_counter()
                session.prepare_data()
                metrics["prepare_s"] = time.perf_counter() - t0
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if args.mode in {"train", "all"}:
            t0 = time.perf_counter()
            session.build_runtime()
            metrics["build_runtime_s"] = time.perf_counter() - t0
            train = _run_epoch(session, "train")
            metrics["train"] = train

        if rank == 0:
            print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    finally:
        if world_size > 1:
            finalize_distributed(dist_ctx)


if __name__ == "__main__":
    main()
