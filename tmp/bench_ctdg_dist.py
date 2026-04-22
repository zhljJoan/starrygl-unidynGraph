#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DATASET_CONFIG = {
    "WIKI": "configs/tgn_wiki.yaml",
    "WikiTalk": "configs/tgn_wikitalk.yaml",
}


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _dist_avg(value: float, device: str) -> float:
    t = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def _dist_sum(value: float, device: str) -> float:
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="WIKI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3000, dest="batch_size")
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=40)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    cfg_path = args.config or DATASET_CONFIG[args.dataset]
    cfg = load_config(cfg_path)
    cfg["runtime"]["device"] = device
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["dist"]["world_size"] = world_size
    cfg["dist"]["rank"] = rank
    cfg["dist"]["local_rank"] = local_rank
    cfg["dist"]["local_world_size"] = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    cfg["dist"]["backend"] = "nccl"

    from starry_unigraph.types import DistributedContext, SessionContext
    dist_ctx = DistributedContext(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_world_size=cfg["dist"]["local_world_size"],
        initialized=True,
    )
    ctx = SessionContext(
        config=cfg,
        project_root=Path(__file__).resolve().parents[1],
        dist=dist_ctx,
        dataset_path=None,
        artifact_root=Path("artifacts") / cfg["data"]["name"],
    )

    from starry_unigraph.runtime.online import CTDGSession
    session = CTDGSession()
    session.prepare_data(ctx)
    dist.barrier()
    session.build_runtime(ctx)
    dist.barrier()

    train_losses: list[float] = []
    train_ap: list[float] = []
    train_auc: list[float] = []
    sync_wait_ms: list[float] = []
    sync_submit_ms: list[float] = []
    step_ms: list[float] = []

    train_batches = 0
    train_events = 0
    t_train0 = time.perf_counter()

    for _ in range(args.epochs):
        for batch in session.iter_train(ctx):
            out = session.train_step(batch)
            train_batches += 1
            train_events += int(batch.size)
            train_losses.append(float(out["loss"]))
            m = out.get("meta", {}).get("metrics", {})
            train_ap.append(float(m.get("ap", 0.0)))
            train_auc.append(float(m.get("auc", 0.0)))
            sync_wait_ms.append(float(out.get("meta", {}).get("sync_wait_ms", 0.0)))
            sync_submit_ms.append(float(out.get("meta", {}).get("sync_submit_ms", 0.0)))
            step_ms.append(float(out.get("meta", {}).get("step_ms", 0.0)))
            if train_batches >= args.max_train_batches:
                break
        if train_batches >= args.max_train_batches:
            break

    dist.barrier()
    train_sec = time.perf_counter() - t_train0

    val_losses: list[float] = []
    val_ap: list[float] = []
    val_auc: list[float] = []
    val_batches = 0
    t_val0 = time.perf_counter()
    for batch in session.iter_eval(ctx, split="val"):
        out = session.eval_step(batch)
        val_batches += 1
        val_losses.append(float(out["loss"]))
        m = out.get("meta", {}).get("metrics", {})
        val_ap.append(float(m.get("ap", 0.0)))
        val_auc.append(float(m.get("auc", 0.0)))
        if val_batches >= args.max_val_batches:
            break
    dist.barrier()
    val_sec = time.perf_counter() - t_val0

    # Aggregate across ranks
    total_events = _dist_sum(float(train_events), device)
    max_train_sec = _dist_sum(float(train_sec), device) / world_size
    throughput_eps = total_events / max(max_train_sec, 1e-9)

    payload = {
        "world_size": world_size,
        "dataset": cfg["data"]["name"],
        "train_batches_per_rank": train_batches,
        "val_batches_per_rank": val_batches,
        "throughput_events_per_sec": throughput_eps,
        "train_time_sec_per_rank": train_sec,
        "val_time_sec_per_rank": val_sec,
        "train_loss": _dist_avg(_mean(train_losses), device),
        "train_ap": _dist_avg(_mean(train_ap), device),
        "train_auc": _dist_avg(_mean(train_auc), device),
        "val_loss": _dist_avg(_mean(val_losses), device),
        "val_ap": _dist_avg(_mean(val_ap), device),
        "val_auc": _dist_avg(_mean(val_auc), device),
        "sync_wait_ms": _dist_avg(_mean(sync_wait_ms), device),
        "sync_submit_ms": _dist_avg(_mean(sync_submit_ms), device),
        "step_ms": _dist_avg(_mean(step_ms), device),
    }

    if rank == 0:
        print(json.dumps(payload, sort_keys=True), flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
