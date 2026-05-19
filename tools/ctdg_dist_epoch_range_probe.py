#!/usr/bin/env python3
"""Probe CTDG distributed train iteration ranges per rank."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tgn_wiki.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=600)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["train"]["batch_size"] = int(args.batch_size)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    cfg["runtime"]["device"] = device
    cfg["dist"]["world_size"] = world_size
    cfg["dist"]["rank"] = rank
    cfg["dist"]["local_rank"] = local_rank
    cfg["dist"]["local_world_size"] = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    cfg["dist"]["backend"] = "nccl"

    from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
    from starry_unigraph.types import DistributedContext, SessionContext

    ctx = SessionContext(
        config=cfg,
        project_root=Path(__file__).resolve().parents[1],
        dist=DistributedContext(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            local_world_size=cfg["dist"]["local_world_size"],
            initialized=True,
        ),
        dataset_path=None,
        artifact_root=Path("artifacts") / cfg["data"]["name"],
    )
    session = CTDGSession()
    session.prepare_data(ctx)
    dist.barrier()
    session.build_runtime(ctx)
    dist.barrier()

    for epoch in range(1, int(args.epochs) + 1):
        num_batches = 0
        nonempty_batches = 0
        num_events = 0
        min_eid = None
        max_eid = None
        first_batch = None
        last_batch = None
        for batch in session.iter_train(ctx):
            num_batches += 1
            if batch.event_ids.numel() == 0:
                continue
            nonempty_batches += 1
            eids = batch.event_ids.cpu()
            bmin = int(eids.min().item())
            bmax = int(eids.max().item())
            num_events += int(eids.numel())
            min_eid = bmin if min_eid is None else min(min_eid, bmin)
            max_eid = bmax if max_eid is None else max(max_eid, bmax)
            entry = (int(batch.index), bmin, bmax, int(eids.numel()))
            if first_batch is None:
                first_batch = entry
            last_batch = entry
        print(
            f"rank={rank} epoch={epoch} batches={num_batches} "
            f"nonempty={nonempty_batches} events={num_events} "
            f"event_range=[{min_eid},{None if max_eid is None else max_eid + 1}) "
            f"first_batch={first_batch} last_batch={last_batch}",
            flush=True,
        )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
