from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import yaml

from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
from starry_unigraph.types import DistributedContext, SessionContext


def _dist_ctx() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tgn_wiki.yaml")
    parser.add_argument("--data-root", default="/mnt/data/zlj/starrygl-data/ctdg")
    parser.add_argument("--artifact-root", default="/tmp/ctdg_comm_smoke")
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()

    dist_ctx = _dist_ctx()
    if dist_ctx.world_size > 1:
        torch.distributed.init_process_group(backend=dist_ctx.backend)
        torch.cuda.set_device(dist_ctx.local_rank)

    cfg = yaml.safe_load(open(args.config))
    cfg["data"]["root"] = args.data_root
    cfg["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["train"]["batch_size"] = args.batch_size
    cfg["model"]["hidden_dim"] = args.hidden_dim
    cfg["dist"]["backend"] = dist_ctx.backend
    cfg["dist"]["world_size"] = dist_ctx.world_size
    cfg["dist"]["rank"] = dist_ctx.rank
    cfg["dist"]["local_rank"] = dist_ctx.local_rank
    cfg["dist"]["local_world_size"] = dist_ctx.local_world_size

    ctx = SessionContext(
        config=cfg,
        project_root=Path.cwd(),
        dataset_path=None,
        artifact_root=Path(args.artifact_root) / f"rank{dist_ctx.rank}",
        dist=dist_ctx,
    )
    session = CTDGSession()
    session.prepare_data(ctx)
    session.build_runtime(ctx)

    losses: list[float] = []
    aps: list[float] = []
    aucs: list[float] = []
    t0 = time.time()
    for i, batch in enumerate(session.iter_train(ctx)):
        out = session.train_step(batch)
        metrics = out["meta"].get("metrics", {})
        losses.append(float(out["loss"]))
        aps.append(float(metrics.get("ap", 0.0)))
        aucs.append(float(metrics.get("auc", 0.0)))
        if i + 1 >= args.max_batches:
            break

    if dist_ctx.world_size > 1:
        torch.distributed.barrier()
    print(
        {
            "rank": dist_ctx.rank,
            "batches": len(losses),
            "loss": sum(losses) / max(1, len(losses)),
            "ap": sum(aps) / max(1, len(aps)),
            "auc": sum(aucs) / max(1, len(aucs)),
            "seconds": time.time() - t0,
        },
        flush=True,
    )
    if dist_ctx.world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
