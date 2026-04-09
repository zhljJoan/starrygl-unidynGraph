#!/usr/bin/env python3
"""Multi-GPU CTDG TGN training via torchrun.

Usage:
    torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WikiTalk --epochs 2
    torchrun --nproc_per_node=4 train_tgn_dist.py --dataset WIKI --epochs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


DATASET_CONFIG = {
    "WIKI":     "configs/tgn_wiki.yaml",
    "WikiTalk": "configs/tgn_wikitalk.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--config",     type=str)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()

    cfg_path = args.config or DATASET_CONFIG[args.dataset]
    cfg = load_config(cfg_path)
    if args.epochs:     cfg["train"]["epochs"]     = args.epochs
    if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
    if args.lr:         cfg["train"]["lr"]          = args.lr

    # --- Distributed init via torchrun env vars ---
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    cfg["runtime"]["device"] = device
    cfg["dist"]["world_size"]       = world_size
    cfg["dist"]["rank"]             = rank
    cfg["dist"]["local_rank"]       = local_rank
    cfg["dist"]["local_world_size"] = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    cfg["dist"]["backend"]          = "nccl"

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
        project_root=Path(__file__).parent,
        dist=dist_ctx,
        dataset_path=None,
        artifact_root=Path("artifacts") / cfg["data"]["name"],
    )

    from starry_unigraph.providers.ctdg import CTDGProvider
    provider = CTDGProvider(task_adapter=None)

    if rank == 0:
        print(f"[dist] world_size={world_size}  dataset={cfg['data']['name']}"
              f"  batch_size={cfg['train']['batch_size']}  epochs={cfg['train']['epochs']}", flush=True)

    # prepare (rank 0 writes artifacts, others wait)
    t_prep = time.time()
    provider.prepare_data(ctx)
    dist.barrier()
    if rank == 0:
        print(f"[prepare] {time.time()-t_prep:.1f}s", flush=True)

    t_build = time.time()
    provider.build_runtime(ctx)
    dist.barrier()
    if rank == 0:
        print(f"[build_runtime] {time.time()-t_build:.1f}s", flush=True)

    epochs = cfg["train"]["epochs"]
    best_val_ap = 0.0
    ckpt_path = cfg["runtime"].get("checkpoint", "./checkpoints/latest.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_losses, train_aps = [], []
        n_batches = 0
        t_log = t0

        for batch in provider.build_train_iterator(ctx, split="train"):
            result = provider.run_train_step(batch)
            train_losses.append(result["loss"])
            train_aps.append(result["meta"]["metrics"].get("ap", 0))
            n_batches += 1
            now = time.time()
            if rank == 0 and now - t_log >= 30.0:
                cur = sum(train_losses) / len(train_losses)
                print(f"  [epoch {epoch}] batch {n_batches}  loss={cur:.4f}  elapsed={now-t0:.0f}s", flush=True)
                t_log = now

        dist.barrier()
        train_time = time.time() - t0

        # Aggregate metrics across ranks
        avg_loss = sum(train_losses) / max(len(train_losses), 1)
        avg_ap   = sum(train_aps)   / max(len(train_aps), 1)
        loss_t = torch.tensor(avg_loss, device=device)
        ap_t   = torch.tensor(avg_ap,   device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        dist.all_reduce(ap_t,   op=dist.ReduceOp.AVG)

        # Val
        t1 = time.time()
        val_losses, val_aps = [], []
        for batch in provider.build_eval_iterator(ctx, split="val"):
            result = provider.run_eval_step(batch)
            val_losses.append(result["loss"])
            val_aps.append(result["meta"]["metrics"].get("ap", 0))
        dist.barrier()
        val_time = time.time() - t1

        avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
        avg_val_ap   = sum(val_aps)   / max(len(val_aps), 1)
        vl_t  = torch.tensor(avg_val_loss, device=device)
        vap_t = torch.tensor(avg_val_ap,   device=device)
        dist.all_reduce(vl_t,  op=dist.ReduceOp.AVG)
        dist.all_reduce(vap_t, op=dist.ReduceOp.AVG)

        if rank == 0:
            saved = ""
            if float(vap_t) > best_val_ap:
                best_val_ap = float(vap_t)
                provider.save_checkpoint(ckpt_path)
                saved = "  [saved]"
            print(
                f"Epoch {epoch:3d}/{epochs}"
                f"  train_loss={float(loss_t):.4f}  train_AP={float(ap_t):.4f}"
                f"  batches={n_batches}  train:{train_time:.1f}s"
                f"  val_loss={float(vl_t):.4f}  val_AP={float(vap_t):.4f}"
                f"  (val:{val_time:.1f}s){saved}",
                flush=True,
            )

    if rank == 0:
        print(f"\n[done] best val AP = {best_val_ap:.4f}  checkpoint: {ckpt_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
