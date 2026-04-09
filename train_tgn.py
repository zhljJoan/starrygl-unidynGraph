#!/usr/bin/env python3
"""
CTDG TGN training script for WIKI / WikiTalk datasets.

Usage:
    # Single GPU, WIKI
    python train_tgn.py --dataset WIKI

    # Single GPU, WikiTalk
    python train_tgn.py --dataset WikiTalk

    # Custom config
    python train_tgn.py --config configs/tgn_wiki.yaml

    # Modes
    python train_tgn.py --dataset WIKI --mode prepare   # preprocess only
    python train_tgn.py --dataset WIKI --mode train     # train + eval (assumes prepare done)
    python train_tgn.py --dataset WIKI --mode all       # prepare + train + eval (default)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.device is not None:
        cfg.setdefault("runtime", {})["device"] = args.device
    return cfg


def build_session(cfg: dict):
    from starry_unigraph.types import DistributedContext, SessionContext

    dist_cfg = cfg.get("dist", {})
    dist_ctx = DistributedContext(
        backend=str(dist_cfg.get("backend", "nccl")),
        world_size=int(dist_cfg.get("world_size", 1)),
        rank=int(dist_cfg.get("rank", 0)),
        local_rank=int(dist_cfg.get("local_rank", 0)),
        local_world_size=int(dist_cfg.get("local_world_size", 1)),
        master_addr=str(dist_cfg.get("master_addr", "127.0.0.1")),
        master_port=int(dist_cfg.get("master_port", 29500)),
        init_method=str(dist_cfg.get("init_method", "env://")),
        launcher=str(dist_cfg.get("launcher", "single_process")),
    )
    return SessionContext(
        config=cfg,
        project_root=Path(__file__).parent,
        dist=dist_ctx,
        dataset_path=None,
        artifact_root=Path("artifacts") / cfg["data"]["name"],
    )


def fmt_metrics(metrics: dict) -> str:
    parts = []
    for k in ("ap", "auc", "mrr"):
        if k in metrics:
            parts.append(f"{k.upper()}={metrics[k]:.4f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_prepare(provider, ctx):
    print("[prepare] preprocessing dataset ...")
    t0 = time.time()
    provider.prepare_data(ctx)
    print(f"[prepare] done in {time.time()-t0:.1f}s")


def phase_train(provider, ctx):
    epochs = int(ctx.config["train"]["epochs"])
    eval_interval = int(ctx.config["train"].get("eval_interval", 1))
    ckpt_path = ctx.config["runtime"].get("checkpoint", "./checkpoints/latest.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    print("[train] building runtime ...")
    t_build = time.time()
    provider.build_runtime(ctx)
    print(f"[train] runtime ready in {time.time()-t_build:.1f}s")

    best_val_ap = 0.0
    for epoch in range(1, epochs + 1):
        # --- train ---
        t0 = time.time()
        train_losses, train_metrics = [], []
        n_batches = 0
        t_last_log = t0
        for batch in provider.build_train_iterator(ctx, split="train"):
            result = provider.run_train_step(batch)
            train_losses.append(result["loss"])
            train_metrics.append(result["meta"]["metrics"])
            n_batches += 1
            # print progress every 30s
            now = time.time()
            if now - t_last_log >= 30.0:
                elapsed = now - t0
                cur_loss = sum(train_losses) / len(train_losses)
                print(f"  [epoch {epoch}] batch {n_batches}  loss={cur_loss:.4f}  elapsed={elapsed:.0f}s")
                t_last_log = now

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)
        avg_train_ap   = sum(m.get("ap",  0) for m in train_metrics) / max(len(train_metrics), 1)
        avg_train_auc  = sum(m.get("auc", 0) for m in train_metrics) / max(len(train_metrics), 1)
        train_time = time.time() - t0

        # --- eval ---
        val_line = ""
        if epoch % eval_interval == 0:
            t1 = time.time()
            val_losses, val_metrics = [], []
            for batch in provider.build_eval_iterator(ctx, split="val"):
                result = provider.run_eval_step(batch)
                val_losses.append(result["loss"])
                val_metrics.append(result["meta"]["metrics"])

            avg_val_loss = sum(val_losses) / max(len(val_losses), 1)
            avg_val_ap   = sum(m.get("ap",  0) for m in val_metrics) / max(len(val_metrics), 1)
            avg_val_auc  = sum(m.get("auc", 0) for m in val_metrics) / max(len(val_metrics), 1)
            val_time = time.time() - t1
            val_line = (
                f"  val_loss={avg_val_loss:.4f}  val_AP={avg_val_ap:.4f}"
                f"  val_AUC={avg_val_auc:.4f}  (val:{val_time:.1f}s)"
            )

            if avg_val_ap > best_val_ap:
                best_val_ap = avg_val_ap
                provider.save_checkpoint(ckpt_path)
                val_line += "  [saved]"

        print(
            f"Epoch {epoch:3d}/{epochs}"
            f"  train_loss={avg_train_loss:.4f}"
            f"  train_AP={avg_train_ap:.4f}"
            f"  train_AUC={avg_train_auc:.4f}"
            f"  batches={n_batches}"
            f"  train:{train_time:.1f}s"
            + val_line
        )

    print(f"\n[train] best val AP = {best_val_ap:.4f}  checkpoint: {ckpt_path}")


def phase_test(provider, ctx):
    ckpt_path = ctx.config["runtime"].get("checkpoint", "./checkpoints/latest.pt")
    if Path(ckpt_path).exists():
        print(f"[test] loading checkpoint: {ckpt_path}")
        provider.load_checkpoint(ckpt_path)

    t0 = time.time()
    test_losses, test_metrics = [], []
    for batch in provider.build_predict_iterator(ctx, split="test"):
        result = provider.run_predict_step(batch)
        test_losses.append(result["loss"])
        test_metrics.append(result["meta"]["metrics"])

    avg_loss = sum(test_losses) / max(len(test_losses), 1)
    avg_ap   = sum(m.get("ap",  0) for m in test_metrics) / max(len(test_metrics), 1)
    avg_auc  = sum(m.get("auc", 0) for m in test_metrics) / max(len(test_metrics), 1)
    avg_mrr  = sum(m.get("mrr", 0) for m in test_metrics) / max(len(test_metrics), 1)
    print(
        f"[test]  loss={avg_loss:.4f}"
        f"  AP={avg_ap:.4f}  AUC={avg_auc:.4f}  MRR={avg_mrr:.4f}"
        f"  ({time.time()-t0:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "WIKI":     "configs/tgn_wiki.yaml",
    "WikiTalk": "configs/tgn_wikitalk.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="TGN CTDG training on WIKI/WikiTalk")
    parser.add_argument("--dataset",    type=str, choices=list(DATASET_CONFIG.keys()),
                        help="Dataset name (WIKI or WikiTalk)")
    parser.add_argument("--config",     type=str, help="Path to yaml config (overrides --dataset)")
    parser.add_argument("--mode",       type=str, default="all",
                        choices=["prepare", "train", "test", "all"],
                        help="Which phase(s) to run (default: all = prepare+train+test)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--device",     type=str,   default=None, help="cpu | cuda | cuda:N")
    args = parser.parse_args()

    if args.config:
        cfg_path = args.config
    elif args.dataset:
        cfg_path = DATASET_CONFIG[args.dataset]
    else:
        parser.error("Provide --dataset or --config")

    cfg = load_config(cfg_path)
    cfg = override_config(cfg, args)

    # Auto-detect device if not specified
    if cfg.get("runtime", {}).get("device") == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        cfg["runtime"]["device"] = "cpu"
        cfg["dist"]["backend"] = "gloo"

    print(f"[config] {cfg_path}")
    print(f"[config] dataset={cfg['data']['name']}  device={cfg['runtime']['device']}"
          f"  epochs={cfg['train']['epochs']}  batch_size={cfg['train']['batch_size']}"
          f"  lr={cfg['train']['lr']}"
          f"  hidden_dim={cfg['model']['hidden_dim']}  neighbor={cfg['sampling']['neighbor_limit']}"
          f"  mailbox_slots={cfg['ctdg']['mailbox_slots']}")

    from starry_unigraph.providers.ctdg import CTDGProvider
    ctx = build_session(cfg)
    provider = CTDGProvider(task_adapter=None)

    if args.mode in ("prepare", "all"):
        phase_prepare(provider, ctx)

    if args.mode in ("train", "all"):
        phase_train(provider, ctx)

    if args.mode in ("test", "all"):
        phase_test(provider, ctx)


if __name__ == "__main__":
    main()
