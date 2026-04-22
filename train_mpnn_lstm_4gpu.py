#!/usr/bin/env python3
"""
4-GPU MPNN-LSTM training script on rec-amazon-ratings.

Usage (step 1 — preprocess, single process):
    python train_mpnn_lstm_4gpu.py --mode prepare

Usage (step 2 — train, via torchrun):
    torchrun --nproc_per_node=4 train_mpnn_lstm_4gpu.py --mode train

Usage (step 3 — predict/evaluate, via torchrun):
    torchrun --nproc_per_node=4 train_mpnn_lstm_4gpu.py --mode predict
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_ROOT = Path("/mnt/data/zlj/starrygl-artifacts/rec-amazon-ratings")
CONFIG_PATH = PROJECT_ROOT / "configs" / "mpnn_lstm_4gpu.yaml"


def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _is_main_rank():
        rank_info = os.environ.get("RANK", "0")
        print(f"[rank {rank_info}] {msg}", flush=True)


def _build_session() -> "SchedulerSession":
    from starry_unigraph.config.schema import load_config, validate_config
    from starry_unigraph.distributed import apply_distributed_env, build_distributed_context
    from starry_unigraph.registry import ModelRegistry, TaskRegistry
    from starry_unigraph.types import SessionContext
    from starry_unigraph.session import SchedulerSession

    config = apply_distributed_env(load_config(CONFIG_PATH))
    warnings = validate_config(config)
    if warnings and _is_main_rank():
        for w in warnings:
            print(f"[warn] {w}", flush=True)

    model_spec = ModelRegistry.resolve(
        model_name=config["model"]["name"],
        family=config["model"]["family"],
    )
    task_cls = TaskRegistry.resolve(config["model"]["task"])

    ctx = SessionContext(
        config=config,
        project_root=PROJECT_ROOT,
        dataset_path=None,
        artifact_root=ARTIFACT_ROOT,
        dist=build_distributed_context(config),
        warnings=warnings,
    )
    session = SchedulerSession(
        session_ctx=ctx,
        model_spec=model_spec,
        task_adapter=task_cls(),
    )
    return session


def run_prepare() -> None:
    _log(f"Preparing data → artifacts at {ARTIFACT_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    session = _build_session()
    t0 = time.perf_counter()
    session.prepare_data()
    elapsed = time.perf_counter() - t0
    _log(f"Preprocessing complete in {elapsed:.1f}s")


def _compute_regression_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if not targets or not predictions:
        return {}
    n = min(len(predictions), len(targets))
    preds = predictions[:n]
    tgts = targets[:n]
    mae = sum(abs(p - t) for p, t in zip(preds, tgts)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, tgts)) / n)
    # Normalized MAE: divide by mean of targets
    mean_t = sum(tgts) / n if n else 1.0
    nmae = mae / (abs(mean_t) + 1e-8)
    return {"mae": mae, "rmse": rmse, "nmae": nmae, "n_samples": n}


def run_train() -> None:
    from starry_unigraph.distributed import initialize_distributed, finalize_distributed

    session = _build_session()
    initialize_distributed(session.ctx.dist)

    rank = session.ctx.dist.rank
    world_size = session.ctx.dist.world_size
    epochs = session.ctx.config["train"]["epochs"]

    _log(f"Starting MPNN-LSTM training | world_size={world_size} | epochs={epochs}")
    _log(f"Dataset: rec-amazon-ratings | snaps={session.ctx.config['train']['snaps']}")

    try:
        session.build_runtime()
        _log("Model initialized.")

        for epoch in range(epochs):
            t0 = time.perf_counter()
            train_result = session.run_epoch(split="train")
            train_elapsed = time.perf_counter() - t0

            t1 = time.perf_counter()
            eval_result = session.run_epoch(split="val")
            eval_elapsed = time.perf_counter() - t1

            if _is_main_rank():
                train_loss = train_result.get("loss", float("nan"))
                eval_loss = eval_result.get("loss", float("nan"))
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"train_loss={train_loss:.6f} ({train_elapsed:.2f}s) | "
                    f"val_loss={eval_loss:.6f} ({eval_elapsed:.2f}s)",
                    flush=True,
                )

        _log("Training complete. Saving checkpoint...")
        ckpt_path = Path(session.ctx.config["runtime"]["checkpoint"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        session.save_checkpoint(ckpt_path)
        _log(f"Checkpoint saved to {ckpt_path}")

    finally:
        finalize_distributed(session.ctx.dist)


def run_predict() -> None:
    from starry_unigraph.distributed import initialize_distributed, finalize_distributed

    session = _build_session()
    initialize_distributed(session.ctx.dist)

    try:
        session.build_runtime()

        ckpt_path = Path(session.ctx.config["runtime"]["checkpoint"])
        if ckpt_path.exists():
            session.load_checkpoint(ckpt_path)
            _log(f"Loaded checkpoint from {ckpt_path}")
            # Move model to the correct per-rank GPU after loading
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                local_rank = session.ctx.dist.local_rank
                device = f"cuda:{local_rank}"
                session.runtime.model = session.runtime.model.to(device)
        else:
            _log(f"No checkpoint found at {ckpt_path}, using initialized model")

        _log("Running predict on test split...")
        t0 = time.perf_counter()
        result = session.predict(split="test")
        elapsed = time.perf_counter() - t0

        if _is_main_rank():
            preds = result.predictions
            tgts = result.targets or []
            print(f"\n=== Prediction complete in {elapsed:.2f}s ===")
            print(f"  Samples predicted: {len(preds)}")
            if tgts:
                metrics = _compute_regression_metrics(preds, tgts)
                print(f"  MAE:  {metrics.get('mae', float('nan')):.6f}")
                print(f"  RMSE: {metrics.get('rmse', float('nan')):.6f}")
                print(f"  NMAE: {metrics.get('nmae', float('nan')):.6f}")
                print(f"  n_samples: {int(metrics.get('n_samples', 0))}")
            else:
                print("  (No ground-truth targets available for test split)")
            print(flush=True)
    finally:
        finalize_distributed(session.ctx.dist)


def main() -> None:
    parser = argparse.ArgumentParser(description="MPNN-LSTM 4-GPU training on rec-amazon-ratings")
    parser.add_argument("--mode", choices=["prepare", "train", "predict"], required=True,
                        help="prepare: preprocess data (single proc); train: distributed training; predict: inference + metrics")
    args = parser.parse_args()

    if args.mode == "prepare":
        run_prepare()
    elif args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()


if __name__ == "__main__":
    main()
