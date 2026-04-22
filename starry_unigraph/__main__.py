"""StarryUniGraph unified command-line entry point.

Usage
-----
    python -m starry_unigraph --config CONFIG [--phase PHASE] [--artifact-root PATH]
                              [--epochs N] [--lr F] [--batch-size N] [--device DEV]

Phases
------
    prepare   Single-process preprocessing → writes artifacts to disk (default for dtdg/chunk)
    train     Distributed training + validation  (use torchrun for multi-GPU)
    predict   Load checkpoint, run inference on test split
    all       prepare → train → predict  (single-GPU modes only; dtdg/chunk: run steps separately)

The graph mode (ctdg / dtdg / chunk) is read from ``data.graph_mode`` in the config
file (or inferred from ``model.family``).  There is no ``--mode`` flag: the config file
is the single source of truth.

Examples
--------
    # CTDG — full pipeline, single GPU
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase all

    # DTDG — preprocess (single process)
    python -m starry_unigraph --config configs/mpnn_lstm_4gpu.yaml --phase prepare

    # DTDG — 4-GPU training via torchrun
    torchrun --nproc_per_node=4 -m starry_unigraph \\
        --config configs/mpnn_lstm_4gpu.yaml --phase train

    # Chunk — preprocess
    python -m starry_unigraph --config configs/chunk_default.yaml --phase prepare

    # Override hyperparameters without editing the config file
    torchrun --nproc_per_node=4 -m starry_unigraph \\
        --config configs/mpnn_lstm_4gpu.yaml --phase train \\
        --epochs 100 --lr 0.001
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _log(msg: str) -> None:
    if _is_main():
        print(f"[rank {_rank()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m starry_unigraph",
        description=(
            "StarryUniGraph unified entry point.\n"
            "Graph mode is determined by data.graph_mode in the config file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="YAML config file (sets graph mode via data.graph_mode or model.family)",
    )
    parser.add_argument(
        "--phase", default="all",
        choices=["prepare", "train", "predict", "all"],
        help="Execution phase (default: all)",
    )
    parser.add_argument(
        "--artifact-root", default=None, metavar="PATH",
        help="Override artifact output root directory",
    )
    # Hyperparameter overrides
    parser.add_argument("--epochs",     type=int,   default=None, help="Training epochs")
    parser.add_argument("--lr",         type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size",
                        help="Batch size")
    parser.add_argument("--device",     default=None,
                        help="Compute device (e.g. cuda:0); overrides runtime.device in config")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _build_session(args: argparse.Namespace) -> "SchedulerSession":
    from starry_unigraph.config.schema import load_config, validate_config
    from starry_unigraph.distributed import apply_distributed_env, build_distributed_context
    from starry_unigraph.registry import ModelRegistry, TaskRegistry
    from starry_unigraph.types import SessionContext
    from starry_unigraph.session import SchedulerSession

    # Build override dict from CLI flags
    overrides: dict = {}
    if args.epochs     is not None: overrides.setdefault("train",   {})["epochs"]     = args.epochs
    if args.lr         is not None: overrides.setdefault("train",   {})["lr"]          = args.lr
    if args.batch_size is not None: overrides.setdefault("train",   {})["batch_size"]  = args.batch_size
    if args.device     is not None: overrides.setdefault("runtime", {})["device"]      = args.device

    config = apply_distributed_env(load_config(args.config, overrides=overrides or None))
    warnings = validate_config(config)
    if warnings and _is_main():
        for w in warnings:
            _log(f"[warn] {w}")

    model_spec = ModelRegistry.resolve(
        model_name=config["model"]["name"],
        family=config["model"]["family"],
    )
    task_cls = TaskRegistry.resolve(config["model"]["task"])

    # Artifact root: CLI flag > config checkpoint dir parent > default pattern
    if args.artifact_root:
        artifact_root = Path(args.artifact_root).expanduser().resolve()
    else:
        ckpt = config.get("runtime", {}).get("checkpoint", "")
        if ckpt:
            artifact_root = Path(ckpt).expanduser().resolve().parent.parent
        else:
            artifact_root = (Path.cwd() / "artifacts" / config["data"]["name"]).resolve()

    ctx = SessionContext(
        config=config,
        project_root=Path(args.config).expanduser().resolve().parent,
        dataset_path=None,
        artifact_root=artifact_root,
        dist=build_distributed_context(config),
        warnings=warnings,
    )
    return SchedulerSession(
        session_ctx=ctx,
        model_spec=model_spec,
        task_adapter=task_cls(),
    )


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_prepare(args: argparse.Namespace) -> None:
    session = _build_session(args)
    _log(f"[prepare] artifact_root = {session.ctx.artifact_root}")
    session.ctx.artifact_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    session.prepare_data()
    _log(f"[prepare] done in {time.perf_counter() - t0:.1f}s")


def _run_train(args: argparse.Namespace) -> None:
    from starry_unigraph.distributed import initialize_distributed, finalize_distributed

    session = _build_session(args)
    _log(f"[train] artifact_root = {session.ctx.artifact_root}")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        initialize_distributed(session.ctx.dist)

    try:
        session.build_runtime()
        epochs = session.ctx.config["train"]["epochs"]
        _log(f"[train] graph_mode={session.ctx.config['data'].get('graph_mode')}  epochs={epochs}")

        for epoch in range(epochs):
            t0 = time.perf_counter()
            train_res = session.run_epoch(split="train")
            t1 = time.perf_counter()
            eval_res  = session.run_epoch(split="val")
            if _is_main():
                print(
                    f"epoch {epoch+1:3d}/{epochs} | "
                    f"train_loss={train_res.get('loss', float('nan')):.6f} "
                    f"({t1-t0:.2f}s) | "
                    f"val_loss={eval_res.get('loss', float('nan')):.6f} "
                    f"({time.perf_counter()-t1:.2f}s)",
                    flush=True,
                )

        ckpt = Path(session.ctx.config["runtime"].get("checkpoint", "checkpoint.pt"))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        session.save_checkpoint(ckpt)
        _log(f"[train] checkpoint saved → {ckpt}")
    finally:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            finalize_distributed(session.ctx.dist)


def _run_predict(args: argparse.Namespace) -> None:
    from starry_unigraph.distributed import initialize_distributed, finalize_distributed
    import torch, torch.distributed as tdist

    session = _build_session(args)
    _log(f"[predict] artifact_root = {session.ctx.artifact_root}")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        initialize_distributed(session.ctx.dist)

    try:
        session.build_runtime()

        ckpt = Path(session.ctx.config["runtime"].get("checkpoint", "checkpoint.pt"))
        if ckpt.exists():
            session.load_checkpoint(ckpt)
            _log(f"[predict] loaded checkpoint ← {ckpt}")
            if tdist.is_available() and tdist.is_initialized():
                local_rank = session.ctx.dist.local_rank
                if session.runtime.model is not None:
                    session.runtime.model = session.runtime.model.to(f"cuda:{local_rank}")
        else:
            _log(f"[predict] no checkpoint at {ckpt}, using init model")

        t0 = time.perf_counter()
        result = session.predict(split="test")
        elapsed = time.perf_counter() - t0

        if _is_main():
            preds = result.predictions
            tgts  = result.targets or []
            print(f"\n=== predict done in {elapsed:.2f}s — {len(preds)} samples ===")
            if tgts:
                n = min(len(preds), len(tgts))
                mae  = sum(abs(p - t) for p, t in zip(preds[:n], tgts[:n])) / n
                rmse = math.sqrt(sum((p-t)**2 for p, t in zip(preds[:n], tgts[:n])) / n)
                mean_t = sum(tgts[:n]) / n or 1.0
                print(f"  MAE={mae:.6f}  RMSE={rmse:.6f}  NMAE={mae/abs(mean_t):.6f}")
            print(flush=True)
    finally:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            finalize_distributed(session.ctx.dist)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _log(f"config={args.config}  phase={args.phase}")

    if args.phase in ("prepare", "all"):
        _run_prepare(args)

    if args.phase in ("train", "all"):
        _run_train(args)

    if args.phase in ("predict", "all"):
        _run_predict(args)


if __name__ == "__main__":
    main()
