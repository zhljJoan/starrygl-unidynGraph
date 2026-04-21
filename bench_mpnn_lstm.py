#!/usr/bin/env python3
"""Quick benchmark: prepare + 5-epoch train on rec-amazon-ratings with 4 GPUs.
Prints per-epoch loss and wall-clock time. Uses snaps=20 for fast turnaround."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_ROOT = Path("/mnt/data/zlj/starrygl-artifacts/rec-amazon-ratings-bench")
CONFIG_PATH = PROJECT_ROOT / "configs" / "mpnn_lstm_bench.yaml"


def _is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _is_main():
        print(f"[rank0] {msg}", flush=True)


def _build_session():
    from starry_unigraph.config.schema import load_config, validate_config
    from starry_unigraph.distributed import apply_distributed_env, build_distributed_context
    from starry_unigraph.registry import ModelRegistry, TaskRegistry
    from starry_unigraph.types import SessionContext
    from starry_unigraph.session import SchedulerSession

    config = apply_distributed_env(load_config(CONFIG_PATH))
    validate_config(config)
    model_spec = ModelRegistry.resolve(model_name=config["model"]["name"], family=config["model"]["family"])
    task_cls = TaskRegistry.resolve(config["model"]["task"])
    ctx = SessionContext(
        config=config,
        project_root=PROJECT_ROOT,
        dataset_path=None,
        artifact_root=ARTIFACT_ROOT,
        dist=build_distributed_context(config),
        warnings=[],
    )
    return SchedulerSession(session_ctx=ctx, model_spec=model_spec, task_adapter=task_cls())


def run_prepare():
    _log(f"=== PREPARE (snaps=20) → {ARTIFACT_ROOT} ===")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    session = _build_session()
    t0 = time.perf_counter()
    session.prepare_data()
    _log(f"Prepare done in {time.perf_counter()-t0:.1f}s")


def run_bench():
    from starry_unigraph.distributed import initialize_distributed, finalize_distributed
    session = _build_session()
    initialize_distributed(session.ctx.dist)
    epochs = session.ctx.config["train"]["epochs"]
    snaps = session.ctx.config["train"]["snaps"]
    _log(f"=== BENCH: {epochs} epochs, snaps={snaps}, 4 GPUs ===")
    try:
        session.build_runtime()
        _log("Model ready. Starting timed epochs...")

        epoch_times = []
        for ep in range(epochs):
            t0 = time.perf_counter()
            tr = session.run_epoch(split="train")
            t_train = time.perf_counter() - t0

            t1 = time.perf_counter()
            ev = session.run_epoch(split="val")
            t_eval = time.perf_counter() - t1

            epoch_times.append(t_train)
            if _is_main():
                print(
                    f"Epoch {ep+1:2d}/{epochs} | "
                    f"train_loss={tr['loss']:.6f} {t_train:.3f}s | "
                    f"val_loss={ev['loss']:.6f} {t_eval:.3f}s",
                    flush=True,
                )

        if _is_main() and epoch_times:
            # skip first epoch (warmup)
            steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
            avg = sum(steady) / len(steady)
            print(f"\n=== Summary ===", flush=True)
            print(f"  Avg train epoch (ep2+): {avg:.3f}s", flush=True)
            print(f"  Min: {min(steady):.3f}s  Max: {max(steady):.3f}s", flush=True)
    finally:
        finalize_distributed(session.ctx.dist)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "bench", "all"], default="all")
    args = p.parse_args()
    if args.mode in ("prepare", "all"):
        run_prepare()
    if args.mode in ("bench", "all"):
        run_bench()
