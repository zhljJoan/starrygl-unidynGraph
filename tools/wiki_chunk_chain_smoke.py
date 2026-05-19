from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.bench_chunk_native import _build_config, _build_session
from starry_unigraph.distributed import finalize_distributed, initialize_distributed
from starry_unigraph.types import DistributedContext


def _make_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="WIKI",
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        batch_size=args.batch_size,
        fanout=args.fanout,
        chunks_per_partition=args.chunks_per_partition,
        epochs=args.epochs,
        device=args.device,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_windows=args.num_windows,
        workers=max(1, int(args.workers)),
        memory_change_threshold=0.0,
        hot_ratio=0.0,
    )


def _configure_chain(cfg: dict, chain: str) -> None:
    cfg["graph"]["partition"] = "balanced"
    cfg.setdefault("chunk", {})["partition_strategy"] = "balanced"
    if chain == "ctdg":
        cfg["model"]["name"] = "tgn"
        cfg["model"]["family"] = "tgn"
        cfg["model"]["task"] = "temporal_link_prediction"
        cfg["chunk"].pop("runtime", None)
    elif chain == "dtdg":
        cfg["model"]["name"] = "mpnn_lstm"
        cfg["model"]["family"] = "mpnn_lstm"
        cfg["model"]["task"] = "temporal_link_prediction"
        cfg["chunk"]["runtime"] = "stgraph"
        cfg.setdefault("dtdg", {})["num_full_snaps"] = 1
        cfg["dtdg"]["chunk_decay"] = "half"
    else:
        raise ValueError(f"Unknown chain: {chain}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=["ctdg", "dtdg"], required=True)
    parser.add_argument("--data-root", default="/mnt/data/zlj/starrygl-data/ctdg")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-windows", type=int, default=2)
    parser.add_argument("--chunks-per-partition", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-strategy", choices=["random", "edge_predict_mixed"], default="random")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda" and world_size > 1:
        torch.cuda.set_device(local_rank)

    cfg = _build_config(_make_args(args), world_size=world_size, rank=rank, local_rank=local_rank)
    _configure_chain(cfg, args.chain)
    cfg.setdefault("sampler", {})["neg_strategy"] = args.neg_strategy
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
        artifact_root = Path(args.artifact_root).expanduser().resolve()
        session = _build_session(cfg, artifact_root=artifact_root, dist_ctx=dist_ctx)
        result: dict[str, object] = {
            "chain": args.chain,
            "world_size": world_size,
            "rank": rank,
            "device": cfg["runtime"]["device"],
            "artifact_root": str(artifact_root),
            "losses": [],
            "metrics": [],
            "steps": [],
        }

        if rank == 0:
            t0 = time.perf_counter()
            session.prepare_data()
            result["prepare_s"] = time.perf_counter() - t0
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        t0 = time.perf_counter()
        session.build_runtime()
        result["build_runtime_s"] = time.perf_counter() - t0

        for epoch in range(args.epochs):
            epoch_result = session.run_epoch("train")
            result["losses"].append(float(epoch_result["loss"]))
            result["metrics"].append(dict(epoch_result.get("metrics", {})))
            result["steps"].append(int(epoch_result["steps"]))
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "chain": args.chain,
                            "world_size": world_size,
                            "epoch": epoch + 1,
                            "loss": float(epoch_result["loss"]),
                            "metrics": epoch_result.get("metrics", {}),
                            "steps": int(epoch_result["steps"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        losses = result["losses"]
        if losses:
            result["loss_delta"] = float(losses[-1] - losses[0])
            result["non_increasing_last"] = bool(len(losses) < 2 or losses[-1] <= losses[0])
        if rank == 0:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        if world_size > 1:
            finalize_distributed(dist_ctx)


if __name__ == "__main__":
    main()
