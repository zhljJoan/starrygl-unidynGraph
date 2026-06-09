from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fanout", type=int, required=True)
    parser.add_argument("--feature-device", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bts-profile", choices=("auto", "tgn", "tgn_large"), default="auto")
    parser.add_argument("--adaptive-split", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-sync", choices=("async", "ddp", "none"), default="async")
    parser.add_argument(
        "--memshare-memory-sync",
        action="store_true",
        help="Use MemShare historical staged memory/mailbox synchronization settings.",
    )
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("graph", {})
    cfg["graph"]["source"] = args.source
    cfg["graph"]["random_node_feat_dim"] = 172
    cfg["graph"]["random_node_feat_seed"] = args.seed
    cfg["graph"]["random_edge_feat_dim"] = 172
    cfg["graph"]["random_edge_feat_seed"] = args.seed
    cfg["graph"]["random_feature_seed_mode"] = "bts"

    dataset_name = (args.dataset_name or Path(args.source).name).lower()
    profile = args.bts_profile
    if profile == "auto":
        profile = "tgn" if dataset_name in {"wiki", "reddit", "lastfm"} else "tgn_large"

    cfg.setdefault("model", {})
    cfg["model"]["name"] = "general"
    cfg["model"]["hidden_dim"] = 100
    cfg["model"]["memory_dim_time"] = 100
    cfg["model"]["gnn_dim_time"] = 100
    cfg["model"]["gnn_arch"] = "transformer_attention"
    cfg["model"]["layers"] = 1
    cfg["model"]["att_head"] = 2
    cfg["model"]["memory_update"] = "gru"
    cfg["model"]["memory_history"] = 1
    cfg["model"]["combine_node_feature"] = True
    cfg["model"]["dropout"] = 0.2
    cfg["model"]["att_dropout"] = 0.2

    cfg.setdefault("gnn", {}).setdefault("sampling", {})
    cfg["gnn"]["sampling"]["fanouts"] = [args.fanout]
    cfg["gnn"]["sampling"]["policy"] = "boundary_recent_decay"
    cfg["gnn"]["sampling"]["probability"] = 0.1

    cfg.setdefault("task", {})
    cfg["task"]["name"] = "edge_prediction"
    cfg["task"]["batch_size"] = args.batch_size

    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["use_new_pipeline"] = True
    cfg["preprocess"]["partition_algorithm"] = "speed_partition"
    cfg["preprocess"]["hot_ratio"] = 0.1
    cfg["preprocess"]["chunks_per_rank"] = 2
    cfg["preprocess"]["build_feature"] = True
    cfg["preprocess"]["batch_size"] = args.batch_size
    cfg["preprocess"]["split_mode"] = "adaptive" if args.adaptive_split else "fixed"
    cfg["preprocess"]["adaptive_split"] = bool(args.adaptive_split)
    cfg["preprocess"]["adaptive_split_min_batch_size"] = args.batch_size
    cfg["preprocess"]["adaptive_split_max_batch_size"] = args.batch_size * 4
    cfg["preprocess"]["adaptive_split_coherence_chunks"] = 8
    cfg["preprocess"]["adaptive_split_max_chunk_entropy_ratio"] = 0.9
    cfg["preprocess"]["adaptive_split_fallback"] = True
    cfg["preprocess"]["train_ratio"] = 0.7
    cfg["preprocess"]["val_ratio"] = 0.15

    cfg.setdefault("runtime", {})
    cfg["runtime"]["build_sampler"] = True
    cfg["runtime"]["sampler_workers"] = 10
    cfg["runtime"]["torch_num_threads"] = 10
    cfg["runtime"]["torch_num_interop_threads"] = 1
    cfg["runtime"]["omp_num_threads"] = 1
    cfg["runtime"]["mkl_num_threads"] = 1
    cfg["runtime"]["openblas_num_threads"] = 1
    cfg["runtime"]["build_feature_runtime"] = True
    cfg["runtime"]["feature_device"] = args.feature_device
    cfg["runtime"]["chunk_feature_layout"] = True
    cfg["runtime"]["chunk_feature_sort_gather"] = False
    cfg["runtime"]["build_memory_runtime"] = True
    cfg["runtime"]["build_mailbox_runtime"] = True
    cfg["runtime"]["memory_dim"] = 100
    cfg["runtime"]["mailbox_size"] = 1
    cfg["runtime"]["negative_ratio"] = 1
    cfg["runtime"]["negative_sampler_policy"] = "memshare_local"
    cfg["runtime"]["negative_dst_pool"] = "global_dst"
    cfg["runtime"]["negative_train_remote_dst_prob"] = 0.1
    cfg["runtime"]["negative_memshare_beta"] = 0.1
    cfg["runtime"]["negative_test_policy"] = "global"
    cfg["runtime"]["negative_seed"] = args.seed
    cfg["runtime"]["train_drop_last"] = True
    cfg["runtime"]["train_drop_last_batch_size"] = args.batch_size
    cfg["runtime"]["predict_updates_memory"] = True
    cfg["runtime"]["eval_updates_memory"] = True
    cfg["runtime"]["commit_memory"] = True
    cfg["runtime"]["reset_memory_each_epoch"] = True
    cfg["runtime"]["gradient_sync"] = str(args.gradient_sync)
    cfg["runtime"]["mailbox_msg_dim"] = 372
    cfg["runtime"]["train_compute_metrics"] = False
    cfg["runtime"]["schedule_async_commit"] = True
    cfg["runtime"]["prefetch_sample_lookahead"] = 3
    cfg["runtime"]["historical"] = {
        "enabled": True,
        "alpha": 0.3,
        "times_threshold": 10,
    }
    cfg["runtime"]["async_memory"] = {
        "shared_filter": True,
        "staged_commit": bool(args.memshare_memory_sync),
        "commit_order": "memshare" if args.memshare_memory_sync else "legacy",
        "delta_compensation": True,
        "delta_compensation_gamma": 0.5,
    }
    if args.memshare_memory_sync:
        cfg["runtime"]["memory_sync_mode"] = "memshare_historical"
    else:
        cfg["runtime"]["memory_sync_mode"] = "memshare_public_exact"

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["lr"] = 0.0004
    cfg["train"]["weight_decay"] = float(args.weight_decay)
    cfg["train"]["seed"] = args.seed

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
