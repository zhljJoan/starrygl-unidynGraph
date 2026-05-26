from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from atc_starrygl_lib.preprocess.dataset import build_dataset
from atc_starrygl_lib.preprocess.dist import build_dist_plan
from atc_starrygl_lib.preprocess.partition_metrics import (
    aggregate_rank_loads,
    compute_rank_load_from_edge_owner,
    count_cross_partition_edges,
    summarize_rank_loads,
)
from atc_starrygl_lib.preprocess.speed_partition_cache import load_speed_partition_cache


DEFAULT_DATASETS: dict[str, dict[str, Any]] = {
    "wikitalk": {
        "path": "/mnt/data/zlj/tgl_data/DATA/wikitalk",
        "batch_size": 12000,
    },
    "stackoverflow": {
        "path": "/mnt/data/zlj/tgl_data/DATA/stackoverflow",
        "batch_size": 12000,
    },
}
DEFAULT_ALGORITHMS = ("speed_partition", "metis", "chunk_load_balance")
DEFAULT_TOPK_VALUES = (0.1, 0.0)
DEFAULT_SPEED_CACHE_ROOT = "../SPEED/partition/divided_nodes_seed_starrygl"


def _parse_name_value_int(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        name, raw = value.split("=", 1)
        out[name.strip().lower()] = int(raw)
    return out


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def evaluate_dataset(
    *,
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    algorithm: str,
    topk: float,
    world_size: int,
    chunks_per_rank: int,
    train_ratio: float,
    val_ratio: float,
    node_count_weight: float,
    speed_beta: float,
    speed_topk_type: str,
    reuse_speed_partition_cache: bool,
    speed_cache_root: str,
    speed_cache_seed: int,
) -> dict[str, Any]:
    graph = build_dataset(
        data=dataset_path,
        mode="event",
        batch_size=int(batch_size),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
    )
    speed_cache_dir = None
    if algorithm == "speed_partition" and bool(reuse_speed_partition_cache):
        cache = load_speed_partition_cache(
            root=speed_cache_root,
            dataset_name=dataset_name_to_cache_key(dataset_name),
            num_nodes=int(graph["num_nodes"]),
            num_edges=int(graph["src"].numel()),
            world_size=int(world_size),
            topk=float(topk),
            seed=int(speed_cache_seed),
        )
        node_owner = cache["node_owner"]
        edge_owner = cache["edge_owner"]
        speed_cache_dir = str(cache["cache_dir"])
        rank_load = compute_rank_load_from_edge_owner(
            src=graph["src"],
            dst=graph["dst"],
            edge_owner=edge_owner,
            time_ptr_2=graph["time_ptr_2"],
            world_size=int(world_size),
            node_count_weight=float(node_count_weight),
        )
    else:
        dist = build_dist_plan(
            src=graph["src"],
            dst=graph["dst"],
            ts=graph["ts"],
            num_nodes=int(graph["num_nodes"]),
            world_size=int(world_size),
            time_ptr_2=graph["time_ptr_2"],
            algorithm=algorithm,
            chunks_per_rank=int(chunks_per_rank),
            hot_ratio=float(topk),
            hot_topk=0,
            node_count_weight=float(node_count_weight),
            speed_beta=float(speed_beta),
            speed_topk_type=str(speed_topk_type),
        )
        node_owner = dist["node_owner"]
        rank_load = aggregate_rank_loads(
            chunk_load=dist["chunk_load"],
            chunk_owner=dist["chunk_owner"],
            world_size=int(world_size),
        )
    cross_edges = count_cross_partition_edges(
        src=graph["src"],
        dst=graph["dst"],
        node_owner=node_owner,
    )
    load_summary = summarize_rank_loads(rank_load)
    num_edges = int(graph["src"].numel())
    return {
        "dataset": dataset_name,
        "path": str(dataset_path),
        "batch_size": int(batch_size),
        "algorithm": str(algorithm),
        "topk": float(topk),
        "world_size": int(world_size),
        "chunks_per_rank": int(chunks_per_rank),
        "speed_cache_dir": speed_cache_dir,
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": num_edges,
        "cross_partition_edges": int(cross_edges),
        "cross_partition_ratio": float(cross_edges / num_edges) if num_edges > 0 else 0.0,
        **load_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cross-partition edge counts and average per-batch load ratios "
            "for graph partition algorithms. Batch load ratio is computed as "
            "max(rank_load) / min(nonzero_rank_load), and batches with zero-load ranks "
            "are counted separately."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["wikitalk", "stackoverflow"])
    parser.add_argument("--dataset-batch-size", action="append", default=[], metavar="NAME=SIZE")
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--topk-values", nargs="+", type=float, default=list(DEFAULT_TOPK_VALUES))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--chunks-per-rank", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--node-count-weight", type=float, default=1.0)
    parser.add_argument("--speed-beta", type=float, default=0.5)
    parser.add_argument("--speed-topk-type", default="degree")
    parser.add_argument("--reuse-speed-partition-cache", action="store_true")
    parser.add_argument("--speed-cache-root", default=DEFAULT_SPEED_CACHE_ROOT)
    parser.add_argument("--speed-cache-seed", type=int, default=123457)
    parser.add_argument("--compare-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def dataset_name_to_cache_key(dataset_name: str) -> str:
    mapping = {
        "wikitalk": "WikiTalk",
        "stackoverflow": "StackOverflow",
    }
    key = str(dataset_name).lower()
    if key not in mapping:
        raise ValueError(f"unsupported speed_partition cache dataset: {dataset_name}")
    return mapping[key]


def build_compare_text(results: list[dict[str, Any]]) -> str:
    lines = [
        "| dataset | algorithm | topk | cross_partition_edges | cross_partition_ratio | avg_batch_load_ratio | zero_min_load_batches | speed_cache_dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["algorithm"]),
                    f'{float(row["topk"]):.2f}',
                    str(int(row["cross_partition_edges"])),
                    f'{float(row["cross_partition_ratio"]):.6f}',
                    _format_float(float(row["avg_batch_load_ratio"])),
                    str(int(row["zero_min_load_batches"])),
                    str(row.get("speed_cache_dir") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    dataset_batch_sizes = _parse_name_value_int(list(args.dataset_batch_size))
    results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        key = str(dataset_name).lower()
        if key not in DEFAULT_DATASETS:
            raise ValueError(f"unsupported dataset: {dataset_name}")
        spec = DEFAULT_DATASETS[key]
        batch_size = dataset_batch_sizes.get(key, int(spec["batch_size"]))
        for algorithm in args.algorithms:
            for topk in args.topk_values:
                results.append(
                    evaluate_dataset(
                        dataset_name=key,
                        dataset_path=str(spec["path"]),
                        batch_size=batch_size,
                        algorithm=str(algorithm),
                        topk=float(topk),
                        world_size=int(args.world_size),
                        chunks_per_rank=int(args.chunks_per_rank),
                        train_ratio=float(args.train_ratio),
                        val_ratio=float(args.val_ratio),
                        node_count_weight=float(args.node_count_weight),
                        speed_beta=float(args.speed_beta),
                        speed_topk_type=str(args.speed_topk_type),
                        reuse_speed_partition_cache=bool(args.reuse_speed_partition_cache),
                        speed_cache_root=str(args.speed_cache_root),
                        speed_cache_seed=int(args.speed_cache_seed),
                    )
                )

    header = (
        f"{'dataset':<14} {'algorithm':<20} {'topk':>6} {'cross_edges':>14} "
        f"{'cross_ratio':>12} {'avg_load_ratio':>16} {'zero_min_batches':>17}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['dataset']:<14} {row['algorithm']:<20} {row['topk']:>6.2f} "
            f"{row['cross_partition_edges']:>14d} {row['cross_partition_ratio']:>12.6f} "
            f"{_format_float(float(row['avg_batch_load_ratio'])):>16} "
            f"{row['zero_min_load_batches']:>17d}"
        )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.compare_out is not None:
        args.compare_out.write_text(build_compare_text(results), encoding="utf-8")


if __name__ == "__main__":
    main()
