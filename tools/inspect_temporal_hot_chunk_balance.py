from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from atc_starrygl_lib.preprocess.dist import (
    assign_chunks_temporal_hot_balance,
    build_local_load_balanced_chunks,
    compute_chunk_event_load,
)


def main() -> None:
    args = _parse_args()
    root = Path(args.artifact_root)
    graph = torch.load(root / "graph.pt", map_location="cpu")
    dist = torch.load(root / "dist.pt", map_location="cpu")
    src = graph["src"].long().cpu()
    dst = graph["dst"].long().cpu()
    time_ptr_2 = graph["time_ptr_2"].long().cpu()
    world_size = int(dist["world_size"])
    chunks_per_rank = int(args.chunks_per_rank or dist.get("chunks_per_rank", 1))
    base_node_owner = dist["node_owner"].long().cpu()
    hot_node_ids = dist.get("hot_node_ids", torch.empty(0, dtype=torch.long)).long().cpu()

    node_to_chunk, _, _, base_chunk_owner = build_local_load_balanced_chunks(
        src=src,
        dst=dst,
        node_owner=base_node_owner,
        chunks_per_rank=chunks_per_rank,
        world_size=world_size,
    )
    chunk_load = compute_chunk_event_load(
        src=src,
        dst=dst,
        node_to_chunk=node_to_chunk,
        time_ptr_2=time_ptr_2,
        num_chunks=world_size * chunks_per_rank,
    )
    chunk_owner = assign_chunks_temporal_hot_balance(
        chunk_load=chunk_load,
        src=src,
        dst=dst,
        node_to_chunk=node_to_chunk,
        hot_node_ids=hot_node_ids,
        world_size=world_size,
        chunks_per_rank=chunks_per_rank,
        affinity_weight=float(args.affinity_weight),
        local_search_iters=int(args.local_search_iters),
        initial_owner=base_chunk_owner,
    )
    new_node_owner = chunk_owner.index_select(0, node_to_chunk)
    new_edge_owner = chunk_owner.index_select(0, node_to_chunk.index_select(0, dst))
    base_edge_owner = dist["edge_owner"].long().cpu()

    out = {
        "artifact_root": str(root),
        "world_size": world_size,
        "chunks_per_rank": chunks_per_rank,
        "base": _owner_stats(
            src=src,
            dst=dst,
            owner=base_edge_owner,
            node_owner=base_node_owner,
            hot_node_ids=hot_node_ids,
            time_ptr_2=time_ptr_2,
            world_size=world_size,
        ),
        "temporal_hot_chunk_balance": _owner_stats(
            src=src,
            dst=dst,
            owner=new_edge_owner,
            node_owner=new_node_owner,
            hot_node_ids=hot_node_ids,
            time_ptr_2=time_ptr_2,
            world_size=world_size,
        ),
        "chunk_owner": chunk_owner.tolist(),
        "base_chunk_owner": base_chunk_owner.tolist(),
    }
    print(json.dumps(out, sort_keys=True))


def _owner_stats(
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    owner: torch.Tensor,
    node_owner: torch.Tensor,
    hot_node_ids: torch.Tensor,
    time_ptr_2: torch.Tensor,
    world_size: int,
) -> dict[str, Any]:
    edge_counts = torch.bincount(owner, minlength=world_size).float()
    ratios = []
    for begin, end in time_ptr_2.tolist():
        if int(end) <= int(begin):
            continue
        counts = torch.bincount(owner[int(begin) : int(end)], minlength=world_size).float()
        mean = float(counts.mean().item())
        if mean > 0.0:
            ratios.append(float(counts.max().item() / mean))
    ratio_t = torch.tensor(ratios, dtype=torch.float32) if ratios else torch.ones(1)
    hot = torch.zeros(int(node_owner.numel()), dtype=torch.bool)
    if int(hot_node_ids.numel()) > 0:
        hot[hot_node_ids] = True
    non_hot_src = ~hot.index_select(0, src)
    src_remote = node_owner.index_select(0, src) != owner
    dst_remote = node_owner.index_select(0, dst) != owner
    return {
        "edge_counts": [int(v) for v in edge_counts.tolist()],
        "edge_max_over_mean": float(edge_counts.max().item() / max(1.0, edge_counts.mean().item())),
        "time_max_over_mean_mean": float(ratio_t.mean().item()),
        "time_max_over_mean_p95": float(torch.quantile(ratio_t, 0.95).item()),
        "time_max_over_mean_max": float(ratio_t.max().item()),
        "src_remote_ratio": float(src_remote.float().mean().item()),
        "non_hot_src_remote_ratio": float((src_remote & non_hot_src).sum().item() / max(1, int(non_hot_src.sum().item()))),
        "dst_remote_ratio": float(dst_remote.float().mean().item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--chunks-per-rank", type=int, default=None)
    parser.add_argument("--affinity-weight", type=float, default=0.02)
    parser.add_argument("--local-search-iters", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    main()
