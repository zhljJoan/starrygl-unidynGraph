from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .dataset import build_dataset
from .dist import build_dist_plan
from .feature import build_all_feature_artifacts
from .partition_data import build_all_partition_data_artifacts
from .rank import build_all_rank_artifacts


def run_preprocess_pipeline(
    *,
    data: Any,
    out_dir: str | Path,
    world_size: int,
    algorithm: str,
    chunks_per_rank: int,
    mode: str = "event",
    build_feature: bool = True,
    build_partition_data: bool = True,
    **dataset_kwargs: Any,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph = build_dataset(data=data, mode=mode, **dataset_kwargs)
    dist = build_dist_plan(
        src=graph["src"],
        dst=graph["dst"],
        ts=graph["ts"],
        num_nodes=graph["num_nodes"],
        world_size=world_size,
        time_ptr_2=graph["time_ptr_2"],
        algorithm=algorithm,
        chunks_per_rank=chunks_per_rank,
    )
    dist, ranks = build_all_rank_artifacts(
        dist_plan=dist,
        src=graph["src"],
        dst=graph["dst"],
        ts=graph["ts"],
        time_ptr_2=graph["time_ptr_2"],
        split_time_ptr=graph.get("split_time_ptr"),
        split=graph["split"],
    )
    feats = build_all_feature_artifacts(
        rank_artifacts=ranks,
        node_feat=graph.get("node_feat"),
        edge_feat=graph.get("edge_feat"),
        node_label=graph.get("node_label"),
        edge_label=graph.get("edge_label"),
    ) if build_feature else []
    pds = build_all_partition_data_artifacts(
        rank_artifacts=ranks,
        dist_plan=dist,
        src=graph["src"],
        dst=graph["dst"],
        time_ptr_2=graph["time_ptr_2"],
        edge_ids=graph["edge_ids"],
        node_feat=graph.get("node_feat"),
        edge_feat=graph.get("edge_feat"),
        node_label=graph.get("node_label"),
        edge_label=graph.get("edge_label"),
        edge_weight=graph.get("edge_weight"),
    ) if build_partition_data else []

    torch.save(graph, out / "graph.pt")
    torch.save(dist, out / "dist.pt")
    for i, rank in enumerate(ranks):
        torch.save(rank, out / f"rank_{i:03d}.pt")
    for i, feat in enumerate(feats):
        torch.save(feat, out / f"feature_{i:03d}.pt")
    for i, pd in enumerate(pds):
        torch.save(pd, out / f"partition_data_{i:03d}.pt")
    meta = {
        "world_size": int(world_size),
        "algorithm": str(algorithm),
        "chunks_per_rank": int(chunks_per_rank),
        "mode": str(mode),
        "num_ranks": len(ranks),
        "has_feature": bool(build_feature),
        "has_partition_data": bool(build_partition_data),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"graph": graph, "dist": dist, "ranks": ranks, "features": feats, "partition_data": pds, "meta": meta}
