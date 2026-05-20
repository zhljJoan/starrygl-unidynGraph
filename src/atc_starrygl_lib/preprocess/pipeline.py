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
    partition_data_dst_node_scope = str(dataset_kwargs.pop("partition_data_dst_node_scope", "active"))
    hot_ratio = float(dataset_kwargs.pop("hot_ratio", 0.0))
    hot_topk = int(dataset_kwargs.pop("hot_topk", 0))
    node_count_weight = float(dataset_kwargs.pop("node_count_weight", 1.0))
    speed_beta = float(dataset_kwargs.pop("speed_beta", 0.5))
    speed_topk_type = str(dataset_kwargs.pop("speed_topk_type", "degree"))
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
        hot_ratio=hot_ratio,
        hot_topk=hot_topk,
        node_count_weight=node_count_weight,
        speed_beta=speed_beta,
        speed_topk_type=speed_topk_type,
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
        node_feat_time_varying=bool(graph.get("node_feat_time_varying", False)),
        node_label_time_varying=bool(graph.get("node_label_time_varying", False)),
    ) if build_feature else []
    pds = build_all_partition_data_artifacts(
        rank_artifacts=ranks,
        dist_plan=dist,
        src=graph["src"],
        dst=graph["dst"],
        time_ptr_2=graph["time_ptr_2"],
        edge_ids=graph["edge_ids"],
        node_feat=graph.get("node_feat"),
        node_feat_time_varying=bool(graph.get("node_feat_time_varying", False)),
        edge_feat=graph.get("edge_feat"),
        node_label=graph.get("node_label"),
        node_label_time_varying=bool(graph.get("node_label_time_varying", False)),
        edge_label=graph.get("edge_label"),
        edge_weight=graph.get("edge_weight"),
        dst_node_scope=partition_data_dst_node_scope,
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
