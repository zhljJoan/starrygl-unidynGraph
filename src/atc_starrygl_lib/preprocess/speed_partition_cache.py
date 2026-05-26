from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def find_speed_partition_cache_dir(
    *,
    root: str | Path,
    dataset_name: str,
    world_size: int,
    topk: float,
    seed: int = 123457,
) -> Path:
    root = Path(root)
    topk_str = _format_topk(topk)
    dataset_dir = root / dataset_name / str(seed) / f"{dataset_name}_{int(world_size)}parts_top{topk_str}"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"speed_partition cache directory not found: {dataset_dir}")
    return dataset_dir


def load_speed_partition_cache(
    *,
    root: str | Path,
    dataset_name: str,
    num_nodes: int,
    num_edges: int,
    world_size: int,
    topk: float,
    seed: int = 123457,
) -> dict[str, Any]:
    cache_dir = find_speed_partition_cache_dir(
        root=root,
        dataset_name=dataset_name,
        world_size=world_size,
        topk=topk,
        seed=seed,
    )
    node_owner = torch.full((int(num_nodes),), -1, dtype=torch.long)
    replica_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    for rank in range(int(world_size)):
        path = cache_dir / f"output{rank}.txt"
        for node_id in _read_index_file(path):
            if int(node_owner[node_id]) >= 0:
                replica_mask[node_id] = True
            node_owner[node_id] = rank

    shared_path = cache_dir / "outputshared.txt"
    if shared_path.exists():
        for node_id in _read_index_file(shared_path):
            replica_mask[node_id] = True

    missing_nodes = (node_owner < 0).nonzero(as_tuple=True)[0]
    if int(missing_nodes.numel()) > 0:
        raise ValueError(f"speed_partition cache has {int(missing_nodes.numel())} unassigned nodes in {cache_dir}")

    edge_owner = torch.full((int(num_edges),), -1, dtype=torch.long)
    for rank in range(int(world_size)):
        path = cache_dir / f"edge_output{rank}.txt"
        for edge_id in _read_index_file(path):
            edge_owner[edge_id] = rank

    missing_edges = (edge_owner < 0).nonzero(as_tuple=True)[0]
    if int(missing_edges.numel()) > 0:
        drop_path = cache_dir / "dropeedge.txt"
        if drop_path.exists():
            for edge_id in _read_drop_edge_file(drop_path):
                edge_owner[edge_id] = world_size
        missing_edges = (edge_owner < 0).nonzero(as_tuple=True)[0]
        if int(missing_edges.numel()) > 0:
            raise ValueError(f"speed_partition cache has {int(missing_edges.numel())} unassigned edges in {cache_dir}")

    return {
        "cache_dir": cache_dir,
        "node_owner": node_owner.contiguous(),
        "edge_owner": edge_owner.contiguous(),
        "replica_mask": replica_mask.contiguous(),
    }


def _read_index_file(path: Path) -> list[int]:
    out: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                out.append(int(text))
    return out


def _read_drop_edge_file(path: Path) -> list[int]:
    out: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                out.append(int(text.split()[0]))
    return out


def _format_topk(topk: float) -> str:
    if float(topk).is_integer():
        return str(int(topk))
    return format(float(topk), "g")
