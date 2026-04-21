"""DTDG data preparation: partitioning, block building, and Flare export.

Public API:

- :func:`build_dtdg_partitions` — run graph partitioning (METIS or random)
  on the merged lifetime edge index.
- :func:`build_flare_partition_data_list` — build per-partition
  :class:`PartitionData` objects with node features, labels, GCN norms,
  routes, and chunk assignments.
- :class:`WebDataLoader` — load temporal edge-list files (various formats)
  into snapshot datasets.
- :func:`build_web_data_loader` — create a :class:`WebDataLoader` with
  per-dataset presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import dgl
import torch
try:
    import numpy as np
except Exception:
    np = None

try:
    import scipy.sparse as sp
except Exception:
    sp = None

try:
    import pymetis
except Exception:
    pymetis = None

from starry_unigraph.data.partition import PartitionData, TensorData
from starry_unigraph.runtime.flare.route import Route
from starry_unigraph.preprocess.dtdg import SnapshotRoutePlan


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _split_line(line: str, sep: str) -> list[str]:
    if sep == ",":
        return [part.strip() for part in line.split(",") if part.strip()]
    return [part for part in re.split(r"\s+", line.strip()) if part]


def _compute_degrees(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        zeros = torch.zeros(num_nodes, dtype=torch.float32)
        return zeros, zeros
    src = edge_index[0].long()
    dst = edge_index[1].long()
    weights = edge_weight.float()
    in_deg = torch.bincount(dst, weights=weights, minlength=num_nodes).float()
    out_deg = torch.bincount(src, weights=weights, minlength=num_nodes).float()
    return in_deg, out_deg


def _to_cpu_long(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", dtype=torch.long, non_blocking=False)


def _merge_snapshot_edges(snapshots: list[dict[str, Any]]) -> torch.Tensor:
    edge_indexes = [snapshot["edge_index"].long() for snapshot in snapshots if snapshot["edge_index"].numel() > 0]
    if not edge_indexes:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.cat(edge_indexes, dim=1)


def _build_lifetime_graph(snapshots: list[dict[str, Any]], num_nodes: int) -> dgl.DGLGraph:
    merged_edge_index = _merge_snapshot_edges(snapshots)
    return dgl.graph(
        (merged_edge_index[0], merged_edge_index[1]),
        num_nodes=max(0, int(num_nodes)),
        idtype=torch.int64,
    )


def _induced_local_edges(graph: dgl.DGLGraph, local_nodes: torch.Tensor) -> torch.Tensor:
    if local_nodes.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    subgraph = dgl.node_subgraph(graph, local_nodes.long(), relabel_nodes=True, store_ids=False)
    src, dst = subgraph.edges(order="eid")
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.stack([src.long(), dst.long()], dim=0)


def apply_partition(edge_index: torch.Tensor, num_nodes: int, num_parts: int) -> torch.Tensor:
    """Partition a graph into ``num_parts`` using METIS (or random fallback).

    Args:
        edge_index: ``[2, E]`` edge list tensor.
        num_nodes: Total number of nodes.
        num_parts: Desired number of partitions.

    Returns:
        Tensor ``[num_nodes]`` mapping each node to a partition ID
        in ``[0, num_parts)``.
    """
    num_nodes = max(0, int(num_nodes))
    num_parts = max(1, int(num_parts))
    if num_nodes == 0:
        return torch.zeros((0,), dtype=torch.long)
    if num_parts <= 1:
        return torch.zeros((num_nodes,), dtype=torch.long)

    src_t, dst_t = _to_cpu_long(edge_index[0]), _to_cpu_long(edge_index[1])
    if pymetis is None or np is None or sp is None:
        print("[Partition] pymetis/numpy/scipy unavailable, fallback random partition.")
        return torch.randint(0, num_parts, (num_nodes,), dtype=torch.long)

    u_np = src_t.numpy()
    v_np = dst_t.numpy()
    ones = np.ones(len(u_np), dtype=np.int32)
    adj_matrix = sp.csr_matrix((ones, (u_np, v_np)), shape=(num_nodes, num_nodes))
    adj_matrix = (adj_matrix + adj_matrix.T).astype(np.int32)
    xadj = adj_matrix.indptr.astype(np.int32)
    adjncy = adj_matrix.indices.astype(np.int32)
    try:
        cuts, membership = pymetis.part_graph(nparts=num_parts, xadj=xadj, adjncy=adjncy)
        print(f"[Partition] pymetis success, edge cuts={cuts}")
        return torch.tensor(membership, dtype=torch.long)
    except Exception as exc:
        print(f"[Partition] pymetis failed: {exc}. fallback random partition.")
        return torch.randint(0, num_parts, (num_nodes,), dtype=torch.long)


def normalize_snapshot_count(raw_dataset: dict[str, Any], snaps: int) -> dict[str, Any]:
    dataset = list(raw_dataset.get("dataset", []))
    if not dataset:
        return raw_dataset
    if len(dataset) > snaps:
        dataset = dataset[:snaps]
    while len(dataset) < snaps:
        template = dataset[-1]
        dataset.append(
            {
                "edge_index": template["edge_index"].clone(),
                "edge_weight": template["edge_weight"].clone(),
                "x": template["x"].clone(),
                "y": None if template.get("y") is None else template["y"].clone(),
            }
        )
    return {
        **raw_dataset,
        "dataset": dataset,
        "num_snapshots": len(dataset),
        "num_life_edges": int(sum(int(item["edge_weight"].numel()) for item in dataset)),
    }


@dataclass
class WebDataLoader:
    name: str
    window: int
    batch_size: int | None = None
    use_batch_split: bool = False
    edge_file: str | None = None
    skiprows: int = 0
    sep: str = r"\s+"
    skiptime: int = 0
    lags: int = 0
    fallback_snaps: int = 8

    def _get_uvwt(self, rows: list[list[str]]) -> tuple[list[int], list[int], list[float], list[int]]:
        src, dst, weights, timestamps = [], [], [], []
        for row in rows:
            if len(row) == 3:
                u, v, ts = row
                w = 1.0
            elif len(row) == 4:
                u, v, w, ts = row
            elif len(row) >= 5:
                _, u, v, ts = row[:4]
                w = 1.0
            else:
                continue
            src.append(int(float(u)))
            dst.append(int(float(v)))
            weights.append(float(w))
            timestamps.append(int(float(ts)))
        return src, dst, weights, timestamps

    def _map_nodes(self, src: list[int], dst: list[int]) -> tuple[list[int], list[int], int]:
        uniq = sorted(set(src) | set(dst))
        node_mapping = {old: new for new, old in enumerate(uniq)}
        return [node_mapping[item] for item in src], [node_mapping[item] for item in dst], len(uniq)

    def _batch_snapshots(
        self,
        src: list[int],
        dst: list[int],
        weights: list[float],
        timestamps: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        if not src:
            return [], [], 0
        src, dst, num_nodes = self._map_nodes(src, dst)
        order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        src = [src[i] for i in order]
        dst = [dst[i] for i in order]
        weights = [weights[i] for i in order]
        timestamps = [timestamps[i] for i in order]

        uniq_tss = sorted(set(timestamps))
        if 0 < self.skiptime < len(uniq_tss):
            keep_ts = uniq_tss[self.skiptime]
            keep_idx = [i for i, ts in enumerate(timestamps) if ts >= keep_ts]
            src = [src[i] for i in keep_idx]
            dst = [dst[i] for i in keep_idx]
            weights = [weights[i] for i in keep_idx]

        batch_size = max(1, self.batch_size or 1)
        edge_snapshots: list[torch.Tensor] = []
        edge_weight_snapshots: list[torch.Tensor] = []
        for start in range(0, len(src), batch_size):
            end = min(len(src), start + batch_size)
            edge_snapshots.append(torch.tensor([src[start:end], dst[start:end]], dtype=torch.int64))
            edge_weight_snapshots.append(torch.tensor(weights[start:end], dtype=torch.float32))
        return edge_snapshots, edge_weight_snapshots, num_nodes

    def _masked_snapshots(
        self,
        src: list[int],
        dst: list[int],
        weights: list[float],
        timestamps: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        if not src:
            return [], [], 0
        src, dst, num_nodes = self._map_nodes(src, dst)
        uniq_tss = sorted(set(timestamps))
        if self.skiptime >= len(uniq_tss):
            return [], [], num_nodes
        uniq_tss = uniq_tss[self.skiptime :]
        start_ts = uniq_tss[0]
        window_count = max(1, self.window)
        end_ts = max(start_ts + 1, uniq_tss[-1])
        time_itv = max(1.0, float(end_ts - start_ts) / float(window_count))

        edge_snapshots: list[torch.Tensor] = []
        edge_weight_snapshots: list[torch.Tensor] = []
        for index in range(max(1, window_count - self.lags)):
            lower = start_ts + index * time_itv
            upper = start_ts + (index + self.lags + 1) * time_itv
            mask = [lower <= ts < upper for ts in timestamps]
            snap_src = [src[i] for i, keep in enumerate(mask) if keep]
            snap_dst = [dst[i] for i, keep in enumerate(mask) if keep]
            snap_weights = [weights[i] for i, keep in enumerate(mask) if keep]
            if not snap_src:
                continue
            edge_snapshots.append(torch.tensor([snap_src, snap_dst], dtype=torch.int64))
            edge_weight_snapshots.append(torch.tensor(snap_weights, dtype=torch.float32))
        return edge_snapshots, edge_weight_snapshots, num_nodes

    def _read_rows(self, path: Path) -> list[list[str]]:
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no < self.skiprows:
                    continue
                line = line.strip()
                if not line:
                    continue
                rows.append(_split_line(line, "," if self.sep == "," else r"\s+"))
        return rows

    def _build_mock_dataset(self) -> dict[str, Any]:
        num_nodes = 8
        dataset = []
        for snap in range(self.fallback_snaps):
            src = torch.tensor([(snap + i) % num_nodes for i in range(4)], dtype=torch.int64)
            dst = torch.tensor([(snap + i + 1) % num_nodes for i in range(4)], dtype=torch.int64)
            edge_index = torch.stack([src, dst], dim=0)
            edge_weight = torch.ones(4, dtype=torch.float32)
            in_deg, out_deg = _compute_degrees(edge_index, edge_weight, num_nodes)
            dataset.append(
                {
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                    "x": torch.stack([in_deg, out_deg], dim=1),
                    "y": torch.log(in_deg + 1.0) if snap < self.fallback_snaps - 1 else None,
                }
            )
        num_edges = int(sum(int(item["edge_weight"].numel()) for item in dataset))
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_snapshots": len(dataset),
            "num_life_edges": num_edges,
            "dataset": dataset,
            "source": "mock",
        }

    def get_dataset(self, root: Path | str | None = None) -> dict[str, Any]:
        if root is None:
            return self._build_mock_dataset()
        root = Path(root).expanduser().resolve()
        path = root / self.edge_file if self.edge_file is not None else root / self.name / f"{self.name}.edges"
        if not path.exists():
            return self._build_mock_dataset()

        rows = self._read_rows(path)
        src, dst, weights, timestamps = self._get_uvwt(rows)
        if self.use_batch_split:
            edges, edge_weights, num_nodes = self._batch_snapshots(src, dst, weights, timestamps)
        else:
            edges, edge_weights, num_nodes = self._masked_snapshots(src, dst, weights, timestamps)
        if not edges:
            return self._build_mock_dataset()

        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor | None] = []
        for edge_index, edge_weight in zip(edges, edge_weights):
            in_deg, out_deg = _compute_degrees(edge_index, edge_weight, num_nodes)
            xs.append(torch.stack([in_deg, out_deg], dim=1))
            ys.append(torch.log(in_deg + 1.0))
        ys[-1] = None

        dataset = []
        for index, edge_index in enumerate(edges):
            dataset.append(
                {
                    "edge_index": edge_index,
                    "edge_weight": edge_weights[index],
                    "x": xs[index],
                    "y": ys[index],
                }
            )

        return {
            "num_nodes": int(num_nodes),
            "num_edges": int(len(src)),
            "num_snapshots": len(dataset),
            "num_life_edges": int(sum(int(item["edge_weight"].numel()) for item in dataset)),
            "dataset": dataset,
            "source": str(path),
        }


def build_web_data_loader(dataset_name: str, snaps: int, config: dict[str, Any]) -> WebDataLoader:
    presets: dict[str, dict[str, Any]] = {
        "ia-slashdot-reply-dir": {"window": 200, "skiprows": 2, "sep": r"\s+", "skiptime": 1, "lags": 30},
        "WikiTalk": {
            "window": 200,
            "batch_size": 3000,
            "use_batch_split": True,
            "edge_file": "WikiTalk/edges.csv",
            "skiprows": 0,
            "sep": ",",
            "skiptime": 1,
            "lags": 30,
        },
        "rec-amazon-ratings": {"window": 100, "skiprows": 2, "sep": ",", "skiptime": 1, "lags": 30},
        "rec-amz-Books": {"window": 100, "skiprows": 0, "sep": ",", "skiptime": 0, "lags": 0},
        "soc-bitcoin": {"window": 100, "skiprows": 0, "sep": r"\s+", "skiptime": 1, "lags": 10},
        "soc-flickr-growth": {"window": 100, "skiprows": 1, "sep": r"\s+", "skiptime": 1, "lags": 30},
        "soc-youtube-growth": {"window": 100, "skiprows": 2, "sep": r"\s+", "skiptime": 1, "lags": 30},
    }
    loader_args = dict(presets.get(dataset_name, {}))
    loader_args.setdefault("window", max(1, snaps))
    loader_args["fallback_snaps"] = max(1, snaps)
    if not loader_args.get("use_batch_split", False):
        loader_args["window"] = max(loader_args["window"], snaps + int(loader_args.get("lags", 0)))
    if dict(config.get("data", {})).get("format") == "mock":
        loader_args["edge_file"] = "__mock__"
    return WebDataLoader(name=dataset_name, **loader_args)



def build_dtdg_partitions(
    graph_data: Any,
    num_parts: int,
    algo: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run graph partitioning on the merged lifetime edge index.

    Args:
        graph_data: Raw dataset dict (output of
            :func:`~starry_unigraph.data.build_snapshot_dataset_from_events`).
        num_parts: Number of partitions.
        algo: Partition algorithm name (e.g. ``"metis"``).
        config: Full session config (used for window_size).

    Returns:
        Dict with ``"algo"``, ``"num_parts"``, ``"node_parts"`` (tensor),
        ``"parts"`` (list of per-partition dicts), ``"graph_stats"``.
    """
    num_nodes = int(graph_data["num_nodes"])
    snapshots = list(graph_data.get("dataset", []))
    merged_edge_index = _merge_snapshot_edges(snapshots)
    node_parts = apply_partition(merged_edge_index, num_nodes=num_nodes, num_parts=max(1, num_parts))
    parts = []
    for part_id in range(max(1, num_parts)):
        node_ids = torch.nonzero(node_parts == part_id, as_tuple=False).view(-1)
        parts.append({"partition_id": part_id, "node_ids": node_ids, "num_nodes": int(node_ids.numel())})
    return {
        "algo": algo,
        "num_parts": max(1, num_parts),
        "node_parts": node_parts,
        "parts": parts,
        "graph_stats": {
            "num_nodes": num_nodes,
            "num_edges": int(graph_data["num_edges"]),
            "num_snapshots": int(graph_data["num_snapshots"]),
            "window_size": _as_int(config.get("model", {}).get("window", {}).get("size"), 1),
        },
    }


def _build_snapshot_blocks(
    snapshot_index: int,
    snapshot: dict[str, Any],
    node_parts: torch.Tensor,
    num_parts: int,
    num_nodes: int,
    route_plan: SnapshotRoutePlan,
) -> list[dgl.DGLBlock]:
    edge_index = snapshot["edge_index"].long()
    edge_weight = snapshot["edge_weight"].float()
    node_x = snapshot["x"]
    labels = snapshot.get("y")
    in_deg, out_deg = _compute_degrees(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
    if edge_index.numel() > 0:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        in_norm = torch.sqrt(in_deg.clamp_min_(1e-12))[dst]
        out_norm = torch.sqrt(out_deg.clamp_min_(1e-12))[src]
        gcn_norm = (edge_weight / (in_norm * out_norm)).nan_to_num_(0.0)
    else:
        gcn_norm = torch.zeros((0,), dtype=torch.float32)

    blocks = Route.from_graph(node_parts, edge_index, num_parts=num_parts)
    for part_id, block in enumerate(blocks):
        src_ids = block.srcdata[dgl.NID].long()
        dst_ids = block.dstdata[dgl.NID].long()
        edge_ids = block.edata[dgl.EID].long()
        block.srcdata["x"] = node_x[src_ids] if src_ids.numel() > 0 else torch.zeros((0, node_x.size(-1)), dtype=node_x.dtype)
        block.dstdata["y"] = labels[dst_ids] if labels is not None and dst_ids.numel() > 0 else torch.zeros((dst_ids.numel(),), dtype=torch.float32)
        block.edata["w"] = edge_weight[edge_ids] if edge_ids.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        block.edata["gcn_norm"] = gcn_norm[edge_ids] if edge_ids.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        block.edata["snapshot_index"] = torch.full((edge_ids.numel(),), snapshot_index, dtype=torch.long)
        block.edata["partition_id"] = torch.full((edge_ids.numel(),), part_id, dtype=torch.long)
        block.graph_meta = {
            "snapshot_index": snapshot_index,
            "partition_id": part_id,
            "route_plan": route_plan.describe(),
            "num_nodes": num_nodes,
            "route": None if getattr(block, "route", None) is None else block.route.describe(),
        }
    return blocks


def build_flare_partition_data_list(
    raw_dataset: dict[str, Any],
    partition_result: dict[str, Any],
    route_plan: SnapshotRoutePlan,
    config: dict[str, Any],
) -> list[PartitionData]:
    """Build per-partition PartitionData objects for Flare training.

    For each partition:
    1. Build DGL blocks from per-snapshot edges using ``Route.from_graph``.
    2. Attach node features (``"x"``), labels (``"y"``), edge weights
       (``"w"``), and GCN normalization (``"gcn_norm"``).
    3. Convert blocks to :class:`PartitionData` via ``from_blocks``.
    4. Add chunk assignment (``"c"``) and partition ID (``"partition_id"``)
       node data.

    Args:
        raw_dataset: Raw dataset dict with ``"dataset"`` snapshot list.
        partition_result: Output of :func:`build_dtdg_partitions`.
        route_plan: :class:`SnapshotRoutePlan` for route metadata.
        config: Full session config (used for cluster count).

    Returns:
        List of :class:`PartitionData`, one per partition, ready for
        ``torch.save`` and Flare training.

    Example::

        parts = build_flare_partition_data_list(raw, part_result, route, cfg)
        for i, p in enumerate(parts):
            p.save(f"artifacts/flare/part_{i:03d}.pth")
    """
    dataset = list(raw_dataset["dataset"])
    node_parts = partition_result["node_parts"].long()
    num_parts = int(partition_result["num_parts"])
    num_nodes = int(raw_dataset["num_nodes"])
    chunk_count = max(1, _as_int(config.get("preprocess", {}).get("cluster", {}).get("num_per_partition"), 1))
    lifetime_graph = _build_lifetime_graph(dataset, num_nodes=num_nodes)
    block_lists = [[] for _ in range(num_parts)]
    for snapshot_index, snapshot in enumerate(dataset):
        for part_id, block in enumerate(
            _build_snapshot_blocks(
                snapshot_index=snapshot_index,
                snapshot=snapshot,
                node_parts=node_parts,
                num_parts=num_parts,
                num_nodes=num_nodes,
                route_plan=route_plan,
            )
        ):
            block_lists[part_id].append(block)

    partition_data_list: list[PartitionData] = []
    for part_id in range(num_parts):
        local_nodes = torch.nonzero(node_parts == part_id, as_tuple=False).view(-1)
        partition_data = PartitionData.from_blocks(block_lists[part_id])
        local_edge_index = _induced_local_edges(lifetime_graph, local_nodes=local_nodes)
        chunk_parts = min(chunk_count, max(1, int(local_nodes.numel())))
        chunk_index = apply_partition(local_edge_index, num_nodes=int(local_nodes.numel()), num_parts=chunk_parts)
        chunk_tensors = [chunk_index.clone() for _ in range(len(partition_data))]
        partition_data.add_ndata("c", TensorData.from_tensors(chunk_tensors))
        partition_data.add_ndata(
            "partition_id",
            TensorData.from_tensors(
                [torch.full((local_nodes.numel(),), part_id, dtype=torch.long) for _ in range(len(partition_data))]
            ),
        )
        partition_data_list.append(partition_data)
    return partition_data_list
