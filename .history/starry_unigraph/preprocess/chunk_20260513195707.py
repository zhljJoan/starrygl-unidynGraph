"""Chunk preprocessing pipeline: raw events -> chunk runtime artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from starry_unigraph.backends.chunk.data.partition import PartitionData, TensorData
from starry_unigraph.backends.chunk.prepare.pipeline import PrepareArtifacts as ChunkPrepareArtifacts
from starry_unigraph.backends.chunk.prepare.pipeline import prepare as prepare_chunks
from starry_unigraph.data import build_snapshot_dataset_from_events, load_raw_temporal_events
from starry_unigraph.preprocess.base import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.types import PreparedArtifacts, SessionContext

ARTIFACT_VERSION = 1


def _cfg_get(config: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _time_ptr_from_snapshots(raw_dataset: dict[str, Any]) -> torch.Tensor:
    counts = [int(item["edge_weight"].numel()) for item in raw_dataset["dataset"]]
    ptr = torch.zeros(len(counts) + 1, dtype=torch.long)
    if counts:
        ptr[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    return ptr


def _split_slices(num_slices: int, split_ratio: dict[str, Any]) -> dict[str, list[int]]:
    train_ratio = float(split_ratio.get("train", 0.7))
    val_ratio = float(split_ratio.get("val", 0.15))
    train_end = min(num_slices, max(0, int(round(num_slices * train_ratio))))
    val_end = min(num_slices, max(train_end, train_end + int(round(num_slices * val_ratio))))
    return {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, num_slices)),
    }


def _build_partition_data_for_part(
    *,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_ts: torch.Tensor,
    edge_ids: torch.Tensor,
    node_owner: torch.Tensor,
    hot_mask: torch.Tensor,
    node_to_chunk: torch.Tensor,
    part_id: int,
    raw_dataset: dict[str, Any] | None = None,
) -> PartitionData:
    owns_dst = node_owner[edge_dst] == part_id
    hot_edge = hot_mask[edge_src] | hot_mask[edge_dst]
    mask = owns_dst | hot_edge
    part = PartitionData.from_edge_events(
        edge_src=edge_src[mask],
        edge_dst=edge_dst[mask],
        edge_timestamps=edge_ts[mask],
        edge_ids=edge_ids[mask],
        node_to_chunk=node_to_chunk,
    )
    if raw_dataset is not None and raw_dataset.get("dataset"):
        dst_ids = part.dst_ids[0].item().long()
        first = raw_dataset["dataset"][0]
        part.node_data["x"] = TensorData.from_tensors([first["x"][dst_ids].float()])
        y = first.get("y")
        if y is not None:
            part.node_data["y"] = TensorData.from_tensors([y[dst_ids].float().view(-1, 1)])
    return part


def _write_route_lists(root: Path, prefix: str, routes: list[list[Any]] | None) -> list[str]:
    if routes is None:
        return []
    written: list[str] = []
    for part_id, part_routes in enumerate(routes):
        route_dir = root / f"{prefix}_{part_id:03d}"
        route_dir.mkdir(parents=True, exist_ok=True)
        for slice_id, route in enumerate(part_routes):
            path = route_dir / f"slice_{slice_id:06d}.pth"
            torch.save(route, path)
        written.append(str(route_dir.relative_to(root)))
    return written


class ChunkPreprocessor(GraphPreprocessor):
    """Preprocessor for chunk graph mode."""

    graph_mode = "chunk"
    artifact_dirs = ("meta", "partitions", "routes", "snapshots", "clusters")

    def prepare_raw(self, session_ctx: SessionContext) -> None:
        dataset_root = (
            session_ctx.dataset_path
            if session_ctx.dataset_path is not None
            else Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        )
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_name = session_ctx.config["data"]["name"]
        snaps = int(_cfg_get(session_ctx.config, "train.snaps", _cfg_get(session_ctx.config, "chunk.time_slices", 1)))
        raw_events = load_raw_temporal_events(root=dataset_root, dataset_name=dataset_name, config=session_ctx.config)
        raw_dataset = build_snapshot_dataset_from_events(events=raw_events, snaps=snaps)

        session_ctx.provider_state["raw_events"] = raw_events
        session_ctx.provider_state["raw_dataset"] = raw_dataset
        session_ctx.provider_state["raw_stats"] = {
            "num_nodes": raw_dataset["num_nodes"],
            "num_edges": raw_dataset["num_edges"],
            "num_snapshots": raw_dataset["num_snapshots"],
        }

    def build_partitions(self, session_ctx: SessionContext) -> None:
        raw_events = session_ctx.provider_state["raw_events"]
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        num_parts = int(session_ctx.config["dist"]["world_size"])
        chunk_cfg = session_ctx.config.get("chunk", {})
        graph_cfg = session_ctx.config.get("graph", {})
        strategy = str(chunk_cfg.get("partition_strategy", graph_cfg.get("partition", "metis")))
        if strategy == "memory_share":
            strategy = "mem_share"

        artifacts = prepare_chunks(
            edge_src=raw_events.src,
            edge_dst=raw_events.dst,
            edge_timestamps=raw_events.ts,
            time_ptr=_time_ptr_from_snapshots(raw_dataset),
            num_nodes=int(raw_events.num_nodes),
            num_partitions=num_parts,
            partition_strategy=strategy,
            hot_topk=int(chunk_cfg.get("hot_topk", 0)),
            hot_ratio=float(chunk_cfg.get("hot_ratio", chunk_cfg.get("shared_ratio", 0.0))),
            num_chunks_per_partition=int(chunk_cfg.get("num_chunks_per_partition", chunk_cfg.get("node_clusters", 32))),
            max_imbalance_ratio=float(chunk_cfg.get("max_imbalance_ratio", 1.2)),
            max_migrations=chunk_cfg.get("max_migrations"),
            build_mem_routes=bool(chunk_cfg.get("build_mem_routes", False)),
            num_candidates=int(chunk_cfg.get("num_candidates", 3)),
        )
        session_ctx.provider_state["chunk_prepare"] = artifacts
        session_ctx.provider_state["partition_manifest"] = {
            "num_parts": num_parts,
            "partition_algo": strategy,
            "num_nodes": raw_dataset["num_nodes"],
            "num_edges": raw_dataset["num_edges"],
            "num_snapshots": raw_dataset["num_snapshots"],
            "num_chunks": artifacts.assignment.total_chunks,
            "num_chunks_per_partition": artifacts.assignment.num_chunks_per_partition,
            "hot_node_count": int(artifacts.hot_node_ids.numel()),
        }

    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        raw_events = session_ctx.provider_state["raw_events"]
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        raw_stats = session_ctx.provider_state["raw_stats"]
        artifacts: ChunkPrepareArtifacts = session_ctx.provider_state["chunk_prepare"]
        num_parts = int(session_ctx.config["dist"]["world_size"])
        root = session_ctx.artifact_root
        root.mkdir(parents=True, exist_ok=True)
        (root / "partitions").mkdir(parents=True, exist_ok=True)

        edge_ids = torch.arange(int(raw_events.num_edges), dtype=torch.long)
        for part_id in range(num_parts):
            part_data = _build_partition_data_for_part(
                edge_src=raw_events.src.long(),
                edge_dst=raw_events.dst.long(),
                edge_ts=raw_events.ts.float(),
                edge_ids=edge_ids,
                node_owner=artifacts.node_owner.long(),
                hot_mask=artifacts.hot_node_mask.bool(),
                node_to_chunk=artifacts.assignment.node_to_chunk.long(),
                part_id=part_id,
                raw_dataset=raw_dataset,
            )
            torch.save(part_data, root / f"part_{part_id:03d}.pth")
            torch.save(part_data, root / "partitions" / f"part_{part_id:03d}.pth")

        torch.save(artifacts.assignment, root / "chunk_assignment.pth")
        torch.save(artifacts.node_owner, root / "node_owner.pt")
        torch.save(artifacts.node_partition, root / "node_partition.pt")
        torch.save(artifacts.hot_node_ids, root / "hot_node_ids.pt")
        torch.save(artifacts.replica_mask, root / "replica_mask.pt")
        torch.save(artifacts.time_ptr, root / "time_ptr.pt")
        artifacts.rebalance_manifest.save(root / "partitions" / "rebalance_manifest.json")
        mem_route_dirs = _write_route_lists(root, "mem_routes", artifacts.mem_routes)
        spatial_route_dirs = _write_route_lists(root, "spatial_routes", artifacts.spatial_routes)

        split_slices = _split_slices(raw_dataset["num_snapshots"], session_ctx.config.get("data", {}).get("split_ratio", {}))
        meta_json = {
            "graph_mode": "chunk",
            "num_nodes": raw_stats["num_nodes"],
            "num_edges": raw_stats["num_edges"],
            "num_slices": raw_dataset["num_snapshots"],
            "splits": split_slices,
            "hot_node_count": int(artifacts.hot_node_ids.numel()),
        }

        provider_meta = {
            "graph_mode": self.graph_mode,
            "artifact_version": ARTIFACT_VERSION,
            "num_parts": num_parts,
            "num_nodes": raw_stats["num_nodes"],
            "num_edges": raw_stats["num_edges"],
            "num_snapshots": raw_stats["num_snapshots"],
            "feature_dim": int(raw_dataset.get("node_feat_dim", session_ctx.config["model"]["hidden_dim"])),
            "edge_feat_dim": int(raw_events.edge_feat.size(-1)) if raw_events.edge_feat.dim() > 1 else 0,
            "label_dim": 1,
            "task_type": session_ctx.config["model"]["task"],
            "partition_strategy": artifacts.partition_strategy,
            "hot_node_count": int(artifacts.hot_node_ids.numel()),
        }

        outputs: list[ArtifactOutput] = [
            ArtifactOutput("meta.json", meta_json),
            ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
            ArtifactOutput(
                "routes/manifest.json",
                {
                    "route_type": str(session_ctx.config.get("graph", {}).get("route", "all2all")),
                    "cache_policy": str(session_ctx.config.get("runtime", {}).get("cache", "gpu_local")),
                    "mem_route_dirs": mem_route_dirs,
                    "spatial_route_dirs": spatial_route_dirs,
                },
            ),
            ArtifactOutput(
                "snapshots/manifest.json",
                {
                    "graph_mode": "chunk",
                    "snapshot_count": raw_dataset["num_snapshots"],
                    "num_parts": num_parts,
                    "splits": split_slices,
                },
            ),
        ]

        for part_id in range(num_parts):
            outputs.append(
                ArtifactOutput(
                    f"clusters/part_{part_id:03d}/cluster_manifest.json",
                    {
                        "partition_id": part_id,
                        "graph_mode": "chunk",
                        "cluster_count": int((artifacts.assignment.chunk_to_owner_partition == part_id).sum().item()),
                    },
                )
            )

        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(provider_meta=provider_meta, outputs=outputs),
        )
