"""Chunk preprocessing pipeline: raw events -> chunk runtime artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from starry_unigraph.backends.chunk.data.dist_index import encode_dist_index
from starry_unigraph.backends.chunk.data.graph_store import ChunkGraphStore
from starry_unigraph.backends.chunk.data.partition import PartitionData, TensorData
from starry_unigraph.backends.chunk.data.plans import ChunkPlacement
from starry_unigraph.backends.chunk.prepare.pipeline import PrepareArtifacts as ChunkPrepareArtifacts
from starry_unigraph.backends.chunk.prepare.pipeline import prepare as prepare_chunks
from starry_unigraph.data import build_snapshot_dataset_from_events, load_raw_temporal_events
from starry_unigraph.preprocess.base import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.types import DistributedContext, PreparedArtifacts, SessionContext

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
        if "x" in first:
            part.node_data["x"] = TensorData.from_tensors([first["x"][dst_ids].float()])
        y = first.get("y")
        if y is not None:
            part.node_data["y"] = TensorData.from_tensors([y[dst_ids].float().view(-1, 1)])
    return part


def _unique_sorted(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.long().cpu()
    return values.long().unique(sorted=True).cpu()


def _build_part_local_layout(
    *,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    node_owner: torch.Tensor,
    shared_mask: torch.Tensor,
    part_id: int,
) -> dict[str, torch.Tensor | int]:
    owns_dst = node_owner[edge_dst] == part_id
    hot_edge = shared_mask[edge_src] | shared_mask[edge_dst]
    touched = _unique_sorted(torch.cat([edge_src[owns_dst | hot_edge], edge_dst[owns_dst | hot_edge]]))
    shared = _unique_sorted(shared_mask.nonzero(as_tuple=True)[0])
    owned = _unique_sorted(((node_owner == part_id) & ~shared_mask).nonzero(as_tuple=True)[0])
    if touched.numel() == 0:
        shadow = torch.empty(0, dtype=torch.long)
    else:
        local_mask = torch.zeros(int(node_owner.numel()), dtype=torch.bool)
        local_mask[shared] = True
        local_mask[owned] = True
        shadow = touched[~local_mask[touched]]
    local_node_ids = torch.cat([shared, owned, shadow]).long()
    return {
        "local_node_ids": local_node_ids,
        "shared_count": int(shared.numel()),
        "owned_count": int(owned.numel()),
        "shadow_1hop_count": int(shadow.numel()),
    }


def _build_placement_artifact(
    *,
    artifacts: ChunkPrepareArtifacts,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    num_edges: int,
    num_parts: int,
) -> dict[str, Any]:
    node_owner = artifacts.node_owner.long().cpu()
    node_master = artifacts.node_partition.long().cpu()
    replica_mask = artifacts.replica_mask.bool().cpu()
    node_to_chunk = artifacts.assignment.node_to_chunk.long().cpu()
    num_nodes = int(node_owner.numel())

    canonical_nid_dist = torch.empty(num_nodes, dtype=torch.long)
    local_node_ids_by_part: list[torch.Tensor] = []
    local_nid_dist_by_part: list[torch.Tensor] = []
    local_node_counts: list[dict[str, int]] = []
    for part_id in range(num_parts):
        layout = _build_part_local_layout(
            edge_src=edge_src.cpu(),
            edge_dst=edge_dst.cpu(),
            node_owner=node_owner,
            shared_mask=replica_mask,
            part_id=part_id,
        )
        local_node_ids = layout["local_node_ids"]
        if not isinstance(local_node_ids, torch.Tensor):
            raise TypeError("local_node_ids must be a tensor")
        local_node_ids_by_part.append(local_node_ids)
        shared_count = int(layout["shared_count"])
        owned_count = int(layout["owned_count"])
        local_node_counts.append(
            {
                "shared": shared_count,
                "owned": owned_count,
                "shadow_1hop": int(layout["shadow_1hop_count"]),
                "total": int(local_node_ids.numel()),
            }
        )
        local_ids = torch.arange(local_node_ids.numel(), dtype=torch.long)
        shared_local = local_ids < shared_count
        shadow_local = local_ids >= shared_count + owned_count
        local_nid_dist_by_part.append(
            encode_dist_index(
                local_ids,
                torch.full((int(local_node_ids.numel()),), part_id, dtype=torch.long),
                shared=shared_local,
                cached=shadow_local,
            )
        )
        canonical_mask = (node_owner[local_node_ids] == part_id) & ~replica_mask[local_node_ids]
        canonical_nodes = local_node_ids[canonical_mask]
        canonical_nid_dist[canonical_nodes] = encode_dist_index(
            local_ids[canonical_mask],
            torch.full((int(canonical_nodes.numel()),), part_id, dtype=torch.long),
        )

    shared_nodes = replica_mask.nonzero(as_tuple=True)[0].long().cpu()
    if shared_nodes.numel() > 0:
        canonical_nid_dist[shared_nodes] = encode_dist_index(
            torch.arange(shared_nodes.numel(), dtype=torch.long),
            node_master[shared_nodes],
            shared=True,
        )

    canonical_eid_dist = torch.empty(num_edges, dtype=torch.long)
    canonical_edge_ids_by_part: list[torch.Tensor] = []
    edge_owner = node_owner[edge_dst.cpu().long()]
    all_eids = torch.arange(num_edges, dtype=torch.long)
    for part_id in range(num_parts):
        part_eids = all_eids[edge_owner == part_id]
        canonical_edge_ids_by_part.append(part_eids)
        canonical_eid_dist[part_eids] = encode_dist_index(
            torch.arange(part_eids.numel(), dtype=torch.long),
            torch.full((int(part_eids.numel()),), part_id, dtype=torch.long),
        )

    return {
        "format": "chunk_dist_index_v1",
        "placement_version": 0,
        "dist_index": {
            "local_bits": 48,
            "shared_bit": 48,
            "cached_bit": 49,
            "part_shift": 50,
            "part_bits": 16,
        },
        "assignment": artifacts.assignment,
        "node_to_chunk": node_to_chunk,
        "node_owner": node_owner,
        "node_master": node_master,
        "node_partition": node_master,
        "replica_mask": replica_mask,
        "hot_node_ids": artifacts.hot_node_ids.long().cpu(),
        "graph_family": artifacts.graph_family,
        "chunk_load_by_slice": artifacts.chunk_load_by_slice,
        "canonical_nid_dist": canonical_nid_dist,
        "canonical_eid_dist": canonical_eid_dist,
        "local_node_ids_by_part": local_node_ids_by_part,
        "local_nid_dist_by_part": local_nid_dist_by_part,
        "local_node_counts": local_node_counts,
        "canonical_edge_ids_by_part": canonical_edge_ids_by_part,
        "time_ptr": None if artifacts.time_ptr is None else artifacts.time_ptr.long().cpu(),
    }


def _write_route_lists(root: Path, prefix: str, routes: list[list[Any]] | None) -> list[str]:
    if routes is None:
        return []
    written: list[str] = []
    for part_id, part_routes in enumerate(routes):
        route_path = root / f"{prefix}_{part_id:03d}.pth"
        torch.save(part_routes, route_path)
        written.append(str(route_path.relative_to(root)))
    return written


class ChunkPreprocessor(GraphPreprocessor):
    """Preprocessor for chunk graph mode."""

    graph_mode = "chunk"
    artifact_dirs = ("meta", "partitions", "routes", "sampling", "snapshots", "clusters")

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
        slice_config = dict(session_ctx.config.get("data", {}).get("slice_config") or {})
        slice_config.setdefault("num_windows", snaps)
        raw_dataset = build_snapshot_dataset_from_events(events=raw_events, slice_config=slice_config, config=session_ctx.config)
        if not raw_dataset.get("dataset"):
            raise RuntimeError("Chunk preprocessing requires data.build_snapshot_dataset=true")

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
            graph_family=str(session_ctx.config.get("data", {}).get("graph_mode", self.graph_mode)),
            partition_strategy=strategy,
            hot_topk=int(chunk_cfg.get("hot_topk", 0)),
            hot_ratio=float(chunk_cfg.get("hot_ratio", chunk_cfg.get("shared_ratio", 0.0))),
            num_chunks_per_partition=int(chunk_cfg.get("num_chunks_per_partition", chunk_cfg.get("node_clusters", 32))),
            max_imbalance_ratio=float(chunk_cfg.get("max_imbalance_ratio", 1.2)),
            max_migrations=chunk_cfg.get("max_migrations"),
            build_mem_routes=bool(chunk_cfg.get("build_mem_routes", True)),
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
        (root / "sampling").mkdir(parents=True, exist_ok=True)

        edge_ids = torch.arange(int(raw_events.num_edges), dtype=torch.long)
        placement = _build_placement_artifact(
            artifacts=artifacts,
            edge_src=raw_events.src.long(),
            edge_dst=raw_events.dst.long(),
            num_edges=int(raw_events.num_edges),
            num_parts=num_parts,
        )
        placement_view = ChunkPlacement(
            placement_version=int(placement.get("placement_version", 0)),
            node_to_chunk=placement["node_to_chunk"].long(),
            node_owner=placement["node_owner"].long(),
            node_master=placement["node_master"].long(),
            replica_mask=placement["replica_mask"].bool(),
        )
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
            part_data.node_to_chunk = None
            torch.save(part_data, root / "partitions" / f"part_{part_id:03d}.pth")
            graph_store = ChunkGraphStore.from_partition_data(part_data, placement=placement_view)
            temporal_index = graph_store.temporal_index_view()
            torch.save(
                {
                    "format": "chunk_temporal_index_v1",
                    "indptr": temporal_index.indptr.cpu().contiguous(),
                    "indices": temporal_index.indices.cpu().contiguous(),
                    "edge_ids": temporal_index.edge_ids.cpu().contiguous(),
                    "timestamps": (
                        None
                        if temporal_index.timestamps is None
                        else temporal_index.timestamps.cpu().contiguous()
                    ),
                    "num_nodes": temporal_index.num_nodes,
                    "num_edges": temporal_index.num_edges,
                },
                root / "sampling" / f"temporal_index_part_{part_id:03d}.pth",
            )

        torch.save(placement, root / "placement.pth")
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
                    "format": "per_partition_bundle",
                    "mem_route_files": mem_route_dirs,
                    "spatial_route_files": spatial_route_dirs,
                },
            ),
            ArtifactOutput(
                "sampling/manifest.json",
                {
                    "format": "chunk_temporal_index_v1",
                    "num_parts": num_parts,
                    "files": [
                        f"temporal_index_part_{part_id:03d}.pth"
                        for part_id in range(num_parts)
                    ],
                    "layout": "csc_by_dst_then_time",
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


def run_chunk_preprocess_from_config(config: dict[str, Any]) -> PreparedArtifacts:
    """Run chunk preprocessing from a plain config dictionary."""
    data_cfg = config.get("data", {})
    dist_cfg = config.get("dist", {})
    artifact_root = Path(
        data_cfg.get("artifact_root")
        or config.get("artifact_root")
        or data_cfg.get("prepared_dir")
        or "artifacts/chunk"
    ).expanduser().resolve()
    dataset_root = data_cfg.get("root")
    session_ctx = SessionContext(
        config=config,
        project_root=Path(config.get("project_root", ".")).expanduser().resolve(),
        dataset_path=None if dataset_root is None else Path(dataset_root).expanduser().resolve(),
        artifact_root=artifact_root,
        dist=DistributedContext(
            backend=str(dist_cfg.get("backend", "nccl" if int(dist_cfg.get("world_size", 1)) > 1 else "single")),
            world_size=int(dist_cfg.get("world_size", 1)),
            rank=int(dist_cfg.get("rank", 0)),
            local_rank=int(dist_cfg.get("local_rank", 0)),
            local_world_size=int(dist_cfg.get("local_world_size", dist_cfg.get("world_size", 1))),
        ),
    )
    preprocessor = ChunkPreprocessor()
    artifacts = preprocessor.run(session_ctx)
    session_ctx.prepared_artifacts = artifacts
    return artifacts
