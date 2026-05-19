"""ChunkRuntimeLoader: runtime接入层，yield BatchData.

取代旧的 STGraphBlob / DTDGBatch，统一 yield chunk-specific BatchData。

设计：
  - 从磁盘加载 PartitionData（每个分区的图数据）
  - 加载 MemoryRouteData（记忆更新路由，预计算）
  - 加载 SpatialRouteData（特征 fetch 路由，预计算）
  - 对每个时间切片，通过 TaskAdapter 构建 BatchData
  - 通过 CommPipeline 管理异步通信（submit/await 流水线）
  - CTDG 采样和 MFG 构造通过 MemShareEventEngine 进入 C++ native 路径

特征存储重分布：
  - redistribute_preprocessed 当前显式拒绝半成品重排。
  - rebuild_from_scratch() 从 prepare 阶段重头执行。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData
from starry_unigraph.backends.chunk.data.graph_store import ChunkGraphStore, TemporalEventTable, TemporalTargetTable
from starry_unigraph.backends.chunk.data.partition import PartitionData
from starry_unigraph.backends.chunk.data.plans import (
    EventView,
    ChunkPlacement,
    CommPlanBundle,
    ExecutionUnit,
    FetchPlan,
    GraphBatchEnvelope,
    PropagationPlan,
    StateSyncPlan,
    TemporalIndexView,
)
from starry_unigraph.backends.chunk.data.route import MemoryRouteData, SpatialRouteData
from starry_unigraph.backends.chunk.data.comm import CommPipeline, FetchResult, MemoryResult, validate_training_comm_backend
from starry_unigraph.backends.chunk.data.dist_index import dist_index_part
from starry_unigraph.backends.chunk.runtime.task_adapter import ChunkTaskAdapter, get_task_adapter
from starry_unigraph.backends.chunk.runtime.sampler import (
    NegativeSamplerHook, MFGBuilderHook,
)
from starry_unigraph.backends.chunk.runtime.train_step import run_batch
from starry_unigraph.backends.chunk.runtime.event_engine import MemShareEventEngine
from starry_unigraph.backends.chunk.prepare.chunk_assignment import ChunkAssignment
from starry_unigraph.backends.chunk.runtime.stg_loader import RNNStateManager, STGraphBlob, STGraphLoader
from starry_unigraph.backends.chunk.model.dtdg_models import build_flare_model, extract_graph_labels


class SimpleChunkModel(nn.Module):
    """Small BatchData-native model for smoke tests and local debugging."""

    def __init__(self, num_nodes: int, hidden_dim: int, task_type: str, output_dim: int = 1) -> None:
        super().__init__()
        self.task_type = task_type
        self.emb = nn.Embedding(max(1, int(num_nodes)), int(hidden_dim))
        self.node_head = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, batch: BatchData) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        if self.task_type in {"edge_predict", "link_prediction"}:
            if batch.pos_src is not None and batch.pos_dst is not None:
                ps = self.emb(batch.pos_src.clamp_min(0))
                pd = self.emb(batch.pos_dst.clamp_min(0))
                out["pos_score"] = (ps * pd).sum(dim=1)
            if batch.neg_src is not None and batch.neg_dst is not None:
                ns = self.emb(batch.neg_src.clamp_min(0))
                nd = self.emb(batch.neg_dst.clamp_min(0))
                out["neg_score"] = (ns * nd).sum(dim=1)
            return out

        node_ids = batch.target_nodes if batch.target_nodes is not None else batch.node_ids
        out["node_pred"] = self.node_head(self.emb(node_ids.clamp_min(0)))
        return out


class STGraphEdgeMLP(nn.Module):
    """STGraph-compatible edge-predict encoder that avoids DGL CUDA kernels."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, blob: STGraphBlob | Any, state: Any = None) -> tuple[Tensor, Any]:
        graph = blob.current_graph if isinstance(blob, STGraphBlob) else blob
        x = graph.dstdata.get("x")
        if x is None:
            x = graph.srcdata.get("x")
            if x is not None:
                x = x[: graph.num_dst_nodes()]
        if x is None:
            x = torch.ones(graph.num_dst_nodes(), 1, dtype=torch.float32, device=graph.device)
        return self.proj(x.float()), None


# ---------------------------------------------------------------------------
# Artifact layout convention
# ---------------------------------------------------------------------------
#
# prepared_dir/
#   placement.pth            Chunk placement and packed distributed indices
#   partitions/part_NNN.pth  PartitionData  (one per rank)
#   mem_routes_NNN.pth       list[MemoryRouteData], one entry per time slice
#   spatial_routes_NNN.pth   list[SpatialRouteData], one entry per time slice
#   cpu_layout_{rank:03d}.pth CPUMemoryLayout
#   meta.json                 dataset metadata (num_nodes, num_slices, splits)


def _load_optional(path: Path, cls):
    """Load a torch.save'd object if the file exists, else return None."""
    if path.exists():
        return torch.load(path, weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Redistribution entry points
# ---------------------------------------------------------------------------

def redistribute_preprocessed(
    prepared_dir: Path,
    output_dir: Path,
    num_chunks_per_partition: int = 32,
    rebalance: bool = True,
    **kwargs,
) -> None:
    """Redistribute already-prepared artifacts under a new chunk assignment."""
    raise NotImplementedError(
        "redistribute_preprocessed: full redistribution not yet implemented. "
        "Use rebuild_from_scratch() to re-run from prepare phase."
    )


def rebuild_from_scratch(config: Dict[str, Any]) -> None:
    """Run full preprocessing pipeline from scratch using config."""
    from starry_unigraph.preprocess.chunk import run_chunk_preprocess_from_config

    run_chunk_preprocess_from_config(config)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

@dataclass
class ChunkRuntimeLoader:
    """Runtime loader for chunk pipeline.  Yields BatchData.

    Replaces old ChunkRuntimeLoader that depended on STGraphBlob/DTDGBatch.

    Attributes:
        part_data:      PartitionData for this rank's slice(s).
        mem_routes:     List of MemoryRouteData, one per time slice.
        spatial_routes: List of SpatialRouteData, one per time slice.
        propagation_routes: DTDG model-layer routes by snapshot/layer.
        task_adapter:   Task-specific batch builder and loss/metric logic.
        neg_sampler:    Negative sampling hook.
        mfg_builder:    DGL MFG construction hook for sampled graph objects.
        pipeline:       Async CommPipeline for feature fetch + memory update.
        split_slices:   Dict mapping split name → list of slice indices.
        num_nodes:      Total graph node count.
        rank:           This process's rank.
        world_size:     Total number of processes.
        device:         Computation device.
    """

    part_data:       PartitionData
    mem_routes:      List[MemoryRouteData]
    spatial_routes:  List[SpatialRouteData]
    task_adapter:    ChunkTaskAdapter
    neg_sampler:     NegativeSamplerHook
    mfg_builder:     MFGBuilderHook
    pipeline:        CommPipeline
    split_slices:    Dict[str, List[int]]
    num_nodes:       int
    rank:            int
    world_size:      int
    device:          torch.device
    chunk_manifest:  Dict[str, Any] = field(default_factory=dict)
    chunk_assignment: Optional[ChunkAssignment] = None
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    propagation_routes: List[List[Any]] = field(default_factory=list)
    graph_store:     Optional[ChunkGraphStore] = None
    event_engine:    Optional[MemShareEventEngine] = None
    memory_change_threshold: float = 0.0
    memory_change_metric: str = "cos"
    _last_memory_result: Optional[MemoryResult] = field(default=None, init=False, repr=False)
    _ctdg_fetch_plan_cache: Dict[tuple[int, int, bytes], FetchPlan] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.graph_store is None:
            self.graph_store = ChunkGraphStore.from_partition_data(self.part_data)
        if self.event_engine is None:
            self.event_engine = MemShareEventEngine.from_config(self.graph_store, {})
        if self.world_size > 1:
            validate_training_comm_backend(self.device)

    # ---------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------

    @classmethod
    def from_prepared_artifacts(
        cls,
        prepared_dir: str | Path,
        rank: int,
        world_size: int,
        device: str | torch.device,
        config: Dict[str, Any],
    ) -> ChunkRuntimeLoader:
        """Load all artifacts for this rank and return a ready loader.

        Args:
            prepared_dir: Directory produced by ChunkPreprocessor.
            rank:         This process's distributed rank.
            world_size:   Total ranks.
            device:       Target device.
            config:       Full config dict (must contain 'task.type').
        """
        prepared_dir = Path(prepared_dir)
        device = torch.device(device)

        # PartitionData. Prefer the compact layout but keep the old root-level
        # file as a compatibility fallback.
        part_path = prepared_dir / "partitions" / f"part_{rank:03d}.pth"
        if not part_path.exists():
            part_path = prepared_dir / f"part_{rank:03d}.pth"
        if not part_path.exists():
            raise FileNotFoundError(f"PartitionData not found: {part_path}")
        part_data: PartitionData = torch.load(part_path, weights_only=False)

        # MemoryRouteData. New artifacts store one list per partition; old
        # artifacts store one file per slice inside a directory.
        mem_routes = cls._load_route_list(prepared_dir, "mem_routes", rank, MemoryRouteData)
        propagation_routes = cls._load_propagation_routes(prepared_dir, rank)

        # Meta
        import json
        meta_path = prepared_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        cls._validate_artifact_manifest(prepared_dir, expected_rank=rank)
        manifest_path = prepared_dir / "partitions" / "manifest.json"
        chunk_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        num_nodes   = meta.get("num_nodes", 0)
        split_info  = meta.get("splits", {"train": [], "val": [], "test": []})
        graph_family = str(config.get("data", {}).get("graph_mode", meta.get("graph_family", meta.get("graph_mode", "chunk")))).lower()
        allow_legacy_spatial = bool(config.get("chunk", {}).get("allow_legacy_spatial_routes", False))
        if graph_family == "ctdg" and not allow_legacy_spatial:
            spatial_routes: list[SpatialRouteData] = []
        else:
            spatial_routes = cls._load_route_list(prepared_dir, "spatial_routes", rank, SpatialRouteData)

        # Build split_slices from meta (fallback: all slices are train)
        T = len(part_data)
        if not split_info.get("train"):
            t_train = int(T * 0.7)
            t_val   = int(T * 0.85)
            split_info = {
                "train": list(range(0, t_train)),
                "val":   list(range(t_train, t_val)),
                "test":  list(range(t_val, T)),
            }

        # Task adapter. Accept both chunk-native task names and top-level
        # model.task names used by the unified CLI configs.
        task_cfg = config.get("task", {})
        task_type = task_cfg.get("type") or config.get("model", {}).get("task", "edge_predict")
        task_aliases = {
            "snapshot_node_regression": "node_regression",
            "snapshot_node_classification": "node_classification",
            "temporal_link_prediction": "link_prediction",
        }
        task_type = task_aliases.get(str(task_type), str(task_type))
        task_kwargs = task_cfg.get("kwargs", {})
        adapter = get_task_adapter(task_type, **task_kwargs)

        # Sampler hooks
        sampler_cfg = config.get("sampler", {})
        neg_sampler = NegativeSamplerHook.from_config(sampler_cfg)
        mfg_builder = MFGBuilderHook.default()

        placement = cls._load_placement(prepared_dir, part_data, rank)
        chunk_assignment = cls._load_chunk_assignment(prepared_dir)
        canonical_events = cls._load_canonical_events(prepared_dir)

        # Configure negative sampler with dst pool (not owner pool)
        if hasattr(neg_sampler, "configure_pools") and placement is not None:
            if canonical_events is not None:
                # Extract dst pool from canonical events
                all_dst = canonical_events.dst.unique(sorted=True)
                local_dst_mask = placement.node_owner[all_dst] == int(rank)
                local_dst_pool = all_dst[local_dst_mask]
                global_dst_pool = all_dst
            else:
                # Fallback: use owner pool (compatibility)
                local_dst_pool = torch.nonzero(placement.node_owner == int(rank), as_tuple=False).flatten().long()
                global_dst_pool = torch.arange(int(num_nodes), dtype=torch.long)

            neg_sampler.configure_pools(local_dst_pool=local_dst_pool, global_dst_pool=global_dst_pool)

        temporal_index = cls._load_temporal_index(prepared_dir, rank, placement)
        canonical_targets = cls._load_canonical_targets(prepared_dir)
        graph_store = ChunkGraphStore.from_partition_data(
            part_data,
            placement=placement,
            temporal_index=temporal_index,
            events=canonical_events,
            targets=canonical_targets,
        )
        event_engine_cfg = dict(sampler_cfg.get("memshare", sampler_cfg))
        event_engine_cfg.setdefault("require_prebuilt_temporal_index", True)
        event_engine_cfg.setdefault("local_part", int(rank))
        event_engine = MemShareEventEngine.from_config(graph_store, event_engine_cfg)
        memory_cfg = config.get("memory", {})
        memshare_cfg = sampler_cfg.get("memshare", {})
        change_threshold = float(
            memory_cfg.get(
                "change_threshold",
                memshare_cfg.get("change_threshold", memshare_cfg.get("alpha", 0.0)),
            )
        )
        change_metric = str(memory_cfg.get("change_metric", memshare_cfg.get("change_metric", "cos")))

        # Comm pipeline
        pipeline = CommPipeline(device=device)

        return cls(
            part_data      = part_data,
            mem_routes     = mem_routes,
            spatial_routes = spatial_routes,
            propagation_routes = propagation_routes,
            task_adapter   = adapter,
            neg_sampler    = neg_sampler,
            mfg_builder    = mfg_builder,
            pipeline       = pipeline,
            graph_store    = graph_store,
            event_engine   = event_engine,
            memory_change_threshold = change_threshold,
            memory_change_metric    = change_metric,
            split_slices   = split_info,
            num_nodes      = num_nodes,
            rank           = rank,
            world_size     = world_size,
            device         = device,
            chunk_manifest = chunk_manifest,
            chunk_assignment = chunk_assignment,
            runtime_config = dict(config),
        )

    @staticmethod
    def _validate_artifact_manifest(prepared_dir: Path, expected_rank: int) -> dict[str, Any]:
        manifest_path = prepared_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing chunk artifact manifest: {manifest_path}")
        import json

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise RuntimeError(
                f"Chunk artifact schema mismatch: expected 1, got {manifest.get('schema_version')}"
            )
        for rel_path in manifest.get("required_files", []):
            path = prepared_dir / str(rel_path)
            if not path.exists():
                raise FileNotFoundError(f"Missing required chunk artifact: {path}")
        rank_read_index = prepared_dir / "indices" / f"read_dist_index_{int(expected_rank):03d}.pth"
        if not rank_read_index.exists():
            raise FileNotFoundError(f"Missing rank read dist index: {rank_read_index}")
        return manifest

    @staticmethod
    def _load_route_list(prepared_dir: Path, prefix: str, rank: int, cls) -> list:
        """Load route artifacts in compact or legacy format."""
        bundle = prepared_dir / f"{prefix}_{rank:03d}.pth"
        if bundle.exists():
            routes = torch.load(bundle, weights_only=False)
            return list(routes)
        directory = prepared_dir / f"{prefix}_{rank:03d}"
        if not directory.exists():
            return []
        files = sorted(directory.glob("slice_*.pth"))
        return [torch.load(f, weights_only=False) for f in files]

    @staticmethod
    def _load_propagation_routes(prepared_dir: Path, rank: int) -> list[list[Any]]:
        route_path = prepared_dir / "routes" / f"propagation_routes_{rank:03d}.pth"
        if route_path.exists():
            routes = torch.load(route_path, weights_only=False)
            return [list(layer_routes) for layer_routes in routes]
        legacy_path = prepared_dir / f"propagation_routes_{rank:03d}.pth"
        if legacy_path.exists():
            routes = torch.load(legacy_path, weights_only=False)
            return [list(layer_routes) for layer_routes in routes]
        return []

    @staticmethod
    def _load_placement(prepared_dir: Path, part_data: PartitionData, rank: int = 0) -> ChunkPlacement | None:
        chunk_path = prepared_dir / "chunks" / "chunk_assignment.pth"
        master_path = prepared_dir / "indices" / "master_dist_index.pth"
        read_path = prepared_dir / "indices" / f"read_dist_index_{rank:03d}.pth"
        placement_path = prepared_dir / "placement.pth"
        if chunk_path.exists() and placement_path.exists():
            payload = torch.load(placement_path, weights_only=False)
            chunk_payload = torch.load(chunk_path, weights_only=False)
            if isinstance(payload, dict) and isinstance(chunk_payload, dict):
                if chunk_payload.get("format") != "chunk_assignment_v1":
                    raise RuntimeError(
                        f"Chunk assignment schema mismatch: expected chunk_assignment_v1, got {chunk_payload.get('format')}"
                    )
                node_to_chunk = chunk_payload.get("node_to_chunk")
                node_owner = payload.get("node_owner")
                node_master = payload.get("node_master", payload.get("node_partition"))
                replica_mask = payload.get("replica_mask")
                if node_to_chunk is not None and node_owner is not None and node_master is not None and replica_mask is not None:
                    legacy_node_to_chunk = payload.get("node_to_chunk")
                    if isinstance(legacy_node_to_chunk, Tensor) and not torch.equal(
                        legacy_node_to_chunk.long().cpu(),
                        node_to_chunk.long().cpu(),
                    ):
                        raise RuntimeError("Chunk placement mismatch: placement.pth node_to_chunk differs from chunks/chunk_assignment.pth")
                    if part_data.node_to_chunk is None:
                        part_data.node_to_chunk = node_to_chunk.long()
                    master_dist_index = torch.load(master_path, weights_only=False) if master_path.exists() else payload.get("master_dist_index", payload.get("canonical_nid_dist"))
                    read_dist_index = torch.load(read_path, weights_only=False) if read_path.exists() else None
                    return ChunkPlacement(
                        placement_version=int(payload.get("placement_version", 0)),
                        node_to_chunk=node_to_chunk.long(),
                        node_owner=node_owner.long(),
                        node_master=node_master.long(),
                        replica_mask=replica_mask.bool(),
                        master_dist_index=None if master_dist_index is None else master_dist_index.long(),
                        read_dist_index=None if read_dist_index is None else read_dist_index.long(),
                    )

        if not placement_path.exists():
            return None
        payload = torch.load(placement_path, weights_only=False)
        if not isinstance(payload, dict):
            return None
        node_to_chunk = payload.get("node_to_chunk")
        node_owner = payload.get("node_owner")
        node_master = payload.get("node_master", payload.get("node_partition"))
        replica_mask = payload.get("replica_mask")
        if node_to_chunk is None or node_owner is None or node_master is None or replica_mask is None:
            return None
        if part_data.node_to_chunk is None:
            part_data.node_to_chunk = node_to_chunk.long()
        master_dist_index = payload.get("master_dist_index", payload.get("canonical_nid_dist"))
        read_dist_by_part = payload.get("read_dist_index_by_part")
        read_dist_index = None
        if isinstance(read_dist_by_part, list) and len(read_dist_by_part) > 0:
            read_dist_index = read_dist_by_part[int(rank)]
        if read_dist_index is None:
            read_dist_index = payload.get("read_dist_index")
        return ChunkPlacement(
            placement_version=int(payload.get("placement_version", 0)),
            node_to_chunk=node_to_chunk.long(),
            node_owner=node_owner.long(),
            node_master=node_master.long(),
            replica_mask=replica_mask.bool(),
            master_dist_index=None if master_dist_index is None else master_dist_index.long(),
            read_dist_index=None if read_dist_index is None else read_dist_index.long(),
        )

    @staticmethod
    def _load_temporal_index(
        prepared_dir: Path,
        rank: int,
        placement: ChunkPlacement | None,
    ) -> TemporalIndexView | None:
        index_path = prepared_dir / "sampling" / f"temporal_index_part_{rank:03d}.pth"
        if not index_path.exists():
            return None
        if placement is None:
            raise FileNotFoundError(
                f"{index_path} exists but placement.pth is missing; "
                "cannot attach placement metadata to temporal index"
            )
        payload = torch.load(index_path, weights_only=False)
        if isinstance(payload, TemporalIndexView):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"Expected temporal index dict, got {type(payload).__name__}")
        return TemporalIndexView(
            indptr=payload["indptr"].long().contiguous(),
            indices=payload["indices"].long().contiguous(),
            edge_ids=payload["edge_ids"].long().contiguous(),
            timestamps=(
                None
                if payload.get("timestamps") is None
                else payload["timestamps"].contiguous()
            ),
            placement=placement.view(),
        )

    @staticmethod
    def _load_canonical_events(prepared_dir: Path) -> TemporalEventTable | None:
        path = prepared_dir / "events" / "canonical_events.pth"
        if not path.exists():
            return None
        payload = torch.load(path, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected canonical events dict, got {type(payload).__name__}")
        if payload.get("format") != "chunk_canonical_events_v1":
            raise RuntimeError(
                f"Canonical events schema mismatch: expected chunk_canonical_events_v1, got {payload.get('format')}"
            )
        required = ("src", "dst", "ts", "eid", "event_owner", "event_dst_chunk", "event_split", "time_ptr")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"Canonical events missing required fields: {missing}")
        return TemporalEventTable(
            src=payload["src"].long().contiguous(),
            dst=payload["dst"].long().contiguous(),
            ts=payload["ts"].contiguous(),
            edge_ids=payload["eid"].long().contiguous(),
            snapshot_event_ptr=payload["time_ptr"].long().contiguous(),
            event_owner=None if payload.get("event_owner") is None else payload["event_owner"].long().contiguous(),
            event_dst_chunk=None if payload.get("event_dst_chunk") is None else payload["event_dst_chunk"].long().contiguous(),
            event_split=None if payload.get("event_split") is None else payload["event_split"].to(torch.uint8).contiguous(),
        )

    @staticmethod
    def _load_canonical_targets(prepared_dir: Path) -> TemporalTargetTable | None:
        path = prepared_dir / "targets" / "node_targets.pth"
        if not path.exists():
            return None
        payload = torch.load(path, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected canonical targets dict, got {type(payload).__name__}")
        if payload.get("format") != "chunk_node_targets_v1":
            raise RuntimeError(
                f"Canonical targets schema mismatch: expected chunk_node_targets_v1, got {payload.get('format')}"
            )
        required = ("target_node", "target_ts", "target_label", "target_owner", "target_split", "target_ptr")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"Canonical targets missing required fields: {missing}")
        return TemporalTargetTable(
            target_node=payload["target_node"].long().contiguous(),
            target_ts=payload["target_ts"].contiguous(),
            target_label=payload["target_label"].contiguous(),
            target_owner=payload["target_owner"].long().contiguous(),
            target_split=payload["target_split"].to(torch.uint8).contiguous(),
            target_ptr=payload["target_ptr"].long().contiguous(),
        )

    @staticmethod
    def _load_chunk_assignment(prepared_dir: Path) -> ChunkAssignment | None:
        path = prepared_dir / "chunks" / "chunk_assignment.pth"
        if not path.exists():
            return None
        payload = torch.load(path, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected chunk assignment dict, got {type(payload).__name__}")
        if payload.get("format") != "chunk_assignment_v1":
            raise RuntimeError(
                f"Chunk assignment schema mismatch: expected chunk_assignment_v1, got {payload.get('format')}"
            )
        required = (
            "node_to_chunk",
            "chunk_ptr",
            "chunk_nodes",
            "chunk_to_initial_partition",
            "chunk_to_owner_partition",
            "num_chunks_per_partition",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"Chunk assignment missing required fields: {missing}")
        return ChunkAssignment(
            num_chunks_per_partition=int(payload["num_chunks_per_partition"]),
            node_to_chunk=payload["node_to_chunk"].long().cpu(),
            chunk_ptr=payload["chunk_ptr"].long().cpu(),
            chunk_nodes=payload["chunk_nodes"].long().cpu(),
            chunk_to_initial_partition=payload["chunk_to_initial_partition"].long().cpu(),
            chunk_to_owner_partition=payload["chunk_to_owner_partition"].long().cpu(),
        )

    # ---------------------------------------------------------------------------
    # Core iterators — yield BatchData
    # ---------------------------------------------------------------------------

    def iter_train(self, split: str = "train") -> Iterator[Any]:
        """Iterate training slices, yielding BatchData."""
        if self.uses_stgraph_runtime:
            yield from self._iter_stgraph(split)
            return
        yield from self._iter_split(split)

    def iter_eval(self, split: str = "val") -> Iterator[Any]:
        """Iterate validation slices, yielding BatchData."""
        if self.uses_stgraph_runtime:
            yield from self._iter_stgraph(split)
            return
        yield from self._iter_split(split)

    def iter_predict(self, split: str = "test") -> Iterator[Any]:
        """Iterate test slices, yielding BatchData."""
        if self.uses_stgraph_runtime:
            yield from self._iter_stgraph(split)
            return
        yield from self._iter_split(split)

    def iter_train_envelopes(self, split: str = "train") -> Iterator[GraphBatchEnvelope]:
        yield from self._iter_envelopes(split)

    def iter_eval_envelopes(self, split: str = "val") -> Iterator[GraphBatchEnvelope]:
        yield from self._iter_envelopes(split)

    def iter_predict_envelopes(self, split: str = "test") -> Iterator[GraphBatchEnvelope]:
        yield from self._iter_envelopes(split)

    def iter_train_units(self, split: str = "train") -> Iterator[ExecutionUnit]:
        yield from self._iter_units(split)

    def iter_eval_units(self, split: str = "val") -> Iterator[ExecutionUnit]:
        yield from self._iter_units(split)

    def iter_predict_units(self, split: str = "test") -> Iterator[ExecutionUnit]:
        yield from self._iter_units(split)

    @property
    def uses_native_ctdg_sampling(self) -> bool:
        runtime_mode = str(self.runtime_config.get("chunk", {}).get("runtime", "")).lower()
        if runtime_mode in {"stgraph", "flare", "flare_stgraph"}:
            return False
        return self.task_adapter.task_type in {"edge_predict", "link_prediction"}

    @property
    def uses_stgraph_runtime(self) -> bool:
        model_name = str(self.runtime_config.get("model", {}).get("name", "")).lower()
        model_family = str(self.runtime_config.get("model", {}).get("family", "")).lower()
        runtime_mode = str(self.runtime_config.get("chunk", {}).get("runtime", "")).lower()
        return runtime_mode in {"stgraph", "flare", "flare_stgraph"} or model_name in {
            "evolvegcn",
            "tgcn",
            "mpnn_lstm",
            "gcn",
        } or model_family in {"evolvegcn", "tgcn", "mpnn_lstm", "gcn"}

    def _iter_stgraph(self, split: str) -> Iterator[STGraphBlob | Any]:
        indices = self.split_slices.get(split, [])
        if not indices:
            return
        cfg = self.runtime_config
        dtdg_cfg = cfg.get("dtdg", {})
        chunk_cfg = cfg.get("chunk", {})
        num_full = int(dtdg_cfg.get("num_full_snaps", chunk_cfg.get("num_full_snaps", 1)))
        decay_cfg = dtdg_cfg.get("chunk_decay", chunk_cfg.get("chunk_decay"))
        if isinstance(decay_cfg, str):
            total = 0 if self.chunk_assignment is None else int(self.chunk_assignment.total_chunks)
            if decay_cfg in {"none", "off", "false"}:
                chunk_decay = None
            elif decay_cfg == "half":
                chunk_decay = list(range(max(0, total // 2)))
            else:
                chunk_decay = list(range(total))
        elif decay_cfg is None:
            total = 0 if self.chunk_assignment is None else int(self.chunk_assignment.total_chunks)
            chunk_decay = list(range(total)) if total > 0 else None
        else:
            chunk_decay = [int(x) for x in decay_cfg]
        loader = STGraphLoader.from_partition_data(
            partition_data=self.part_data,
            device=self.device,
            chunk_assignment=self.chunk_assignment,
            num_full_snaps=num_full,
            chunk_decay=chunk_decay,
            rnn_state_mode=str(dtdg_cfg.get("rnn_state_mode", chunk_cfg.get("rnn_state_mode", "pad"))),
            disable_routes=bool(dtdg_cfg.get("disable_routes", chunk_cfg.get("disable_stgraph_routes", True))),
        )
        for blob_idx, blob in enumerate(loader()):
            if blob_idx not in indices:
                continue
            self._patch_stgraph_targets(blob, blob_idx)
            yield blob

    def _patch_stgraph_targets(self, blob: STGraphBlob | Any, snapshot_idx: int) -> None:
        graph = blob.current_graph if isinstance(blob, STGraphBlob) else blob
        targets = None if self.graph_store is None else self.graph_store.temporal_targets()
        if targets is None:
            return
        target_indices = self.graph_store.target_indices_for_snapshot(snapshot_idx, owner_part=int(self.rank))
        if target_indices.numel() == 0:
            return
        if "y" in graph.dstdata:
            return
        dst_ids = graph.dstdata.get("_ID")
        if dst_ids is None:
            try:
                import dgl

                dst_ids = graph.dstdata.get(dgl.NID)
            except Exception:
                dst_ids = None
        if dst_ids is None:
            return
        target_nodes = targets.target_node[target_indices].long().to(dst_ids.device)
        labels = targets.target_label[target_indices].to(dst_ids.device)
        if labels.dim() == 1:
            labels = labels.view(-1, 1)
        order = torch.argsort(target_nodes, stable=True)
        sorted_nodes = target_nodes[order]
        sorted_labels = labels[order]
        row = torch.searchsorted(sorted_nodes, dst_ids.long())
        safe_row = row.clamp_max(max(0, int(sorted_nodes.numel()) - 1))
        keep = (row < int(sorted_nodes.numel())) & (sorted_nodes[safe_row] == dst_ids.long())
        y = sorted_labels.new_zeros((int(dst_ids.numel()), *sorted_labels.shape[1:]))
        if keep.any():
            y[keep] = sorted_labels[safe_row[keep]]
        graph.dstdata["y"] = y

    def _iter_envelopes(self, split: str) -> Iterator[GraphBatchEnvelope]:
        for t, batch in self._iter_split_with_index(split):
            yield self._make_envelope(batch=batch, block_id=t)

    def _iter_units(self, split: str) -> Iterator[ExecutionUnit]:
        indices = self.split_slices.get(split, [])
        if not indices:
            return
        if self.event_engine is not None and self.uses_native_ctdg_sampling:
            yield from self.event_engine.iter_units(indices, plans_fn=self._make_ctdg_base_comm_plan)
            return
        yield from self._iter_envelopes(split)

    def _iter_split(self, split: str) -> Iterator[BatchData]:
        """Core iteration loop with async communication pipeline.

        Pattern (one slice ahead):
          t=0: submit comm for slice 0
          t=1: compute on slice 0 result, submit comm for slice 1
          ...

        CTDG training can use ``iter_train_units`` to keep sampling and native
        MFG construction in ``MemShareEventEngine``.  ``iter_train`` yields
        BatchData with CSC-backed DGL blocks materialized from PartitionData.
        """
        indices = self.split_slices.get(split, [])
        if not indices:
            return

        for _, batch in self._iter_split_with_index(split):
            yield batch

        self.pipeline.drain_all_sync()

    def _iter_split_with_index(self, split: str) -> Iterator[tuple[int, BatchData]]:
        indices = self.split_slices.get(split, [])
        if not indices:
            return

        for i, t in enumerate(indices):
            # Select event positions (latest per unique node in this slice)
            mem_route = self.mem_routes[t] if t < len(self.mem_routes) else None
            event_pos = (
                mem_route.latest_pos()
                if mem_route is not None and mem_route.num_unique > 0
                else torch.zeros(0, dtype=torch.long)
            )

            # Build BatchData via task adapter
            batch = self.task_adapter.build_batch(
                part         = self.part_data,
                snapshot_idx = min(t, len(self.part_data) - 1),
                event_pos    = event_pos,
                split        = split,
                neg_sampler  = self.neg_sampler,
                num_nodes    = self.num_nodes,
            )
            self._patch_node_targets_from_store(batch, int(t))

            if batch.mfgs is None:
                raise RuntimeError(
                    "BatchData.mfgs is required. Task adapters must materialize "
                    "CSC-backed DGL blocks or CTDG callers must use iter_train_units "
                    "with MemShareEventEngine native sampling."
                )

            # Async comm: submit memory update for next batch, await for current
            # (pipeline is a no-op in single-rank or when routes are empty)
            if self.world_size > 1 and mem_route is not None and mem_route.num_unique > 0:
                # In a real training loop: await prev, then submit current
                # Here we just run sync (awaiting after submit immediately)
                pass  # CommPipeline.submit_memory / await_memory called by trainer

            yield t, batch

        self.pipeline.drain_all_sync()

    def _patch_node_targets_from_store(self, batch: BatchData, time_slice_id: int) -> None:
        if self.task_adapter.task_type in {"edge_predict", "link_prediction", "edge_regress"}:
            return
        if self.graph_store is None or self.graph_store.temporal_targets() is None:
            return
        target_indices = self.graph_store.target_indices_for_snapshot(time_slice_id, owner_part=int(self.rank))
        if target_indices.numel() == 0:
            batch.target_nodes = torch.empty(0, dtype=torch.long)
            batch.labels = torch.empty(0, dtype=torch.long if self.task_adapter.task_type == "node_classify" else torch.float32)
            return
        targets = self.graph_store.temporal_targets()
        if targets is None:
            return
        batch.target_nodes = targets.target_node[target_indices].long().contiguous()
        batch.labels = targets.target_label[target_indices].contiguous()

    def _make_envelope(self, batch: BatchData, block_id: int) -> GraphBatchEnvelope:
        return GraphBatchEnvelope(
            mode="ctdg" if self.task_adapter.task_type in {"edge_predict", "link_prediction"} else "dtdg",
            block_id=int(block_id),
            placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
            payload=batch,
            comm_plan=self._make_comm_plan(block_id),
            profile_hint={
                "num_nodes": float(int(batch.node_ids.numel())),
                "has_mfgs": float(batch.mfgs is not None),
            },
        )

    def _make_comm_plan(self, block_id: int) -> CommPlanBundle:
        fetch = None
        if block_id < len(self.spatial_routes):
            route = self.spatial_routes[block_id]
            feature_node_ids = route.recv_node_ids.contiguous()
            counts = route.recv_ptr[1:] - route.recv_ptr[:-1]
            owners = torch.repeat_interleave(
                torch.arange(counts.numel(), dtype=torch.long, device=counts.device),
                counts,
            ).contiguous()
            fetch = FetchPlan(
                block_id=int(block_id),
                placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
                feature_node_ids=feature_node_ids,
                feature_owners=owners,
                cache_policy="route",
            )

        state_sync = self._make_state_sync_plan(block_id)
        propagation = self._make_propagation_plan(block_id)

        return CommPlanBundle(fetch=fetch, propagation=propagation, state_sync=state_sync)

    def _make_ctdg_base_comm_plan(self, time_slice_id: int) -> CommPlanBundle:
        """Base CTDG plan before sampling.

        CTDG feature/memory fetch depends on the sampled remote read set, so it
        is built later from ``CTDGSampleResult.remote_read_index``.  The only
        stable pre-sampling route is the memory writeback/state-sync relation
        for the time slice.
        """
        return CommPlanBundle(fetch=None, propagation=None, state_sync=self._make_state_sync_plan(time_slice_id))

    def _make_propagation_plan(self, block_id: int) -> PropagationPlan | None:
        if block_id < len(self.propagation_routes):
            layer_routes = self.propagation_routes[block_id]
            return PropagationPlan(
                block_id=int(block_id),
                placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
                layer_routes=list(layer_routes),
                autograd_enabled=True,
            )
        return None

    def _make_state_sync_plan(self, block_id: int) -> StateSyncPlan | None:
        if block_id < len(self.mem_routes):
            route = self.mem_routes[block_id]
            counts = route.send_ptr[1:] - route.send_ptr[:-1]
            owners = torch.repeat_interleave(
                torch.arange(counts.numel(), dtype=torch.long, device=counts.device),
                counts,
            ).contiguous()
            return StateSyncPlan(
                block_id=int(block_id),
                placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
                update_node_ids=route.unique_nodes.contiguous(),
                update_owners=owners,
                update_index=None if route.unique_index is None else route.unique_index.contiguous(),
                replica_node_ids=None if route.replica_idx is None else route.unique_nodes[route.replica_idx].contiguous(),
                replica_owners=None,
                replica_index=None if route.replica_idx is None or route.unique_index is None else route.unique_index[route.replica_idx].contiguous(),
                sync_policy="owner_write",
                change_threshold=self.memory_change_threshold,
                change_metric=self.memory_change_metric,
            )
        return None

    def _make_dynamic_fetch_plan(
        self,
        block_id: int,
        remote_read_index: Optional[Tensor],
        local_read_index: Optional[Tensor] = None,
        remote_node_ids: Optional[Tensor] = None,
        local_node_ids: Optional[Tensor] = None,
    ) -> FetchPlan | None:
        """Build a CTDG fetch plan from sampled packed DistIndex values."""
        has_remote = remote_read_index is not None and remote_read_index.numel() > 0
        has_local = local_read_index is not None and local_read_index.numel() > 0
        if not has_remote and not has_local:
            return None
        if remote_read_index is None:
            remote_read_index = torch.empty(0, dtype=torch.long)
        remote_read_index = remote_read_index.long().contiguous()
        owners = dist_index_part(remote_read_index).long().contiguous()
        if remote_read_index.numel() > 1:
            order = torch.argsort(owners, stable=True)
            remote_read_index = remote_read_index[order].contiguous()
            owners = owners[order].contiguous()
            if remote_node_ids is not None:
                remote_node_ids = remote_node_ids.long().contiguous()[order].contiguous()
        if local_read_index is not None:
            local_read_index = local_read_index.long().contiguous()
        if local_node_ids is not None:
            local_node_ids = local_node_ids.long().contiguous()
        plan = FetchPlan(
            block_id=int(block_id),
            placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
            feature_node_ids=torch.empty(0, dtype=torch.long, device=remote_read_index.device),
            feature_owners=owners,
            remote_read_index=remote_read_index,
            local_read_index=local_read_index,
            remote_node_ids=remote_node_ids,
            local_node_ids=local_node_ids,
            memory_read_index=remote_read_index,
            cache_policy="sampled_packed_dist_index",
        )
        return plan

    @staticmethod
    def _runtime_tensor(runtime: Any, *names: str) -> Optional[Tensor]:
        for name in names:
            value = getattr(runtime, name, None)
            if isinstance(value, Tensor):
                return value
        return None

    @staticmethod
    def _runtime_feature_tensor(runtime: Any) -> Optional[Tensor]:
        value = ChunkRuntimeLoader._runtime_tensor(runtime, "node_features", "features")
        if value is not None:
            return value
        feature_store = getattr(runtime, "feature_store", None)
        if feature_store is not None:
            value = getattr(feature_store, "node_features", None)
            if isinstance(value, Tensor):
                return value
        return None

    @staticmethod
    def _runtime_memory_tensor(runtime: Any) -> Optional[Tensor]:
        value = ChunkRuntimeLoader._runtime_tensor(runtime, "node_memory", "memory")
        if value is not None:
            return value
        memory_store = getattr(runtime, "memory_store", None)
        if memory_store is not None:
            value = getattr(memory_store, "node_memory", None)
            if isinstance(value, Tensor):
                return value
            value = getattr(memory_store, "memory", None)
            if isinstance(value, Tensor):
                return value
        return None

    @staticmethod
    def _flatten_mfgs(mfgs: Any) -> list[Any]:
        if mfgs is None:
            return []
        if isinstance(mfgs, (list, tuple)):
            out: list[Any] = []
            for item in mfgs:
                out.extend(ChunkRuntimeLoader._flatten_mfgs(item))
            return out
        return [mfgs]

    @staticmethod
    def _patch_mfg_srcdata(mfgs: Any, node_ids: Tensor, values: Tensor, key: str) -> int:
        patched = 0
        if node_ids.numel() == 0 or values.numel() == 0:
            return patched
        original_values = values
        node_ids = node_ids.to(device=values.device).long().contiguous()
        if node_ids.numel() > 1:
            order = torch.argsort(node_ids, stable=True)
            node_ids = node_ids[order].contiguous()
            values = values[order].contiguous()
        for block in ChunkRuntimeLoader._flatten_mfgs(mfgs):
            if not hasattr(block, "srcdata"):
                continue
            src_ids = None
            try:
                import dgl

                if dgl.NID in block.srcdata:
                    src_ids = block.srcdata[dgl.NID]
            except Exception:
                src_ids = None
            if src_ids is None:
                if "ID" in block.srcdata:
                    src_ids = block.srcdata["ID"]
                elif "__ID" in block.srcdata:
                    idx = block.srcdata["__ID"].long()
                    block.srcdata[key] = original_values[idx.to(original_values.device)]
                    patched += 1
                    continue
            if src_ids is None:
                continue
            src_query = src_ids.to(device=values.device).long().contiguous()
            row_index_t = torch.searchsorted(node_ids, src_query)
            valid_rows = row_index_t.clamp_max(max(0, int(node_ids.numel()) - 1))
            keep = (row_index_t < int(node_ids.numel())) & (node_ids[valid_rows] == src_query)
            out = values.new_zeros((int(src_query.numel()), *values.shape[1:]))
            if keep.any():
                out[keep] = values[row_index_t[keep]]
            block.srcdata[key] = out.to(src_ids.device)
            patched += 1
        return patched

    def _execute_and_patch_fetch(self, runtime: Any, batch: BatchData) -> Optional[FetchResult]:
        manifest = batch.remote_manifest or {}
        plan = manifest.get("dynamic_fetch_plan")
        if plan is None:
            return None

        feature_rows = self._runtime_feature_tensor(runtime)
        memory_rows = self._runtime_memory_tensor(runtime)
        if feature_rows is None and memory_rows is None:
            return None

        handle = self.pipeline.submit_fetch(
            plan,
            feature_rows=feature_rows,
            memory_rows=memory_rows,
        )
        fetch_result = asyncio.run(self.pipeline.await_handle(handle))
        if not isinstance(fetch_result, FetchResult):
            return None

        local_nodes = getattr(plan, "local_node_ids", None)
        if local_nodes is None:
            local_nodes = manifest.get("local_node_ids")
        remote_nodes = getattr(plan, "remote_node_ids", None)
        if remote_nodes is None:
            remote_nodes = manifest.get("remote_node_ids")
        feature_nodes: list[Tensor] = []
        feature_values: list[Tensor] = []
        if fetch_result.local_features is not None and isinstance(local_nodes, Tensor):
            feature_nodes.append(local_nodes.to(fetch_result.local_features.device))
            feature_values.append(fetch_result.local_features)
        if fetch_result.remote_features is not None and isinstance(remote_nodes, Tensor):
            feature_nodes.append(remote_nodes.to(fetch_result.remote_features.device))
            feature_values.append(fetch_result.remote_features)
        if feature_values:
            nodes = torch.cat(feature_nodes, dim=0).long()
            feats = torch.cat(feature_values, dim=0)
            manifest["fetched_node_ids"] = nodes
            manifest["fetched_features"] = feats
            self._patch_mfg_srcdata(batch.mfgs, nodes, feats, "h")

        memory_nodes: list[Tensor] = []
        memory_values: list[Tensor] = []
        if fetch_result.local_memory is not None and isinstance(local_nodes, Tensor):
            memory_nodes.append(local_nodes.to(fetch_result.local_memory.device))
            memory_values.append(fetch_result.local_memory)
        if fetch_result.remote_memory is not None and isinstance(remote_nodes, Tensor):
            memory_nodes.append(remote_nodes.to(fetch_result.remote_memory.device))
            memory_values.append(fetch_result.remote_memory)
        if memory_values:
            nodes = torch.cat(memory_nodes, dim=0).long()
            mem = torch.cat(memory_values, dim=0)
            manifest["fetched_memory_node_ids"] = nodes
            manifest["fetched_memory"] = mem
            self._patch_mfg_srcdata(batch.mfgs, nodes, mem, "mem")

        batch.remote_manifest = manifest
        return fetch_result

    def submit_memory_update(
        self,
        block_id: int,
        memory: Tensor,
        ts: Tensor,
        baseline_memory: Optional[Tensor] = None,
    ) -> Optional[MemoryResult]:
        """Submit one routed memory update and synchronously collect results.

        ``memory`` and ``ts`` must be aligned with
        ``self.mem_routes[block_id].unique_nodes``.  The filtering threshold is
        taken from the chunk config and the underlying all-to-all uses the
        process group's NCCL backend in distributed CUDA training.
        """
        if block_id >= len(self.mem_routes):
            return None
        route = self.mem_routes[block_id]
        if route.num_unique == 0:
            return None
        if memory.size(0) != route.num_unique or ts.size(0) != route.num_unique:
            raise ValueError(
                "memory update tensors must be aligned with route.unique_nodes: "
                f"route={route.num_unique}, memory={memory.size(0)}, ts={ts.size(0)}"
            )
        if baseline_memory is not None and baseline_memory.size(0) != route.num_unique:
            raise ValueError(
                "baseline_memory must be aligned with route.unique_nodes: "
                f"route={route.num_unique}, baseline={baseline_memory.size(0)}"
            )
        if self.world_size <= 1:
            return None

        validate_training_comm_backend(self.device)
        handle = self.pipeline.submit_memory(
            route,
            memory,
            ts,
            baseline_memory=baseline_memory,
            change_threshold=self.memory_change_threshold,
            change_metric=self.memory_change_metric,
        )
        self._last_memory_result = asyncio.run(self.pipeline.await_handle(handle))
        return self._last_memory_result

    # ---------------------------------------------------------------------------
    # Async iteration (for use with asyncio training loop)
    # ---------------------------------------------------------------------------

    async def async_iter_split(self, split: str):
        """Async generator with overlapped communication.

        Usage::
            async for batch in loader.async_iter_split("train"):
                result = model(batch)
        """
        raise NotImplementedError(
            "async_iter_split is disabled until the caller supplies an explicit "
            "compute->submit_state_sync contract. Use iter_*_units plus "
            "run_train_unit_step, or iter_* with synchronous step APIs."
        )
        indices = self.split_slices.get(split, [])
        if not indices:
            return

        prev_mem_result = None

        for t in indices:
            mem_route = self.mem_routes[t] if t < len(self.mem_routes) else None
            event_pos = (
                mem_route.latest_pos()
                if mem_route is not None and mem_route.num_unique > 0
                else torch.zeros(0, dtype=torch.long)
            )

            batch = self.task_adapter.build_batch(
                part         = self.part_data,
                snapshot_idx = min(t, len(self.part_data) - 1),
                event_pos    = event_pos,
                split        = split,
                neg_sampler  = self.neg_sampler,
                num_nodes    = self.num_nodes,
            )

            # Await previous memory result (overlapped with batch build above)
            prev_mem_result = await self.pipeline.await_memory()
            # (caller updates memory store from prev_mem_result)

            yield batch, prev_mem_result

        self.pipeline.drain_all_sync()

    # ---------------------------------------------------------------------------
    # Observability
    # ---------------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        T = len(self.part_data)
        return {
            "rank":       self.rank,
            "world_size": self.world_size,
            "device":     str(self.device),
            "num_slices": T,
            "task_type":  self.task_adapter.task_type,
            "splits": {k: len(v) for k, v in self.split_slices.items()},
            "mem_routes":     len(self.mem_routes),
            "spatial_routes": len(self.spatial_routes),
        }

    def describe_window_state(self) -> Dict[str, Any]:
        return {"num_slices": len(self.part_data), "splits": self.split_slices}

    def describe_route_cache(self) -> Dict[str, Any]:
        return {"mem_routes": len(self.mem_routes), "spatial_routes": len(self.spatial_routes)}

    def dump_state(self) -> Dict[str, Any]:
        return self.describe()

    def build_default_model(self, config: Dict[str, Any]) -> nn.Module:
        hidden_dim = int(config.get("model", {}).get("hidden_dim", 64))
        if self.uses_stgraph_runtime:
            packed_x = self.part_data.node_data.get("x")
            if packed_x is not None and packed_x.data.dim() > 1 and packed_x.data.numel() > 0:
                input_dim = int(packed_x.data.size(-1))
            else:
                first = self.part_data[0].to_block(0)
                x = first.srcdata.get("x")
                input_dim = int(x.size(-1)) if isinstance(x, Tensor) and x.dim() > 1 else 1
            output_dim = hidden_dim if self.task_adapter.task_type in {"edge_predict", "link_prediction"} else 1
            if self.task_adapter.task_type in {"edge_predict", "link_prediction"} and bool(
                config.get("chunk", {}).get("stgraph_edge_mlp", True)
            ):
                return STGraphEdgeMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
            targets = None if self.graph_store is None else self.graph_store.temporal_targets()
            if self.task_adapter.task_type not in {"edge_predict", "link_prediction"} and targets is not None and targets.target_label.numel() > 0:
                label = targets.target_label
                output_dim = int(label.size(-1)) if label.dim() > 1 else 1
            return build_flare_model(
                str(config.get("model", {}).get("name", "mpnn_lstm")),
                input_size=input_dim,
                hidden_size=hidden_dim,
                output_size=output_dim,
            ).to(self.device)
        return SimpleChunkModel(
            num_nodes=self.num_nodes,
            hidden_dim=hidden_dim,
            task_type=self.task_adapter.task_type,
        ).to(self.device)

    def run_train_step(self, runtime: Any, batch: BatchData) -> dict[str, Any]:
        if isinstance(batch, STGraphBlob) or hasattr(batch, "current_graph"):
            return self._run_stgraph_step(runtime, batch, train=True)
        fetch_result = self._execute_and_patch_fetch(runtime, batch)
        result = run_batch(
            model=runtime.model,
            batch=batch,
            task_adapter=self.task_adapter,
            optimizer=runtime.optimizer,
            train=True,
        )
        if fetch_result is not None:
            remote_count = 0 if fetch_result.remote_read_index is None else int(fetch_result.remote_read_index.numel())
            local_count = 0 if fetch_result.local_read_index is None else int(fetch_result.local_read_index.numel())
            result.setdefault("meta", {})["fetch_remote_rows"] = remote_count
            result["meta"]["fetch_local_rows"] = local_count
        memory_update = result.get("output", {}).get("memory_update")
        if isinstance(memory_update, dict):
            mem = memory_update.get("memory")
            ts = memory_update.get("ts")
            baseline = memory_update.get("baseline_memory")
            if mem is not None and ts is not None and batch.chunk_id is not None:
                mem_result = self.submit_memory_update(
                    int(batch.chunk_id),
                    mem,
                    ts,
                    baseline_memory=baseline,
                )
                result.setdefault("meta", {})["memory_sync_recv"] = (
                    0 if mem_result is None else int(mem_result.recv_node_ids.numel())
                )
        return result

    def _batch_from_sampled_unit(self, unit: ExecutionUnit, split: str = "train") -> BatchData:
        if self.event_engine is None:
            raise RuntimeError("Chunk CTDG sampled-unit execution requires MemShareEventEngine")
        view = unit.payload
        if not isinstance(view, EventView):
            raise TypeError(f"Expected EventView payload, got {type(view).__name__}")

        events = self.graph_store.temporal_events()
        event_indices = view.event_indices
        if event_indices is None:
            start, end = int(view.event_start), int(view.event_end)
            event_indices = torch.arange(start, end, dtype=torch.long, device=events.src.device)
        else:
            event_indices = event_indices.long().to(device=events.src.device).contiguous()
        pos_src = events.src[event_indices].long().contiguous()
        pos_dst = events.dst[event_indices].long().contiguous()
        timestamps = events.ts[event_indices].contiguous()
        neg_src, neg_dst = self.neg_sampler.sample(pos_src, pos_dst, self.num_nodes, 1, split=split)
        neg_ratio = max(1, int(neg_dst.numel() // max(1, pos_dst.numel())))
        neg_ts = timestamps.repeat_interleave(neg_ratio)[: neg_dst.numel()].contiguous()
        required_nodes = torch.cat([pos_src, pos_dst, neg_src, neg_dst], dim=0).unique(sorted=True).contiguous()

        sample_sig = inspect.signature(self.event_engine.sample)
        if "extra_root_nodes" in sample_sig.parameters:
            sampled = self.event_engine.sample(
                unit,
                extra_root_nodes=neg_dst,
                extra_root_ts=neg_ts,
                required_nodes=required_nodes,
            )
        else:
            sampled = self.event_engine.sample(unit)
        node_ids = sampled.input_nodes.long().contiguous()
        if node_ids.numel() == 0:
            node_ids = required_nodes
        elif "extra_root_nodes" not in sample_sig.parameters:
            node_ids = torch.cat([node_ids, required_nodes], dim=0).unique(sorted=True).contiguous()

        fetch_plan = self._make_dynamic_fetch_plan(
            int(unit.block_id),
            sampled.remote_read_index,
            sampled.local_read_index,
            remote_node_ids=sampled.remote_node_ids,
            local_node_ids=sampled.local_node_ids,
        )
        return BatchData(
            mfgs=sampled.mfgs,
            node_ids=node_ids,
            id_map_nodes=sampled.id_map_nodes if sampled.id_map_nodes is not None else node_ids,
            pos_src=pos_src,
            pos_dst=pos_dst,
            neg_src=neg_src,
            neg_dst=neg_dst,
            timestamps=timestamps,
            chunk_id=int(view.time_slice_id),
            remote_manifest={
                "batch_id": int(unit.block_id),
                "time_slice_id": int(view.time_slice_id),
                "batch_offset": int(view.batch_offset),
                "event_indices": event_indices,
                "sampled_input_nodes": sampled.input_nodes,
                "id_map_nodes": sampled.id_map_nodes if sampled.id_map_nodes is not None else node_ids,
                "sampled_output_nodes": sampled.output_nodes,
                "sampled_edge_ids": sampled.edge_ids,
                "remote_node_ids": sampled.remote_node_ids,
                "local_node_ids": sampled.local_node_ids,
                "remote_read_index": sampled.remote_read_index,
                "local_read_index": sampled.local_read_index,
                "dynamic_fetch_plan": fetch_plan,
                "native_sampling": True,
            },
        )

    def run_train_unit_step(self, runtime: Any, unit: ExecutionUnit) -> dict[str, Any]:
        batch = self._batch_from_sampled_unit(unit, split="train")
        result = self.run_train_step(runtime, batch)
        result.setdefault("meta", {})["native_sampling"] = True
        result["meta"]["sampled_mfg_count"] = len(batch.mfgs) if isinstance(batch.mfgs, list) else 1
        manifest = batch.remote_manifest or {}
        fetch_plan = manifest.get("dynamic_fetch_plan")
        result["meta"]["dynamic_fetch_nodes"] = 0 if fetch_plan is None else int(fetch_plan.remote_read_index.numel())
        result["meta"]["time_slice_id"] = manifest.get("time_slice_id")
        result["meta"]["batch_offset"] = manifest.get("batch_offset")
        return result

    def run_eval_unit_step(self, runtime: Any, unit: ExecutionUnit) -> dict[str, Any]:
        batch = self._batch_from_sampled_unit(unit, split="test")
        result = self.run_eval_step(runtime, batch)
        result.setdefault("meta", {})["native_sampling"] = True
        result["meta"]["sampled_mfg_count"] = len(batch.mfgs) if isinstance(batch.mfgs, list) else 1
        manifest = batch.remote_manifest or {}
        fetch_plan = manifest.get("dynamic_fetch_plan")
        result["meta"]["dynamic_fetch_nodes"] = 0 if fetch_plan is None else int(fetch_plan.remote_read_index.numel())
        result["meta"]["time_slice_id"] = manifest.get("time_slice_id")
        result["meta"]["batch_offset"] = manifest.get("batch_offset")
        return result

    def run_eval_step(self, runtime: Any, batch: BatchData) -> dict[str, Any]:
        if isinstance(batch, STGraphBlob) or hasattr(batch, "current_graph"):
            return self._run_stgraph_step(runtime, batch, train=False)
        with torch.no_grad():
            fetch_result = self._execute_and_patch_fetch(runtime, batch)
            result = run_batch(
                model=runtime.model,
                batch=batch,
                task_adapter=self.task_adapter,
                optimizer=None,
                train=False,
            )
            if fetch_result is not None:
                remote_count = 0 if fetch_result.remote_read_index is None else int(fetch_result.remote_read_index.numel())
                local_count = 0 if fetch_result.local_read_index is None else int(fetch_result.local_read_index.numel())
                result.setdefault("meta", {})["fetch_remote_rows"] = remote_count
                result["meta"]["fetch_local_rows"] = local_count
            return result

    def run_predict_step(self, runtime: Any, batch: BatchData) -> dict[str, Any]:
        if isinstance(batch, STGraphBlob) or hasattr(batch, "current_graph"):
            result = self._run_stgraph_step(runtime, batch, train=False)
            pred = result.get("output", {}).get("node_pred")
            targets = result.get("targets")
            return {
                "predictions": [] if pred is None else pred.detach().cpu().view(-1).tolist(),
                "targets": None if targets is None else targets.detach().cpu().view(-1).tolist(),
                "meta": {"metrics": result.get("metrics", {})},
            }
        result = self.run_eval_step(runtime, batch)
        output = result.get("output", {})
        pred = output.get("node_pred")
        if pred is None:
            pred = output.get("pos_score")
        predictions = [] if pred is None else pred.detach().cpu().view(-1).tolist()
        targets = None if batch.labels is None else batch.labels.detach().cpu().view(-1).tolist()
        return {"predictions": predictions, "targets": targets, "meta": {"metrics": result.get("metrics", {})}}

    def _run_stgraph_step(self, runtime: Any, blob: STGraphBlob | Any, train: bool) -> dict[str, Any]:
        model = runtime.model
        model.train(mode=train)
        if train and runtime.optimizer is not None:
            runtime.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad() if train else torch.no_grad():
            raw_pred, state = model(blob, runtime.state.get("chunk_stgraph_state"))
            pred = raw_pred[-1] if isinstance(raw_pred, list) else raw_pred
            output: dict[str, Tensor]
            targets: Optional[Tensor] = None
            metrics: dict[str, float]
            if self.task_adapter.task_type in {"edge_predict", "link_prediction"}:
                batch = self._stgraph_edge_batch(blob)
                output = self._stgraph_edge_scores(pred, blob, batch)
                loss = self.task_adapter.compute_loss(output, batch)
                metrics = self.task_adapter.compute_metrics(output, batch)
            else:
                targets = extract_graph_labels(blob)
                if targets is None:
                    raise RuntimeError("Chunk STGraph training requires node labels in graph dstdata['y'] or targets/node_targets.pth")
                targets = targets.to(device=pred.device, dtype=pred.dtype)
                if targets.shape != pred.shape:
                    targets = targets.view_as(pred)
                loss = torch.nn.functional.mse_loss(pred, targets)
                metrics = {"mse": float(loss.detach().item())}
                output = {"node_pred": pred}
            if train and loss.requires_grad and runtime.optimizer is not None:
                loss.backward()
                runtime.optimizer.step()
            runtime.state["chunk_stgraph_state"] = RNNStateManager.state_detach(state)
        return {
            "loss": loss.detach(),
            "metrics": metrics,
            "output": {key: value.detach() for key, value in output.items()},
            "targets": None if targets is None else targets.detach(),
            "meta": {
                "chain": "load_snapshot->chunk_decay->route_apply->state_fetch->state_transition->state_writeback",
                "stage_payloads": {
                    "chunk_decay": True,
                    "state_transition": True,
                    "edge_predict_head": self.task_adapter.task_type in {"edge_predict", "link_prediction"},
                },
                "model": str(self.runtime_config.get("model", {}).get("name", "stgraph")),
                "stgraph": True,
            },
        }

    def _stgraph_edge_batch(self, blob: STGraphBlob | Any) -> BatchData:
        graph = blob.current_graph if isinstance(blob, STGraphBlob) else blob
        snapshot_idx = int(getattr(graph, "flare_snapshot_id", 0))
        pos_src, pos_dst = self.part_data.to_edge_index(snapshot_index=snapshot_idx, global_ids=True)
        if pos_src.numel() == 0:
            raise RuntimeError(f"Chunk STGraph edge prediction found no positive edges in snapshot {snapshot_idx}")
        neg_src, neg_dst = self.neg_sampler.sample(
            pos_src.detach().cpu(),
            pos_dst.detach().cpu(),
            self.num_nodes,
            1,
            split="train",
        )
        return BatchData(
            mfgs=graph,
            node_ids=graph.dstdata.get("_ID", torch.arange(graph.num_dst_nodes(), dtype=torch.long)),
            pos_src=pos_src.long(),
            pos_dst=pos_dst.long(),
            neg_src=neg_src.long(),
            neg_dst=neg_dst.long(),
            chunk_id=snapshot_idx,
        )

    def _stgraph_edge_scores(self, embeddings: Tensor, blob: STGraphBlob | Any, batch: BatchData) -> dict[str, Tensor]:
        graph = blob.current_graph if isinstance(blob, STGraphBlob) else blob
        dst_ids = graph.dstdata.get("_ID")
        if dst_ids is None:
            try:
                import dgl

                dst_ids = graph.dstdata.get(dgl.NID)
            except Exception:
                dst_ids = None
        if dst_ids is None:
            dst_ids = torch.arange(embeddings.size(0), dtype=torch.long, device=embeddings.device)
        dst_ids = dst_ids.to(device=embeddings.device).long()
        if dst_ids.numel() != embeddings.size(0):
            dst_ids = dst_ids[: embeddings.size(0)]
        order = torch.argsort(dst_ids, stable=True)
        sorted_ids = dst_ids[order]
        sorted_emb = embeddings[order]

        def gather_pair(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
            src = src.to(device=embeddings.device).long()
            dst = dst.to(device=embeddings.device).long()
            src_row = torch.searchsorted(sorted_ids, src)
            dst_row = torch.searchsorted(sorted_ids, dst)
            max_row = max(0, int(sorted_ids.numel()) - 1)
            src_safe = src_row.clamp_max(max_row)
            dst_safe = dst_row.clamp_max(max_row)
            keep = (
                (src_row < int(sorted_ids.numel()))
                & (dst_row < int(sorted_ids.numel()))
                & (sorted_ids[src_safe] == src)
                & (sorted_ids[dst_safe] == dst)
            )
            if not keep.any():
                keep = torch.ones_like(src, dtype=torch.bool)
                src_safe = src_safe.clamp_max(max_row)
                dst_safe = dst_safe.clamp_max(max_row)
            return sorted_emb[src_safe[keep]], sorted_emb[dst_safe[keep]]

        pos_src_h, pos_dst_h = gather_pair(batch.pos_src, batch.pos_dst)
        neg_src_h, neg_dst_h = gather_pair(batch.neg_src, batch.neg_dst)
        return {
            "pos_score": (pos_src_h * pos_dst_h).sum(dim=-1),
            "neg_score": (neg_src_h * neg_dst_h).sum(dim=-1),
        }
