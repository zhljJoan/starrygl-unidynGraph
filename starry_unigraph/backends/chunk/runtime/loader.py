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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData
from starry_unigraph.backends.chunk.data.graph_store import ChunkGraphStore
from starry_unigraph.backends.chunk.data.partition import PartitionData
from starry_unigraph.backends.chunk.data.plans import (
    EventView,
    ChunkPlacement,
    CommPlanBundle,
    ExecutionUnit,
    FetchPlan,
    GraphBatchEnvelope,
    StateSyncPlan,
    TemporalIndexView,
)
from starry_unigraph.backends.chunk.data.route import MemoryRouteData, SpatialRouteData
from starry_unigraph.backends.chunk.data.comm import CommPipeline, MemoryResult, validate_training_comm_backend
from starry_unigraph.backends.chunk.data.dist_index import dist_index_part
from starry_unigraph.backends.chunk.runtime.task_adapter import ChunkTaskAdapter, get_task_adapter
from starry_unigraph.backends.chunk.runtime.sampler import (
    NegativeSamplerHook, MFGBuilderHook,
)
from starry_unigraph.backends.chunk.runtime.train_step import run_batch
from starry_unigraph.backends.chunk.runtime.event_engine import MemShareEventEngine


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

        # SpatialRouteData
        spatial_routes = cls._load_route_list(prepared_dir, "spatial_routes", rank, SpatialRouteData)

        # Meta
        import json
        meta_path = prepared_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        manifest_path = prepared_dir / "partitions" / "manifest.json"
        chunk_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        num_nodes   = meta.get("num_nodes", 0)
        split_info  = meta.get("splits", {"train": [], "val": [], "test": []})

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

        placement = cls._load_placement(prepared_dir, part_data)
        if hasattr(neg_sampler, "configure_pools") and placement is not None:
            local_pool = torch.nonzero(placement.node_owner == int(rank), as_tuple=False).flatten().long()
            global_pool = torch.arange(int(num_nodes), dtype=torch.long)
            neg_sampler.configure_pools(local_dst_pool=local_pool, global_dst_pool=global_pool)
        temporal_index = cls._load_temporal_index(prepared_dir, rank, placement)
        graph_store = ChunkGraphStore.from_partition_data(
            part_data,
            placement=placement,
            temporal_index=temporal_index,
        )
        event_engine_cfg = dict(sampler_cfg.get("memshare", sampler_cfg))
        event_engine_cfg.setdefault("event_batch_size", int(config.get("train", {}).get("batch_size", 0)))
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
        )

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
    def _load_placement(prepared_dir: Path, part_data: PartitionData) -> ChunkPlacement | None:
        placement_path = prepared_dir / "placement.pth"
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
        master_dist_index = payload.get("canonical_nid_dist", payload.get("master_dist_index"))
        read_dist_by_part = payload.get("read_dist_index_by_part", payload.get("local_nid_dist_by_part"))
        read_dist_index = None
        if isinstance(read_dist_by_part, list) and len(read_dist_by_part) > 0:
            # local_nid_dist_by_part is not a global lookup table. Keep it out
            # of hot-path indexing until prepare emits read_dist_index[rank].
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

    # ---------------------------------------------------------------------------
    # Core iterators — yield BatchData
    # ---------------------------------------------------------------------------

    def iter_train(self, split: str = "train") -> Iterator[BatchData]:
        """Iterate training slices, yielding BatchData."""
        yield from self._iter_split(split)

    def iter_eval(self, split: str = "val") -> Iterator[BatchData]:
        """Iterate validation slices, yielding BatchData."""
        yield from self._iter_split(split)

    def iter_predict(self, split: str = "test") -> Iterator[BatchData]:
        """Iterate test slices, yielding BatchData."""
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
        return self.task_adapter.task_type in {"edge_predict", "link_prediction"}

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

        return CommPlanBundle(fetch=fetch, propagation=None, state_sync=state_sync)

    def _make_ctdg_base_comm_plan(self, time_slice_id: int) -> CommPlanBundle:
        """Base CTDG plan before sampling.

        CTDG feature/memory fetch depends on the sampled remote read set, so it
        is built later from ``CTDGSampleResult.remote_read_index``.  The only
        stable pre-sampling route is the memory writeback/state-sync relation
        for the time slice.
        """
        return CommPlanBundle(fetch=None, propagation=None, state_sync=self._make_state_sync_plan(time_slice_id))

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
                replica_node_ids=None if route.replica_idx is None else route.unique_nodes[route.replica_idx].contiguous(),
                replica_owners=None,
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
    ) -> FetchPlan | None:
        """Build a CTDG fetch plan from sampled packed DistIndex values."""
        if remote_read_index is None or remote_read_index.numel() == 0:
            return None
        remote_read_index = remote_read_index.long().contiguous()
        owners = dist_index_part(remote_read_index).long().contiguous()
        if remote_read_index.numel() > 1:
            order = torch.argsort(owners, stable=True)
            remote_read_index = remote_read_index[order].contiguous()
            owners = owners[order].contiguous()
        cache_key = (
            int(self.graph_store.placement_version if self.graph_store is not None else 0),
            int(remote_read_index.numel()),
            remote_read_index.cpu().numpy().tobytes(),
        )
        cached = self._ctdg_fetch_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        plan = FetchPlan(
            block_id=int(block_id),
            placement_version=self.graph_store.placement_version if self.graph_store is not None else 0,
            feature_node_ids=torch.empty(0, dtype=torch.long, device=remote_read_index.device),
            feature_owners=owners,
            remote_read_index=remote_read_index,
            local_read_index=None if local_read_index is None else local_read_index.long().contiguous(),
            memory_read_index=remote_read_index,
            cache_policy="sampled_packed_dist_index",
        )
        self._ctdg_fetch_plan_cache[cache_key] = plan
        return plan

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
        return SimpleChunkModel(
            num_nodes=self.num_nodes,
            hidden_dim=hidden_dim,
            task_type=self.task_adapter.task_type,
        ).to(self.device)

    def run_train_step(self, runtime: Any, batch: BatchData) -> dict[str, Any]:
        result = run_batch(
            model=runtime.model,
            batch=batch,
            task_adapter=self.task_adapter,
            optimizer=runtime.optimizer,
            train=True,
        )
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

        sampled = self.event_engine.sample(unit)
        events = self.graph_store.temporal_events()
        start, end = int(view.event_start), int(view.event_end)
        pos_src = events.src[start:end].long().contiguous()
        pos_dst = events.dst[start:end].long().contiguous()
        timestamps = events.ts[start:end].contiguous()
        neg_src, neg_dst = self.neg_sampler.sample(pos_src, pos_dst, self.num_nodes, 1, split=split)
        node_ids = sampled.input_nodes.long().contiguous()
        if node_ids.numel() == 0:
            node_ids = torch.cat([pos_src, pos_dst, neg_src, neg_dst], dim=0).unique(sorted=True).contiguous()

        fetch_plan = self._make_dynamic_fetch_plan(
            int(unit.block_id),
            sampled.remote_read_index,
            sampled.local_read_index,
        )
        return BatchData(
            mfgs=sampled.mfgs,
            node_ids=node_ids,
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
                "sampled_input_nodes": sampled.input_nodes,
                "sampled_output_nodes": sampled.output_nodes,
                "sampled_edge_ids": sampled.edge_ids,
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
        with torch.no_grad():
            return run_batch(
                model=runtime.model,
                batch=batch,
                task_adapter=self.task_adapter,
                optimizer=None,
                train=False,
            )

    def run_predict_step(self, runtime: Any, batch: BatchData) -> dict[str, Any]:
        result = self.run_eval_step(runtime, batch)
        output = result.get("output", {})
        pred = output.get("node_pred")
        if pred is None:
            pred = output.get("pos_score")
        predictions = [] if pred is None else pred.detach().cpu().view(-1).tolist()
        targets = None if batch.labels is None else batch.labels.detach().cpu().view(-1).tolist()
        return {"predictions": predictions, "targets": targets, "meta": {"metrics": result.get("metrics", {})}}
