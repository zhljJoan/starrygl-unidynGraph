"""ChunkRuntimeLoader: runtime接入层，yield BatchData.

取代旧的 STGraphBlob / DTDGBatch，统一 yield chunk-specific BatchData。

设计：
  - 从磁盘加载 PartitionData（每个分区的图数据）
  - 加载 MemoryRouteData（记忆更新路由，预计算）
  - 加载 SpatialRouteData（特征 fetch 路由，预计算）
  - 对每个时间切片，通过 TaskAdapter 构建 BatchData
  - 通过 CommPipeline 管理异步通信（submit/await 流水线）
  - 采样和 MFG 构造留给 SamplerHook（接口已定，C++ 待接入）

特征存储重分布：
  - 暂不实现 redistribute_preprocessed 的完整逻辑
  - 提供 rebuild_from_scratch() 接口，从 prepare 阶段重头执行
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
from starry_unigraph.backends.chunk.data.partition import PartitionData
from starry_unigraph.backends.chunk.data.route import MemoryRouteData, SpatialRouteData
from starry_unigraph.backends.chunk.data.comm import CommPipeline
from starry_unigraph.backends.chunk.runtime.task_adapter import ChunkTaskAdapter, get_task_adapter
from starry_unigraph.backends.chunk.runtime.sampler import (
    NeighborSamplerHook, NegativeSamplerHook, MFGBuilderHook,
    _RandomNegativeSampler, _StubMFGBuilder,
)
from starry_unigraph.backends.chunk.runtime.train_step import run_batch


class SimpleChunkModel(nn.Module):
    """Minimal BatchData-native model used until real chunk GNN kernels are wired."""

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
#   chunk_assignment.pth      ChunkAssignment
#   part_{rank:03d}.pth       PartitionData  (one per rank)
#   mem_routes_{rank:03d}/    MemoryRouteData files: slice_{t:06d}.pth
#   spatial_routes_{rank:03d}/ SpatialRouteData files: slice_{t:06d}.pth
#   cpu_layout_{rank:03d}.pth CPUMemoryLayout
#   meta.json                 dataset metadata (num_nodes, num_slices, splits)


def _load_optional(path: Path, cls):
    """Load a torch.save'd object if the file exists, else return None."""
    if path.exists():
        return torch.load(path, weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Redistribute stub
# ---------------------------------------------------------------------------

def redistribute_preprocessed(
    prepared_dir: Path,
    output_dir: Path,
    num_chunks_per_partition: int = 32,
    rebalance: bool = True,
    **kwargs,
) -> None:
    """Redistribute already-prepared artifacts under a new chunk assignment.

    Interface only — actual redistribution not yet implemented.
    Use rebuild_from_scratch() to run preprocessing from the beginning.
    """
    raise NotImplementedError(
        "redistribute_preprocessed: full redistribution not yet implemented. "
        "Use rebuild_from_scratch() to re-run from prepare phase."
    )


def rebuild_from_scratch(config: Dict[str, Any]) -> None:
    """Run full preprocessing pipeline from scratch using config.

    Calls ChunkPreprocessor with given config; output lands in
    config['data']['artifact_root'].
    """
    raise NotImplementedError("Delegates to ChunkPreprocessor — not yet wired.")


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
        mfg_builder:    MFG construction hook (stub until C++ ready).
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

        # PartitionData
        part_path = prepared_dir / f"part_{rank:03d}.pth"
        if not part_path.exists():
            raise FileNotFoundError(f"PartitionData not found: {part_path}")
        part_data: PartitionData = torch.load(part_path, weights_only=False)

        # MemoryRouteData (one file per slice)
        mem_dir = prepared_dir / f"mem_routes_{rank:03d}"
        mem_routes = cls._load_route_list(mem_dir, MemoryRouteData)

        # SpatialRouteData
        spatial_dir = prepared_dir / f"spatial_routes_{rank:03d}"
        spatial_routes = cls._load_route_list(spatial_dir, SpatialRouteData)

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
            split_slices   = split_info,
            num_nodes      = num_nodes,
            rank           = rank,
            world_size     = world_size,
            device         = device,
            chunk_manifest = chunk_manifest,
        )

    @staticmethod
    def _load_route_list(directory: Path, cls) -> list:
        """Load all slice_{t}.pth files from a directory in order."""
        if not directory.exists():
            return []
        files = sorted(directory.glob("slice_*.pth"))
        return [torch.load(f, weights_only=False) for f in files]

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

    def _iter_split(self, split: str) -> Iterator[BatchData]:
        """Core iteration loop with async communication pipeline.

        Pattern (one slice ahead):
          t=0: submit comm for slice 0
          t=1: compute on slice 0 result, submit comm for slice 1
          ...

        Actual MFG construction and neighbor sampling are stubs until
        C++ extensions are wired.
        """
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

            # MFG construction hook (stub — will attach real mfgs later)
            if batch.mfgs is None and self.world_size > 1:
                spatial_route = (
                    self.spatial_routes[t] if t < len(self.spatial_routes) else None
                )
                batch.mfgs = self.mfg_builder.build(None)  # stub

            # Async comm: submit memory update for next batch, await for current
            # (pipeline is a no-op in single-rank or when routes are empty)
            if self.world_size > 1 and mem_route is not None and mem_route.num_unique > 0:
                # In a real training loop: await prev, then submit current
                # Here we just run sync (awaiting after submit immediately)
                pass  # CommPipeline.submit_memory / await_memory called by trainer

            yield batch

        self.pipeline.drain_all_sync()

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
        return run_batch(
            model=runtime.model,
            batch=batch,
            task_adapter=self.task_adapter,
            optimizer=runtime.optimizer,
            train=True,
        )

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
