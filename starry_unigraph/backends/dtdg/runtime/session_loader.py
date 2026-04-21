"""FlareRuntimeLoader: split management and iteration for Flare DTDG.

Wraps :class:`STGraphLoader` with train/val/test split computation,
chunk-order / chunk-decay configuration, and exposes
``iter_train`` / ``iter_eval`` / ``iter_predict`` iterators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import dgl
import torch
from dgl.heterograph import DGLBlock

from starry_unigraph.data.partition import PartitionData
from starry_unigraph.runtime._split import normalize_split_ratio, split_bounds
from .loader import STGraphLoader
from .state import STGraphBlob


# ---------------------------------------------------------------------------
# Data types (formerly in core/)
# ---------------------------------------------------------------------------

@dataclass
class SnapshotRoutePlan:
    route_type: str
    cache_policy: str

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "SnapshotRoutePlan"}


@dataclass
class DTDGWindowState:
    window_size: int
    last_snapshot: int | None = None
    stored_windows: int = 0

    def store(self, snapshot_index: int) -> None:
        self.last_snapshot = snapshot_index
        self.stored_windows += 1

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DTDGBatch:
    index: int
    split: str
    window_size: int
    route_plan: SnapshotRoutePlan
    adjacency: list[list[float]]
    dense_features: list[float]
    graph: Any = None
    graph_meta: dict[str, Any] = field(default_factory=dict)
    chain: str = "load_snapshot->route_apply->state_fetch->state_transition->state_writeback"

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "chain": self.chain,
            "window": {"snapshot_id": self.index, "window_size": self.window_size},
            "route_plan": self.route_plan.describe(),
            "adjacency": self.adjacency,
            "dense_features": self.dense_features,
            "graph_meta": self.graph_meta,
        }


def _feature_summary(x: torch.Tensor) -> list[float]:
    if x.numel() == 0:
        return [0.0, 0.0]
    mean_x = x.float().mean(dim=0)
    if mean_x.numel() == 1:
        return [float(mean_x[0].item()), 0.0]
    return [float(mean_x[0].item()), float(mean_x[1].item())]


def _parse_chunk_order(mode: Any, chunk_count: int) -> torch.Tensor | None:
    if chunk_count <= 0:
        return None
    if mode in (None, "none"):
        return None
    if mode == "rand":
        return torch.randperm(chunk_count, dtype=torch.long)
    if mode == "reverse":
        return torch.arange(chunk_count - 1, -1, -1, dtype=torch.long)
    return torch.arange(chunk_count, dtype=torch.long)


def _parse_chunk_decay(value: Any, chunk_count: int) -> list[int]:
    if chunk_count <= 0 or value in (None, "none", [], ()):
        return []
    if isinstance(value, int):
        steps = max(1, min(chunk_count, value))
        return list(range(1, steps + 1))
    if isinstance(value, (list, tuple)):
        parsed = [max(1, min(chunk_count, int(item))) for item in value]
        return sorted(set(parsed))
    if value == "half":
        return [max(1, chunk_count // 2)]
    if value == "all":
        return list(range(1, chunk_count + 1))
    return []


def _block_to_batch(
    block_or_blob: DGLBlock | STGraphBlob,
    split: str,
    window_size: int,
    route_plan: SnapshotRoutePlan,
) -> DTDGBatch:
    blob_meta: dict[str, Any] | None = None
    block = block_or_blob
    if isinstance(block_or_blob, STGraphBlob):
        blob_meta = block_or_blob.describe()
        block = block_or_blob.current_graph
    src, _ = block.edges()
    local_src = int((src < block.num_dst_nodes()).sum().item())
    boundary_src = int(src.numel() - local_src)
    edge_total = max(1, int(src.numel()))
    dense_features = _feature_summary(block.dstdata["x"] if "x" in block.dstdata else block.srcdata["x"])
    snapshot_index = 0
    partition_id = -1
    if "snapshot_index" in block.edata and block.edata["snapshot_index"].numel() > 0:
        snapshot_index = int(block.edata["snapshot_index"][0].item())
    if "partition_id" in block.edata and block.edata["partition_id"].numel() > 0:
        partition_id = int(block.edata["partition_id"][0].item())
    return DTDGBatch(
        index=snapshot_index,
        split=split,
        window_size=window_size,
        route_plan=route_plan,
        adjacency=[
            [local_src / edge_total, boundary_src / edge_total],
            [float(block.num_dst_nodes()) / max(1, int(block.num_src_nodes())), 1.0],
        ],
        dense_features=dense_features,
        graph=block,
        graph_meta={
            "num_src_nodes": int(block.num_src_nodes()),
            "num_dst_nodes": int(block.num_dst_nodes()),
            "num_edges": int(block.num_edges()),
            "partition_id": partition_id,
            "flare_is_full_snapshot": bool(getattr(block, "flare_is_full_snapshot", True)),
            "flare_remap": getattr(block, "flare_remap", {}),
            "window_blob": blob_meta,
        },
    )


@dataclass
class FlareRuntimeLoader:
    """High-level loader managing splits, chunk ordering, and iteration.

    Wraps a :class:`PartitionData` + :class:`STGraphLoader` and exposes
    three iteration modes:

    - :meth:`iter_train` — yields :class:`STGraphBlob` (multi-frame,
      with chunk-decay support).
    - :meth:`iter_eval` — yields :class:`DTDGBatch` with
      ``chain="snapshot_warmup"`` for pre-train snapshots and
      ``chain="snapshot_eval"`` for actual validation.
    - :meth:`iter_predict` — same structure as eval but with
      ``chain="snapshot_predict"``.

    Created via :meth:`from_partition_data` which parses config for
    window size, chunk order, chunk decay, split ratio, etc.
    """

    partition_id: int
    route_plan: SnapshotRoutePlan
    window_state: DTDGWindowState
    partition_data: PartitionData
    graph_loader: STGraphLoader
    device: str
    rank: int
    world_size: int
    chunk_index: torch.Tensor
    train_chunk_order: Any = None
    train_chunk_decay: list[int] = field(default_factory=list)
    train_num_full_snaps: int = 1
    split_ratio: dict[str, float] | None = None
    cursor: int = 0

    def __post_init__(self) -> None:
        self.split_ratio = normalize_split_ratio(self.split_ratio)
        total = len(self.partition_data)
        self._split_bounds_cache = {
            s: split_bounds(total, s, self.split_ratio)
            for s in ("train", "val", "test")
        }

    @classmethod
    def from_partition_data(
        cls,
        data: PartitionData,
        device: str,
        rank: int,
        world_size: int,
        config: dict[str, Any],
    ) -> "FlareRuntimeLoader":
        chunk_index = data.node_data["c"][0].item().long() if "c" in data.node_data else torch.zeros(data.num_dst_nodes, dtype=torch.long)
        part_tensor = data.node_data["partition_id"][0].item() if "partition_id" in data.node_data else torch.zeros(data.num_dst_nodes, dtype=torch.long)
        partition_id = int(part_tensor[0].item()) if part_tensor.numel() > 0 else rank
        route_plan = SnapshotRoutePlan(
            route_type=str(config["graph"]["route"]),
            cache_policy=str(config["runtime"]["cache"]),
        )
        graph_loader = STGraphLoader.from_partition_data(
            data=data,
            device=device,
            chunk_index=chunk_index,
            rank=rank,
            size=world_size,
        )
        return cls(
            partition_id=partition_id,
            route_plan=route_plan,
            window_state=DTDGWindowState(window_size=int(config["model"]["window"]["size"])),
            partition_data=data,
            graph_loader=graph_loader,
            device=device,
            rank=rank,
            world_size=world_size,
            chunk_index=chunk_index,
            train_chunk_order=config.get("dtdg", {}).get("chunk_order"),
            train_chunk_decay=_parse_chunk_decay(config.get("dtdg", {}).get("chunk_decay"), graph_loader.chunk_count),
            train_num_full_snaps=max(1, int(config.get("dtdg", {}).get("num_full_snaps", 1))),
            split_ratio=config.get("data", {}).get("split_ratio"),
        )

    def load_block(self, snapshot_index: int) -> DGLBlock:
        return self.graph_loader.fetch_graph(snapshot_index)

    def load_snapshot(self, snapshot_index: int, split: str) -> DTDGBatch:
        return _block_to_batch(
            block_or_blob=self.load_block(snapshot_index),
            split=split,
            window_size=self.window_state.window_size,
            route_plan=self.route_plan,
        )

    def _split_range(self, split: str) -> range:
        start, end = self._split_bounds_cache[split]
        return range(start, end)

    def iter_train(self, split: str = "train") -> Iterable[Any]:
        start, end = self._split_bounds_cache["train"]
        train_loader = STGraphLoader.from_partition_data(
            data=self.partition_data[start:end],
            device=self.device,
            chunk_index=self.chunk_index,
            rank=self.rank,
            size=self.world_size,
        )
        chunk_order = _parse_chunk_order(self.train_chunk_order, train_loader.chunk_count)
        for blob in train_loader(
            chunk_order=chunk_order,
            chunk_decay=self.train_chunk_decay,
            num_full_snaps=self.train_num_full_snaps,
        ):
            self.cursor = blob.snapshot_index + 1
            yield blob

    def iter_eval(self, split: str = "val") -> Iterable[DTDGBatch]:
        _, train_end = self._split_bounds_cache["train"]
        eval_start, eval_end = self._split_bounds_cache[split]
        for index in range(eval_end):
            batch = self.load_snapshot(index, split=split)
            if index < train_end or index < eval_start:
                batch.chain = "snapshot_warmup"
            else:
                batch.chain = "snapshot_eval"
            self.cursor = index + 1
            yield batch

    def iter_predict(self, split: str = "test") -> Iterable[DTDGBatch]:
        _, train_end = self._split_bounds_cache["train"]
        pred_start, pred_end = self._split_bounds_cache[split]
        for index in range(pred_end):
            batch = self.load_snapshot(index, split=split)
            if index < train_end or index < pred_start:
                batch.chain = "snapshot_warmup"
            else:
                batch.chain = "snapshot_predict"
            self.cursor = index + 1
            yield batch

    def build_snapshot_index(self) -> dict[str, Any]:
        return self.graph_loader.build_snapshot_index() | {
            "window_size": self.window_state.window_size,
            "partition_id": self.partition_id,
            "pipeline": "flare_native",
        }

    def dump_state(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "partition_id": self.partition_id,
            "snapshot_count": len(self.partition_data),
            "snapshot_index": self.build_snapshot_index(),
            "window_state": self.window_state.describe(),
            "route_plan": self.route_plan.describe(),
            "local_nodes": int(self.partition_data.num_dst_nodes),
            "chunk_count": int(self.chunk_index.max().item() + 1) if self.chunk_index.numel() > 0 else 0,
        }

    def describe_window_state(self) -> dict[str, Any]:
        return self.window_state.describe()

    def describe_route_cache(self) -> dict[str, Any]:
        return self.route_plan.describe() | {
            "partition_id": self.partition_id,
            "local_nodes": int(self.partition_data.num_dst_nodes),
            "chunk_count": int(self.chunk_index.max().item() + 1) if self.chunk_index.numel() > 0 else 0,
            "loader": "STGraphLoader",
            "chunk_order": self.train_chunk_order,
            "chunk_decay": self.train_chunk_decay,
            "num_full_snaps": self.train_num_full_snaps,
        }

    def run_train_step(self, runtime: Any, blob: Any) -> dict[str, Any]:
        """Execute one training step using Flare step handler."""
        from .training import run_flare_train_step
        return run_flare_train_step(runtime, blob, {"meta": {}})

    def run_eval_step(self, runtime: Any, batch: Any) -> dict[str, Any]:
        """Execute one evaluation step using Flare step handler."""
        from .training import run_flare_eval_step
        return run_flare_eval_step(runtime, batch, {"meta": {}})

    def run_predict_step(self, runtime: Any, batch: Any) -> dict[str, Any]:
        """Execute one prediction step using Flare step handler."""
        from .training import run_flare_predict_step
        return run_flare_predict_step(runtime, batch, {"meta": {}})
