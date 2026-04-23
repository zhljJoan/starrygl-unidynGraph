"""CTDG event-stream dataset and batch containers.

- :class:`CTDGDataBatch` — a mini-batch of temporal events (src, dst, ts,
  edge features).
- :class:`TGTemporalDataset` — loads raw temporal events from disk, splits
  them into train/val/test, provides batched iteration and sampler graph
  construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from starry_unigraph.runtime._split import normalize_split_ratio, split_bounds
from starry_unigraph.data import load_raw_temporal_events


@dataclass
class CTDGDataBatch:
    """A mini-batch of continuous-time temporal events.

    Attributes:
        index: Batch index within the current epoch.
        split: ``"train"``, ``"val"``, or ``"test"``.
        event_ids: Global event indices, shape ``[B]``.
        src: Source node IDs, shape ``[B]``.
        dst: Destination node IDs, shape ``[B]``.
        ts: Timestamps, shape ``[B]``.
        edge_feat: Edge feature matrix, shape ``[B, E]``.
    """
    index: int
    split: str
    event_ids: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    ts: torch.Tensor
    edge_feat: torch.Tensor
    src_node_feat: torch.Tensor | None = None
    dst_node_feat: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.src.numel())

    @property
    def is_empty(self) -> bool:
        return self.size == 0


class TGTemporalDataset:
    """Temporal graph dataset with train/val/test splitting.

    Loads raw temporal events via :func:`load_raw_temporal_events`, stores
    them as flat tensors (src, dst, ts, edge_feat), and provides batched
    iteration with distributed sharding.

    Args:
        root: Root directory containing dataset folders.
        name: Dataset name (subdirectory under *root*).
        split_ratio: Optional custom split ratio dict.
        config: Full session config dict.

    Example::

        dataset = TGTemporalDataset("/data/raw", "wikipedia")
        for batch in dataset.iter_batches("train", batch_size=200):
            print(batch.src.shape)
    """
    def __init__(
        self,
        root: str | Path,
        name: str,
        split_ratio: dict[str, float] | None = None,
        config: dict | None = None,
    ):
        self.root = Path(root).expanduser().resolve() / name
        merged_config = dict(config or {})
        merged_config.setdefault("data", {})
        merged_config["data"]["split_ratio"] = split_ratio or merged_config["data"].get("split_ratio")
        raw = load_raw_temporal_events(root=Path(root).expanduser().resolve(), dataset_name=name, config=merged_config)
        self.event_ids = torch.arange(raw.num_edges, dtype=torch.long)
        self.src = raw.src.long()
        self.dst = raw.dst.long()
        self.ts = raw.ts.float()
        self.int_roll = torch.zeros(raw.num_edges, dtype=torch.long)
        self.ext_roll = torch.zeros(raw.num_edges, dtype=torch.long)
        self.num_nodes = int(raw.num_nodes)
        self.num_edges = int(raw.num_edges)
        self.edge_feat = raw.edge_feat.float()
        self.edge_feat_dim = int(self.edge_feat.size(-1))
        self.node_temporal_features = raw.node_temporal_features
        self.node_feat_dim = 0 if raw.node_temporal_features is None else raw.node_temporal_features.dim
        self._split_cache: dict[str, torch.Tensor] = {}
        self._sampler_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.split_ratio = normalize_split_ratio(merged_config["data"].get("split_ratio"))
        self.node_parts: torch.Tensor | None = None
        self.edge_parts: torch.Tensor | None = None
        self.partition_rank: int = 0
        self.partition_world_size: int = 1
        self._node_feat_history = self._build_node_feat_history()

    def configure_partition(
        self,
        *,
        node_parts: torch.Tensor | None,
        edge_parts: torch.Tensor | None,
        rank: int,
        world_size: int,
    ) -> None:
        self.node_parts = node_parts
        self.edge_parts = edge_parts
        self.partition_rank = rank
        self.partition_world_size = world_size

    def split_event_ids(self, split: str) -> torch.Tensor:
        cached = self._split_cache.get(split)
        if cached is not None:
            return cached
        start, end = split_bounds(total=self.num_edges, split=split, split_ratio=self.split_ratio)
        event_ids = torch.arange(start, end, dtype=torch.long)
        self._split_cache[split] = event_ids
        return event_ids

    def _build_node_feat_history(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        table = self.node_temporal_features
        if table is None or table.size == 0 or table.dim == 0:
            return {}
        history: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        unique_nodes = torch.unique(table.node_ids, sorted=True)
        for node_id in unique_nodes.tolist():
            mask = table.node_ids == int(node_id)
            ts = table.ts[mask].float()
            vals = table.values[mask].float()
            order = torch.argsort(ts, stable=True)
            history[int(node_id)] = (ts[order], vals[order])
        return history

    def lookup_node_features(self, node_ids: torch.Tensor, ts: torch.Tensor) -> torch.Tensor | None:
        if self.node_feat_dim <= 0 or not self._node_feat_history:
            return None
        out = torch.zeros(node_ids.numel(), self.node_feat_dim, dtype=torch.float32)
        unique_nodes = torch.unique(node_ids.long(), sorted=True)
        for node_id in unique_nodes.tolist():
            history = self._node_feat_history.get(int(node_id))
            if history is None:
                continue
            hist_ts, hist_vals = history
            node_mask = node_ids.long() == int(node_id)
            row_ids = torch.nonzero(node_mask, as_tuple=False).view(-1)
            query_ts = ts[row_ids].float()
            idx = torch.searchsorted(hist_ts, query_ts, right=True) - 1
            valid = idx >= 0
            if valid.any():
                out[row_ids[valid]] = hist_vals[idx[valid]]
        return out

    def iter_batches(self, split: str, batch_size: int) -> Iterator[CTDGDataBatch]:
        """Iterate over mini-batches for the given split.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            batch_size: Number of events per batch.

        Yields:
            :class:`CTDGDataBatch` instances.
        """
        indices = self.split_event_ids(split)
        global_batch_size = batch_size
        if self.partition_world_size > 1 and self.edge_parts is not None:
            global_batch_size = batch_size * self.partition_world_size

        for batch_index, start in enumerate(range(0, int(indices.numel()), global_batch_size)):
            event_ids = indices[start : start + global_batch_size]
            if self.partition_world_size > 1 and self.edge_parts is not None:
                local_mask = self.edge_parts[event_ids] == self.partition_rank
                event_ids = event_ids[local_mask]
            batch_src = self.src[event_ids]
            batch_dst = self.dst[event_ids]
            batch_ts = self.ts[event_ids]
            yield CTDGDataBatch(
                index=batch_index,
                split=split,
                event_ids=event_ids,
                src=batch_src,
                dst=batch_dst,
                ts=batch_ts,
                edge_feat=self.edge_feat[event_ids],
                src_node_feat=self.lookup_node_features(batch_src, batch_ts),
                dst_node_feat=self.lookup_node_features(batch_dst, batch_ts),
            )

    def sampler_graph(self, split: str) -> dict[str, torch.Tensor]:
        """Build the temporal neighbor graph for the BTS sampler.

        Returns a dict with ``"row"``, ``"col"``, ``"eid"``, ``"ts"``
        tensors covering all events visible to the given split (train
        events for ``"train"``, train+val for ``"val"``, all for ``"test"``).

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Dict of CPU tensors suitable for :func:`build_temporal_neighbor_block`.
        """
        cached = self._sampler_cache.get(split)
        if cached is not None:
            return cached
        train_start, train_end = split_bounds(total=self.num_edges, split="train", split_ratio=self.split_ratio)
        _, val_end = split_bounds(total=self.num_edges, split="val", split_ratio=self.split_ratio)
        _, test_end = split_bounds(total=self.num_edges, split="test", split_ratio=self.split_ratio)
        if split == "train":
            indices = torch.arange(train_start, train_end, dtype=torch.long)
        elif split == "val":
            indices = torch.arange(0, val_end, dtype=torch.long)
        elif split == "test":
            indices = torch.arange(0, test_end, dtype=torch.long)
        else:
            raise KeyError(f"Unsupported CTDG split: {split}")
        row = self.src[indices].long()
        cached = {
            "row": row,
            "col": self.dst[indices].long(),
            "eid": indices.long(),
            "ts": self.ts[indices].long(),
        }
        self._sampler_cache[split] = cached
        return cached

    def describe(self) -> dict[str, int]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edge_feat_dim": self.edge_feat_dim,
            "node_feat_dim": self.node_feat_dim,
        }
