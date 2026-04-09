from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from starry_unigraph.data import load_raw_temporal_events


@dataclass
class CTDGDataBatch:
    index: int
    split: str
    event_ids: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    ts: torch.Tensor
    edge_feat: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.src.numel())


def _normalize_split_ratio(split_ratio: dict[str, float] | None) -> dict[str, float]:
    ratio = dict(split_ratio or {})
    train = float(ratio.get("train", 0.7))
    val = float(ratio.get("val", 0.15))
    test = float(ratio.get("test", 0.15))
    train = max(0.0, train)
    val = max(0.0, val)
    test = max(0.0, test)
    total = train + val + test
    if total <= 0:
        return {"train": 0.7, "val": 0.15, "test": 0.15}
    return {"train": train / total, "val": val / total, "test": test / total}


def _split_bounds(total: int, split: str, split_ratio: dict[str, float]) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    raw = {key: total * split_ratio[key] for key in ("train", "val", "test")}
    counts = {key: int(raw[key]) for key in raw}
    remainder = total - sum(counts.values())
    if remainder > 0:
        order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
        for idx in range(remainder):
            counts[order[idx % len(order)]] += 1
    positive = [key for key in ("train", "val", "test") if split_ratio[key] > 0]
    if total >= len(positive):
        for key in positive:
            if counts[key] > 0:
                continue
            donor = max(positive, key=lambda item: counts[item])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[key] += 1
    train_end = min(total, counts["train"])
    val_end = min(total, train_end + counts["val"])
    if split == "train":
        return 0, min(total, train_end)
    if split == "val":
        return min(total, train_end), max(min(total, val_end), min(total, train_end))
    if split == "test":
        return min(total, val_end), total
    raise KeyError(f"Unsupported CTDG split: {split}")


class TGTemporalDataset:
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
        self._split_cache: dict[str, torch.Tensor] = {}
        self._sampler_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.split_ratio = _normalize_split_ratio(merged_config["data"].get("split_ratio"))

    def split_event_ids(self, split: str) -> torch.Tensor:
        cached = self._split_cache.get(split)
        if cached is not None:
            return cached
        start, end = _split_bounds(total=self.num_edges, split=split, split_ratio=self.split_ratio)
        event_ids = torch.arange(start, end, dtype=torch.long)
        self._split_cache[split] = event_ids
        return event_ids

    def iter_batches(self, split: str, batch_size: int, rank: int = 0, world_size: int = 1) -> Iterator[CTDGDataBatch]:
        indices = self.split_event_ids(split)
        if world_size > 1:
            indices = indices[rank::world_size]
        for batch_index, start in enumerate(range(0, int(indices.numel()), batch_size)):
            event_ids = indices[start : start + batch_size]
            yield CTDGDataBatch(
                index=batch_index,
                split=split,
                event_ids=event_ids,
                src=self.src[event_ids],
                dst=self.dst[event_ids],
                ts=self.ts[event_ids],
                edge_feat=self.edge_feat[event_ids],
            )

    def sampler_graph(self, split: str) -> dict[str, torch.Tensor]:
        cached = self._sampler_cache.get(split)
        if cached is not None:
            return cached
        train_start, train_end = _split_bounds(total=self.num_edges, split="train", split_ratio=self.split_ratio)
        _, val_end = _split_bounds(total=self.num_edges, split="val", split_ratio=self.split_ratio)
        _, test_end = _split_bounds(total=self.num_edges, split="test", split_ratio=self.split_ratio)
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
        }
