from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass
class RawTemporalEvents:
    src: torch.Tensor
    dst: torch.Tensor
    ts: torch.Tensor
    weight: torch.Tensor
    edge_feat: torch.Tensor
    num_nodes: int
    num_edges: int
    source: str


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_edge_row(parts: list[str]) -> tuple[int, int, float, float] | None:
    if len(parts) == 3:
        u, v, ts = parts
        w = 1.0
    elif len(parts) == 4:
        u, v, w, ts = parts
    elif len(parts) >= 5:
        _, u, v, ts = parts[:4]
        w = 1.0
    else:
        return None
    return int(float(u)), int(float(v)), float(ts), float(w)

#仅接口测试
def _mock_events(dataset_name: str, config: dict[str, Any]) -> RawTemporalEvents:
    snaps = max(1, _as_int(config.get("train", {}).get("snaps"), 8))
    event_count = max(16, snaps * 4)
    num_nodes = max(8, min(128, event_count // 2))
    src = torch.tensor([idx % num_nodes for idx in range(event_count)], dtype=torch.long)
    dst = torch.tensor([(idx + 1) % num_nodes for idx in range(event_count)], dtype=torch.long)
    ts = torch.arange(event_count, dtype=torch.float32)
    weight = torch.ones(event_count, dtype=torch.float32)
    edge_feat = torch.ones(event_count, 1, dtype=torch.float32)
    return RawTemporalEvents(
        src=src,
        dst=dst,
        ts=ts,
        weight=weight,
        edge_feat=edge_feat,
        num_nodes=num_nodes,
        num_edges=event_count,
        source=f"mock:{dataset_name}",
    )


def _resolve_dataset_dir(root: Path, dataset_name: str) -> Path:
    return root / dataset_name


def _load_from_ctdg_csv(dataset_dir: Path) -> RawTemporalEvents | None:
    edges_path = dataset_dir / "edges.csv"
    if not edges_path.exists():
        return None
    frame = pd.read_csv(edges_path)
    if frame.empty:
        return RawTemporalEvents(
            src=torch.empty(0, dtype=torch.long),
            dst=torch.empty(0, dtype=torch.long),
            ts=torch.empty(0, dtype=torch.float32),
            weight=torch.empty(0, dtype=torch.float32),
            edge_feat=torch.empty(0, 1, dtype=torch.float32),
            num_nodes=0,
            num_edges=0,
            source=str(edges_path),
        )
    src = torch.as_tensor(frame["src"].to_numpy(dtype="int64"))
    dst = torch.as_tensor(frame["dst"].to_numpy(dtype="int64"))
    ts_col = "time" if "time" in frame.columns else ("ts" if "ts" in frame.columns else frame.columns[-1])
    ts = torch.as_tensor(frame[ts_col].to_numpy(dtype="float32"))
    weight_col = "w" if "w" in frame.columns else ("weight" if "weight" in frame.columns else None)
    if weight_col is None:
        weight = torch.ones(src.numel(), dtype=torch.float32)
    else:
        weight = torch.as_tensor(frame[weight_col].to_numpy(dtype="float32"))
    num_nodes = int(max(int(src.max().item()), int(dst.max().item())) + 1) if src.numel() else 0
    feat_path = dataset_dir / "edge_features.pt"
    if feat_path.exists():
        edge_feat = torch.load(feat_path, map_location="cpu").float()
        if edge_feat.dim() == 1:
            edge_feat = edge_feat.view(-1, 1)
    else:
        edge_feat = torch.ones(src.numel(), 1, dtype=torch.float32)
    return RawTemporalEvents(
        src=src.long(),
        dst=dst.long(),
        ts=ts.float(),
        weight=weight.float(),
        edge_feat=edge_feat,
        num_nodes=num_nodes,
        num_edges=int(src.numel()),
        source=str(edges_path),
    )


def _load_from_edge_file(dataset_dir: Path, dataset_name: str) -> RawTemporalEvents | None:
    candidates = (
        dataset_dir / f"{dataset_name}.edges",
        dataset_dir / "edges.txt",
        dataset_dir / "edges.csv",
    )
    edge_path = next((item for item in candidates if item.exists()), None)
    if edge_path is None:
        return None
    src: list[int] = []
    dst: list[int] = []
    ts: list[float] = []
    weights: list[float] = []
    with edge_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")] if "," in line else [part for part in line.split() if part]
            parsed = _parse_edge_row(parts)
            if parsed is None:
                continue
            u, v, t, w = parsed
            src.append(u)
            dst.append(v)
            ts.append(t)
            weights.append(w)
    if not src:
        return None
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    ts_t = torch.tensor(ts, dtype=torch.float32)
    weight_t = torch.tensor(weights, dtype=torch.float32)
    order = torch.argsort(ts_t, stable=True)
    src_t = src_t[order]
    dst_t = dst_t[order]
    ts_t = ts_t[order]
    weight_t = weight_t[order]
    unique = torch.unique(torch.cat([src_t, dst_t], dim=0), sorted=True)
    mapping = {int(node.item()): idx for idx, node in enumerate(unique)}
    src_t = torch.tensor([mapping[int(v.item())] for v in src_t], dtype=torch.long)
    dst_t = torch.tensor([mapping[int(v.item())] for v in dst_t], dtype=torch.long)
    edge_feat = torch.ones(src_t.numel(), 1, dtype=torch.float32)
    return RawTemporalEvents(
        src=src_t,
        dst=dst_t,
        ts=ts_t,
        weight=weight_t,
        edge_feat=edge_feat,
        num_nodes=int(unique.numel()),
        num_edges=int(src_t.numel()),
        source=str(edge_path),
    )


def load_raw_temporal_events(root: Path | str, dataset_name: str, config: dict[str, Any]) -> RawTemporalEvents:
    if str(config.get("data", {}).get("format", "auto")).lower() == "mock":
        return _mock_events(dataset_name=dataset_name, config=config)
    root_path = Path(root).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(root_path, dataset_name)
    events = _load_from_ctdg_csv(dataset_dir)
    if events is None:
        events = _load_from_edge_file(dataset_dir, dataset_name)
    if events is None:
        return _mock_events(dataset_name=dataset_name, config=config)
    return events


def _compute_degrees(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    out_deg = torch.zeros(num_nodes, dtype=torch.float32)
    if edge_index.numel() == 0:
        return in_deg, out_deg
    src = edge_index[0].long()
    dst = edge_index[1].long()
    in_deg.index_add_(0, dst, edge_weight)
    out_deg.index_add_(0, src, edge_weight)
    return in_deg, out_deg


def build_snapshot_dataset_from_events(events: RawTemporalEvents, snaps: int, input_x = None, input_y = None) -> dict[str, Any]:
    snaps = max(1, int(snaps))
    num_edges = int(events.num_edges)
    per_snap = max(1, (num_edges + snaps - 1) // snaps) if num_edges > 0 else 1
    dataset: list[dict[str, Any]] = []
    for snap_idx in range(snaps):
        start = snap_idx * per_snap
        end = min(num_edges, start + per_snap)
        if start >= end:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.stack([events.src[start:end], events.dst[start:end]], dim=0).long()
            edge_weight = events.weight[start:end].float()
        in_deg, out_deg = _compute_degrees(edge_index, edge_weight, events.num_nodes)
        labels = torch.log(in_deg + 1.0)
        if snap_idx == snaps - 1:
            labels = None
        dataset.append(
            {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "x": torch.stack([in_deg, out_deg], dim=1) if input_x is None else input_x[snap_idx],
                "y": labels if input_y is None else input_y[snap_idx],
            }
        )
    return {
        "num_nodes": int(events.num_nodes),
        "num_edges": int(events.num_edges),
        "num_snapshots": int(len(dataset)),
        "num_life_edges": int(sum(int(item["edge_weight"].numel()) for item in dataset)),
        "dataset": dataset,
        "source": events.source,
    }
