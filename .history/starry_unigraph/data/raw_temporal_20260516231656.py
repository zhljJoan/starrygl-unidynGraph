from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


NODE_FEATURE_FILE_CANDIDATES = (
    "node_temporal_features.csv",
    "node_features.csv",
)


@dataclass
class NodeTemporalFeatureTable:
    node_ids: torch.Tensor
    ts: torch.Tensor
    values: torch.Tensor
    feature_names: tuple[str, ...]
    source: str

    @property
    def dim(self) -> int:
        return int(self.values.size(-1)) if self.values.dim() == 2 else 0

    @property
    def size(self) -> int:
        return int(self.node_ids.numel())


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
    node_temporal_features: NodeTemporalFeatureTable | None = None


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


def _node_feature_path(dataset_dir: Path) -> Path | None:
    for name in NODE_FEATURE_FILE_CANDIDATES:
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def _feature_columns(frame: pd.DataFrame, reserved: set[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in reserved:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _load_node_temporal_features(dataset_dir: Path) -> NodeTemporalFeatureTable | None:
    feat_path = _node_feature_path(dataset_dir)
    if feat_path is None:
        return None

    frame = pd.read_csv(feat_path)
    if frame.empty:
        return NodeTemporalFeatureTable(
            node_ids=torch.empty(0, dtype=torch.long),
            ts=torch.empty(0, dtype=torch.float32),
            values=torch.empty(0, 0, dtype=torch.float32),
            feature_names=(),
            source=str(feat_path),
        )

    node_col = next((col for col in ("node_id", "node", "id") if col in frame.columns), None)
    if node_col is None:
        raise ValueError(f"Node feature file {feat_path} must contain one of: node_id, node, id")

    ts_col = next((col for col in ("time", "ts", "timestamp") if col in frame.columns), None)
    feature_cols = _feature_columns(frame, reserved={node_col, *( [ts_col] if ts_col else [] )})
    if not feature_cols:
        raise ValueError(f"Node feature file {feat_path} must contain at least one numeric feature column")

    node_ids = torch.as_tensor(frame[node_col].to_numpy(dtype="int64"), dtype=torch.long)
    if ts_col is None:
        ts = torch.zeros(node_ids.numel(), dtype=torch.float32)
    else:
        ts = torch.as_tensor(frame[ts_col].to_numpy(dtype="float32"), dtype=torch.float32)
    values = torch.as_tensor(frame[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32)

    order = torch.argsort(ts, stable=True)
    node_ids = node_ids[order]
    ts = ts[order]
    values = values[order]
    return NodeTemporalFeatureTable(
        node_ids=node_ids,
        ts=ts,
        values=values,
        feature_names=tuple(feature_cols),
        source=str(feat_path),
    )


def _remap_node_temporal_features(
    table: NodeTemporalFeatureTable | None,
    mapping: dict[int, int],
) -> NodeTemporalFeatureTable | None:
    if table is None:
        return None
    if table.size == 0:
        return table

    keep: list[int] = []
    remapped_ids: list[int] = []
    for index, node_id in enumerate(table.node_ids.tolist()):
        mapped = mapping.get(int(node_id))
        if mapped is None:
            continue
        keep.append(index)
        remapped_ids.append(mapped)

    if not keep:
        return NodeTemporalFeatureTable(
            node_ids=torch.empty(0, dtype=torch.long),
            ts=torch.empty(0, dtype=torch.float32),
            values=torch.empty(0, table.dim, dtype=torch.float32),
            feature_names=table.feature_names,
            source=table.source,
        )

    keep_t = torch.tensor(keep, dtype=torch.long)
    return NodeTemporalFeatureTable(
        node_ids=torch.tensor(remapped_ids, dtype=torch.long),
        ts=table.ts[keep_t].clone(),
        values=table.values[keep_t].clone(),
        feature_names=table.feature_names,
        source=table.source,
    )


def _node_feature_dim(table: NodeTemporalFeatureTable | None) -> int:
    return 0 if table is None else table.dim


def _node_feature_size(table: NodeTemporalFeatureTable | None) -> int:
    return 0 if table is None else table.size


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

    node_temporal_features = _load_node_temporal_features(dataset_dir)
    frame = pd.read_csv(edges_path)
    if frame.empty:
        return RawTemporalEvents(
            src=torch.empty(0, dtype=torch.long),
            dst=torch.empty(0, dtype=torch.long),
            ts=torch.empty(0, dtype=torch.float32),
            weight=torch.empty(0, dtype=torch.float32),
            edge_feat=torch.empty(0, 1, dtype=torch.float32),
            num_nodes=max(_node_feature_size(node_temporal_features), 0),
            num_edges=0,
            source=str(edges_path),
            node_temporal_features=node_temporal_features,
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
    if node_temporal_features is not None and node_temporal_features.size > 0:
        num_nodes = max(num_nodes, int(node_temporal_features.node_ids.max().item()) + 1)

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
        node_temporal_features=node_temporal_features,
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

    node_temporal_features = _load_node_temporal_features(dataset_dir)
    try:
        return _load_from_edge_file_numpy(edge_path, node_temporal_features)
    except Exception:
        return _load_from_edge_file_python(edge_path, node_temporal_features)


def _load_from_edge_file_numpy(
    edge_path: Path,
    node_temporal_features: NodeTemporalFeatureTable | None,
) -> RawTemporalEvents | None:
    data = _read_edge_file_mmap(edge_path)
    if data.size == 0:
        return None

    if data.shape[1] == 3:
        src_np = data[:, 0].astype(np.int64, copy=False)
        dst_np = data[:, 1].astype(np.int64, copy=False)
        ts_np = data[:, 2].astype(np.float32, copy=False)
        weight_np = np.ones(data.shape[0], dtype=np.float32)
    elif data.shape[1] == 4:
        src_np = data[:, 0].astype(np.int64, copy=False)
        dst_np = data[:, 1].astype(np.int64, copy=False)
        weight_np = data[:, 2].astype(np.float32, copy=False)
        ts_np = data[:, 3].astype(np.float32, copy=False)
    elif data.shape[1] >= 5:
        src_np = data[:, 1].astype(np.int64, copy=False)
        dst_np = data[:, 2].astype(np.int64, copy=False)
        ts_np = data[:, 3].astype(np.float32, copy=False)
        weight_np = np.ones(data.shape[0], dtype=np.float32)
    else:
        return None

    order = np.argsort(ts_np, kind="stable")
    src_np = np.ascontiguousarray(src_np[order])
    dst_np = np.ascontiguousarray(dst_np[order])
    ts_np = np.ascontiguousarray(ts_np[order])
    weight_np = np.ascontiguousarray(weight_np[order])

    src_t, dst_t, node_temporal_features, num_nodes = _remap_edges_numpy(
        src_np=src_np,
        dst_np=dst_np,
        node_temporal_features=node_temporal_features,
    )
    ts_t = torch.from_numpy(ts_np)
    weight_t = torch.from_numpy(weight_np)
    edge_feat = torch.ones(src_t.numel(), 1, dtype=torch.float32)
    return RawTemporalEvents(
        src=src_t,
        dst=dst_t,
        ts=ts_t,
        weight=weight_t,
        edge_feat=edge_feat,
        num_nodes=num_nodes,
        num_edges=int(src_t.numel()),
        source=str(edge_path),
        node_temporal_features=node_temporal_features,
    )


def _read_edge_file_mmap(edge_path: Path) -> np.ndarray:
    delimiter = "," if _looks_like_csv(edge_path) else r"\s+"
    comment = _detect_comment_char(edge_path)
    frame = pd.read_csv(
        edge_path,
        header=None,
        sep=delimiter,
        comment=comment,
        dtype=np.float64,
        engine="c",
        memory_map=True,
    )
    if frame.empty:
        return np.empty((0, 0), dtype=np.float64)
    # Some text files may have trailing separators; drop fully empty columns.
    frame = frame.dropna(axis=1, how="all")
    return frame.to_numpy(dtype=np.float64, copy=False)


def _looks_like_csv(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            return "," in line
    return False


def _detect_comment_char(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                return "#"
            if line.startswith("%"):
                return "%"
            return None
    return None


def _remap_edges_numpy(
    *,
    src_np: np.ndarray,
    dst_np: np.ndarray,
    node_temporal_features: NodeTemporalFeatureTable | None,
) -> tuple[torch.Tensor, torch.Tensor, NodeTemporalFeatureTable | None, int]:
    if node_temporal_features is None or node_temporal_features.size == 0:
        unique, inverse = np.unique(np.concatenate([src_np, dst_np]), return_inverse=True)
        edge_count = src_np.shape[0]
        src_t = torch.from_numpy(np.ascontiguousarray(inverse[:edge_count].astype(np.int64, copy=False)))
        dst_t = torch.from_numpy(np.ascontiguousarray(inverse[edge_count:].astype(np.int64, copy=False)))
        return src_t, dst_t, None, int(unique.size)

    feature_nodes = node_temporal_features.node_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    unique = np.unique(np.concatenate([src_np, dst_np, feature_nodes]))
    src_t = torch.from_numpy(np.searchsorted(unique, src_np).astype(np.int64, copy=False))
    dst_t = torch.from_numpy(np.searchsorted(unique, dst_np).astype(np.int64, copy=False))
    remapped_features = _remap_node_temporal_features_numpy(node_temporal_features, unique)
    return src_t.contiguous(), dst_t.contiguous(), remapped_features, int(unique.size)


def _remap_node_temporal_features_numpy(
    table: NodeTemporalFeatureTable,
    unique_nodes: np.ndarray,
) -> NodeTemporalFeatureTable:
    if table.size == 0:
        return table
    ids_np = table.node_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    pos = np.searchsorted(unique_nodes, ids_np)
    valid = (pos < unique_nodes.size) & (unique_nodes[pos.clip(max=max(0, unique_nodes.size - 1))] == ids_np)
    if not np.any(valid):
        return NodeTemporalFeatureTable(
            node_ids=torch.empty(0, dtype=torch.long),
            ts=torch.empty(0, dtype=torch.float32),
            values=torch.empty(0, table.dim, dtype=torch.float32),
            feature_names=table.feature_names,
            source=table.source,
        )
    keep_t = torch.from_numpy(np.nonzero(valid)[0].astype(np.int64, copy=False))
    remapped_ids = torch.from_numpy(pos[valid].astype(np.int64, copy=False))
    return NodeTemporalFeatureTable(
        node_ids=remapped_ids,
        ts=table.ts[keep_t].clone(),
        values=table.values[keep_t].clone(),
        feature_names=table.feature_names,
        source=table.source,
    )


def _load_from_edge_file_python(
    edge_path: Path,
    node_temporal_features: NodeTemporalFeatureTable | None,
) -> RawTemporalEvents | None:
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

    if node_temporal_features is not None and node_temporal_features.size > 0:
        unique = torch.unique(torch.cat([src_t, dst_t, node_temporal_features.node_ids], dim=0), sorted=True)
    else:
        unique = torch.unique(torch.cat([src_t, dst_t], dim=0), sorted=True)
    mapping = {int(node.item()): idx for idx, node in enumerate(unique)}
    src_t = torch.tensor([mapping[int(v.item())] for v in src_t], dtype=torch.long)
    dst_t = torch.tensor([mapping[int(v.item())] for v in dst_t], dtype=torch.long)
    node_temporal_features = _remap_node_temporal_features(node_temporal_features, mapping)
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
        node_temporal_features=node_temporal_features,
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


def _snapshot_ranges(num_edges: int, snaps: int) -> list[tuple[int, int]]:
    per_snap = max(1, (num_edges + snaps - 1) // snaps) if num_edges > 0 else 1
    ranges: list[tuple[int, int]] = []
    for snap_idx in range(snaps):
        start = snap_idx * per_snap
        end = min(num_edges, start + per_snap)
        ranges.append((start, end))
    return ranges


def _snapshot_end_timestamps(events: RawTemporalEvents, snaps: int) -> list[float]:
    if events.num_edges == 0:
        return [0.0 for _ in range(snaps)]
    ranges = _snapshot_ranges(int(events.num_edges), snaps)
    end_ts: list[float] = []
    last_ts = float(events.ts[-1].item()) if events.ts.numel() else 0.0
    prev_ts = float(events.ts[0].item()) if events.ts.numel() else 0.0
    for start, end in ranges:
        if start < end:
            prev_ts = float(events.ts[end - 1].item())
            end_ts.append(prev_ts)
        else:
            end_ts.append(prev_ts if end_ts else last_ts)
    return end_ts


def _build_snapshot_node_features(
    events: RawTemporalEvents,
    snaps: int,
) -> list[torch.Tensor] | None:
    table = events.node_temporal_features
    if table is None or table.size == 0 or table.dim == 0:
        return None

    end_ts = _snapshot_end_timestamps(events, snaps)
    order = torch.argsort(table.ts, stable=True)
    feat_node_ids = table.node_ids[order]
    feat_ts = table.ts[order]
    feat_values = table.values[order]

    current = torch.zeros(events.num_nodes, table.dim, dtype=torch.float32)
    cursor = 0
    snapshot_x: list[torch.Tensor] = []
    for snapshot_end_ts in end_ts:
        while cursor < feat_ts.numel() and float(feat_ts[cursor].item()) <= snapshot_end_ts:
            current[int(feat_node_ids[cursor].item())] = feat_values[cursor]
            cursor += 1
        snapshot_x.append(current.clone())
    return snapshot_x


def _get_masked_snapshots(self, data: np.ndarray):
    src, dst, wei, tss = self._get_uvwt(data)
    src_mapped, dst_mapped, num_nodes = self._get_mapped_uv(src, dst)
    uniq_tss = np.unique(tss)
    uniq_tss = np.sort(uniq_tss)[self.skiptime:]
    start_ts = uniq_tss[0]
    time_itv = (uniq_tss[-1] - uniq_tss[0]) / self.window
    edge_snapshots = [None for i in range(self.window - self.lags)]
    edge_weight_snapshots = [None for i in range(self.window - self.lags)]
    N = num_nodes
    for i in range(self.window - self.lags):
        mask = (tss >= start_ts + i * time_itv) & (tss < start_ts + (i + self.lags + 1) * time_itv)
        edge_snapshots[i] = np.array([src_mapped[mask], dst_mapped[mask]])
        edge_weight_snapshots[i] = np.array(wei[mask])
    
    edge_snapshots = [arr for arr in edge_snapshots if arr.size > 100] # filter out small snapshots
    edge_weight_snapshots = [arr for arr in edge_weight_snapshots if arr.size > 50]
    return edge_snapshots, edge_weight_snapshots, num_nodes

def _get_batch_snapshots(self, data: np.ndarray):
    if self.batch_size is None or self.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    src, dst, wei, tss = self._get_uvwt(data)
    src_mapped, dst_mapped, num_nodes = self._get_mapped_uv(src, dst)
    order = np.argsort(tss, kind="stable")
    src_mapped = src_mapped[order]
    dst_mapped = dst_mapped[order]
    wei = wei[order]
    tss = tss[order]
    uniq_tss = np.unique(tss)
    if self.skiptime > 0 and self.skiptime < uniq_tss.size:
        start_ts = uniq_tss[self.skiptime]
        keep = tss >= start_ts
        src_mapped = src_mapped[keep]
        dst_mapped = dst_mapped[keep]
        wei = wei[keep]
    edge_snapshots = []
    edge_weight_snapshots = []
    n = src_mapped.shape[0]
    for i in range(0, n, self.batch_size):
        j = min(i + self.batch_size, n)
        if j <= i:
            continue
        edge_snapshots.append(np.array([src_mapped[i:j], dst_mapped[i:j]]))
        edge_weight_snapshots.append(np.array(wei[i:j]))
    return edge_snapshots, edge_weight_snapshots, num_nodes


def _read_web_data(self, path):
        path = Path(path).expanduser().resolve()
        data = pd.read_csv(path, skiprows=self.skiprows, sep=self.sep).to_numpy()
        # Fallback: some sources are whitespace-delimited even when sep is configured differently.
        # If an entire line is parsed as one string column, retry with whitespace splitting.
        if data.ndim == 2 and data.shape[1] == 1 and data.size > 0:
            first = data[0, 0]
            if isinstance(first, str) and " " in first.strip():
                data = pd.read_csv(path, skiprows=self.skiprows, sep=r"\s+").to_numpy()

        num_edges = data.shape[0]
        if self.use_batch_split:
            if self.batch_size is None:
                raise ValueError("batch_size is required when use_batch_split=True")
            edge_snapshots, edge_weight_snapshots, num_nodes = self._get_batch_snapshots(data)
        else:
            edge_snapshots, edge_weight_snapshots, num_nodes = self._get_masked_snapshots(data)

        edge_snapshots = [torch.from_numpy(arr).type(torch.int32) for arr in edge_snapshots]
        edge_weight_snapshots = [torch.from_numpy(arr).type(torch.float32) for arr in edge_weight_snapshots]
        
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)

        return edge_snapshots, edge_weight_snapshots, num_nodes, num_edges
    
def build_snapshot_dataset_from_events(events: RawTemporalEvents, snaps: int, input_x = None, input_y = None,
                                       windows_slices=None, slice_config=None) -> dict[str, Any]:
    snaps = max(1, int(snaps))
    num_edges = int(events.num_edges)
    assert slice_config is not None, "slice_config is required for building snapshot dataset"
    if slice_config['use_batch_split'] is True:
        if windows_slices is None:
            if slice_config['batch_size'] is None:
                raise ValueError("batch_size is required when use_batch_split=True")
            windows_slices = _get_batch_slices(num_edges, slice_config['batch_size'])
    else:
        if windows_slices is None:
            if slice_config['batch_size'] is None:
                raise ValueError("batch_size is required for slicing when use_batch_split=False")
            windows_slices = _get_snapshots_slice(events, slice_config['windows_size'])
        else:
            windows_slices = _get_masked_slices(events, windows_slices)
        windows_slices = _snapshot_ranges(num_edges, snaps)
    dataset: list[dict[str, Any]] = []
    in_degrees: list[torch.Tensor] = []
    for snap_idx in range(snaps):
        start = snap_idx * per_snap if windows_slices is None else windows_slices[snap_idx]
        end = min(num_edges, start + per_snap) if windows_slices is None else windows_slices[snap_idx + 1]
        if start >= end:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.stack([events.src[start:end], events.dst[start:end]], dim=0).long()
            edge_weight = events.weight[start:end].float()
        in_deg, out_deg = _compute_degrees(edge_index, edge_weight, events.num_nodes)
        in_degrees.append(in_deg)
        if snapshot_features is None:
            node_x = torch.stack([in_deg, out_deg], dim=1)
        else:
            node_x = snapshot_features[snap_idx]
        dataset.append(
            {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "x": node_x,
                "y": None,
            }
        )
    for snap_idx, item in enumerate(dataset):
        if input_y is None:
            item["y"] = None if snap_idx == snaps - 1 else torch.log(in_degrees[snap_idx + 1] + 1.0)
        else:
            item["y"] = input_y[snap_idx]
    return {
        "num_nodes": int(events.num_nodes),
        "num_edges": int(events.num_edges),
        "num_snapshots": int(len(dataset)),
        "num_life_edges": int(sum(int(item["edge_weight"].numel()) for item in dataset)),
        "dataset": dataset,
        "source": events.source,
        "node_feat_dim": int(dataset[0]["x"].size(-1)) if dataset else 0,
        "node_feature_source": (
            "temporal_node_features"
            if events.node_temporal_features is not None and events.node_temporal_features.dim > 0 and snapshot_features is not None
            else "degree_fallback"
        ),
    }
