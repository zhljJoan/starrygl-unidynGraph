from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


STATIC_NODE_FEATURE_FILE = "node_features.pt"
EDGE_FEATURE_FILE = "edge_features.pt"
LABEL_FILE_CANDIDATES = ("node_labels.pt", "labels.pt", "labels.csv")


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
    node_feat: torch.Tensor | None = None
    node_label: torch.Tensor | None = None
    node_temporal_features: NodeTemporalFeatureTable | None = None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _as_2d_float_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.float().cpu()
    if tensor.dim() == 1:
        tensor = tensor.view(-1, 1)
    return tensor


def _load_static_node_features(dataset_dir: Path) -> torch.Tensor | None:
    feat_path = dataset_dir / STATIC_NODE_FEATURE_FILE
    if not feat_path.exists():
        return None

    return _as_2d_float_tensor(torch.load(feat_path, map_location="cpu"))


def _load_static_edge_features(dataset_dir: Path, num_edges: int) -> torch.Tensor:
    feat_path = dataset_dir / EDGE_FEATURE_FILE
    if not feat_path.exists():
        return torch.ones(num_edges, 1, dtype=torch.float32)

    edge_feat = _as_2d_float_tensor(torch.load(feat_path, map_location="cpu"))
    if edge_feat.size(0) != num_edges:
        raise ValueError(
            f"Edge feature file {feat_path} has {edge_feat.size(0)} rows, "
            f"but edges contain {num_edges} rows"
        )
    return edge_feat


def _load_static_node_labels(dataset_dir: Path) -> torch.Tensor | None:
    label_path = next((dataset_dir / name for name in LABEL_FILE_CANDIDATES if (dataset_dir / name).exists()), None)
    if label_path is None:
        return None

    if label_path.suffix == ".pt":
        labels = torch.load(label_path, map_location="cpu")
        return labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)

    frame = pd.read_csv(label_path)
    if frame.empty or "node" not in frame.columns or "label" not in frame.columns:
        return None
    if "time" in frame.columns:
        frame = frame.sort_values("time", kind="stable")

    node_ids = torch.as_tensor(frame["node"].to_numpy(dtype="int64"), dtype=torch.long)
    raw_labels = torch.as_tensor(frame["label"].to_numpy())
    num_nodes = int(node_ids.max().item()) + 1 if node_ids.numel() else 0
    labels = torch.zeros(num_nodes, dtype=raw_labels.dtype)
    if node_ids.numel() > 0:
        labels[node_ids] = raw_labels
    return labels


def _num_nodes_from_edges_and_features(
    src: torch.Tensor,
    dst: torch.Tensor,
    node_feat: torch.Tensor | None,
    node_label: torch.Tensor | None,
) -> int:
    num_nodes = int(max(int(src.max().item()), int(dst.max().item())) + 1) if src.numel() else 0
    if node_feat is not None:
        num_nodes = max(num_nodes, int(node_feat.size(0)))
    if node_label is not None:
        num_nodes = max(num_nodes, int(node_label.size(0)))
    return num_nodes


def _load_real_edges_csv(dataset_dir: Path) -> RawTemporalEvents | None:
    edges_path = dataset_dir / "edges.csv"
    if not edges_path.exists():
        return None

    frame = pd.read_csv(edges_path)
    if not {"src", "dst"}.issubset(frame.columns):
        raise ValueError(f"{edges_path} must contain src and dst columns")
    ts_col = next((col for col in ("time", "ts", "timestamp") if col in frame.columns), None)
    if ts_col is None:
        raise ValueError(f"{edges_path} must contain one of: time, ts, timestamp")

    src = torch.as_tensor(frame["src"].to_numpy(dtype="int64"), dtype=torch.long)
    dst = torch.as_tensor(frame["dst"].to_numpy(dtype="int64"), dtype=torch.long)
    ts = torch.as_tensor(frame[ts_col].to_numpy(dtype="float32"), dtype=torch.float32)
    weight_col = next((col for col in ("weight", "w") if col in frame.columns), None)
    if weight_col is None:
        weight = torch.ones(src.numel(), dtype=torch.float32)
    else:
        weight = torch.as_tensor(frame[weight_col].to_numpy(dtype="float32"), dtype=torch.float32)

    node_feat = _load_static_node_features(dataset_dir)
    node_label = _load_static_node_labels(dataset_dir)
    edge_feat = _load_static_edge_features(dataset_dir, int(src.numel()))
    return RawTemporalEvents(
        src=src,
        dst=dst,
        ts=ts,
        weight=weight,
        edge_feat=edge_feat,
        num_nodes=_num_nodes_from_edges_and_features(src, dst, node_feat, node_label),
        num_edges=int(src.numel()),
        source=str(edges_path),
        node_feat=node_feat,
        node_label=node_label,
    )


def _load_real_edges_file(dataset_dir: Path, dataset_name: str) -> RawTemporalEvents | None:
    candidates = (
        dataset_dir / f"{dataset_name}.edges",
        dataset_dir / "edges.txt",
    )
    edge_path = next((item for item in candidates if item.exists()), None)
    if edge_path is None:
        return None

    frame = pd.read_csv(
        edge_path,
        header=None,
        sep=r"\s+",
        comment="%",
        engine="c",
    )
    frame = frame.dropna(axis=1, how="all")
    if frame.empty:
        return None
    if frame.shape[1] == 3:
        src = torch.as_tensor(frame.iloc[:, 0].to_numpy(dtype="int64"), dtype=torch.long)
        dst = torch.as_tensor(frame.iloc[:, 1].to_numpy(dtype="int64"), dtype=torch.long)
        ts = torch.as_tensor(frame.iloc[:, 2].to_numpy(dtype="float32"), dtype=torch.float32)
        weight = torch.ones(src.numel(), dtype=torch.float32)
    elif frame.shape[1] >= 4:
        src = torch.as_tensor(frame.iloc[:, 0].to_numpy(dtype="int64"), dtype=torch.long)
        dst = torch.as_tensor(frame.iloc[:, 1].to_numpy(dtype="int64"), dtype=torch.long)
        weight = torch.as_tensor(frame.iloc[:, 2].to_numpy(dtype="float32"), dtype=torch.float32)
        ts = torch.as_tensor(frame.iloc[:, 3].to_numpy(dtype="float32"), dtype=torch.float32)
    else:
        raise ValueError(f"{edge_path} must contain either src dst ts or src dst weight ts")

    node_feat = _load_static_node_features(dataset_dir)
    node_label = _load_static_node_labels(dataset_dir)
    edge_feat = _load_static_edge_features(dataset_dir, int(src.numel()))
    return RawTemporalEvents(
        src=src,
        dst=dst,
        ts=ts,
        weight=weight,
        edge_feat=edge_feat,
        num_nodes=_num_nodes_from_edges_and_features(src, dst, node_feat, node_label),
        num_edges=int(src.numel()),
        source=str(edge_path),
        node_feat=node_feat,
        node_label=node_label,
    )


def _load_real_pth(root: Path, dataset_dir: Path, dataset_name: str) -> RawTemporalEvents | None:
    candidates = (
        root / f"{dataset_name}.pth",
        root / f"{dataset_name.lower()}.pth",
        dataset_dir / f"{dataset_name}.pth",
        dataset_dir / f"{dataset_name.lower()}.pth",
        dataset_dir / "data.pth",
    )
    pth_path = next((item for item in candidates if item.exists()), None)
    if pth_path is None:
        return None

    payload = torch.load(pth_path, map_location="cpu")
    dataset = payload.get("dataset", payload) if isinstance(payload, dict) else payload
    if not isinstance(dataset, dict) or "edge_index" not in dataset:
        raise ValueError(f"{pth_path} must contain a dict with edge_index")

    edge_index = dataset["edge_index"].long().cpu()
    if edge_index.dim() != 2 or edge_index.size(0) < 2:
        raise ValueError(f"{pth_path}: edge_index must have shape [2 or 3, num_edges]")

    src = edge_index[0].long().contiguous()
    dst = edge_index[1].long().contiguous()
    if edge_index.size(0) >= 3:
        ts = edge_index[2].float().contiguous()
    elif "ts" in dataset:
        ts = torch.as_tensor(dataset["ts"], dtype=torch.float32).cpu().contiguous()
    elif "timestamp" in dataset:
        ts = torch.as_tensor(dataset["timestamp"], dtype=torch.float32).cpu().contiguous()
    else:
        ts = torch.arange(src.numel(), dtype=torch.float32)
    if ts.numel() != src.numel():
        raise ValueError(f"{pth_path}: timestamp length {ts.numel()} != edge count {src.numel()}")

    weight = dataset.get("edge_weight", dataset.get("weight"))
    if weight is None:
        weight = torch.ones(src.numel(), dtype=torch.float32)
    else:
        weight = torch.as_tensor(weight, dtype=torch.float32).cpu().view(-1)
        if weight.numel() != src.numel():
            raise ValueError(f"{pth_path}: weight length {weight.numel()} != edge count {src.numel()}")

    node_feat = dataset.get("node_feat")
    if node_feat is not None:
        node_feat = _as_2d_float_tensor(node_feat)
    node_label = dataset.get("node_label", dataset.get("labels"))
    if node_label is not None:
        node_label = node_label if isinstance(node_label, torch.Tensor) else torch.as_tensor(node_label)
        node_label = node_label.cpu()

    edge_feat = dataset.get("edge_feat")
    if edge_feat is None:
        edge_feat = torch.ones(src.numel(), 1, dtype=torch.float32)
    else:
        edge_feat = _as_2d_float_tensor(edge_feat)
        if edge_feat.size(0) != src.numel():
            raise ValueError(f"{pth_path}: edge_feat rows {edge_feat.size(0)} != edge count {src.numel()}")

    num_nodes = int(payload.get("num_nodes", 0)) if isinstance(payload, dict) else 0
    num_nodes = max(num_nodes, _num_nodes_from_edges_and_features(src, dst, node_feat, node_label))
    return RawTemporalEvents(
        src=src,
        dst=dst,
        ts=ts,
        weight=weight,
        edge_feat=edge_feat,
        num_nodes=num_nodes,
        num_edges=int(src.numel()),
        source=str(pth_path),
        node_feat=node_feat,
        node_label=node_label,
    )


def _sort_raw_temporal_events_by_time(events: RawTemporalEvents) -> RawTemporalEvents:
    if events.ts.numel() <= 1:
        return events

    order = torch.argsort(events.ts, stable=True)
    edge_feat = events.edge_feat
    if edge_feat.dim() > 0 and edge_feat.size(0) == order.numel():
        edge_feat = edge_feat[order]

    return RawTemporalEvents(
        src=events.src[order],
        dst=events.dst[order],
        ts=events.ts[order],
        weight=events.weight[order],
        edge_feat=edge_feat,
        num_nodes=events.num_nodes,
        num_edges=events.num_edges,
        source=events.source,
        node_feat=events.node_feat,
        node_label=events.node_label,
        node_temporal_features=events.node_temporal_features,
    )


def load_raw_temporal_events(root: Path | str, dataset_name: str, config: dict[str, Any]) -> RawTemporalEvents:
    if str(config.get("data", {}).get("format", "auto")).lower() == "mock":
        return _sort_raw_temporal_events_by_time(_mock_events(dataset_name=dataset_name, config=config))
    root_path = Path(root).expanduser().resolve()
    dataset_dir = _resolve_dataset_dir(root_path, dataset_name)
    events = _load_real_pth(root_path, dataset_dir, dataset_name)
    if events is None:
        events = _load_real_edges_csv(dataset_dir)
    if events is None:
        events = _load_real_edges_file(dataset_dir, dataset_name)
    if events is None:
        raise FileNotFoundError(
            f"Could not find {dataset_name}.pth under {root_path}, "
            f"or edges.csv/{dataset_name}.edges under {dataset_dir}"
        )
    return _sort_raw_temporal_events_by_time(events)


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
    snaps: int | None = None,
    end_timestamps: list[float] | None = None,
) -> list[torch.Tensor] | None:
    table = events.node_temporal_features
    if table is None or table.size == 0 or table.dim == 0:
        return None

    if end_timestamps is None:
        if snaps is None:
            raise ValueError("snaps is required when end_timestamps is not provided")
        end_timestamps = _snapshot_end_timestamps(events, snaps)
    order = torch.argsort(table.ts, stable=True)
    feat_node_ids = table.node_ids[order]
    feat_ts = table.ts[order]
    feat_values = table.values[order]

    current = torch.zeros(events.num_nodes, table.dim, dtype=torch.float32)
    cursor = 0
    snapshot_x: list[torch.Tensor] = []
    for snapshot_end_ts in end_timestamps:
        while cursor < feat_ts.numel() and float(feat_ts[cursor].item()) <= snapshot_end_ts:
            current[int(feat_node_ids[cursor].item())] = feat_values[cursor]
            cursor += 1
        snapshot_x.append(current.clone())
    return snapshot_x

def _get_snapshots_slice(events: RawTemporalEvents, num_windows: int = 0, window_size: float=0, lags: int = 0, skip_time: int = 0) -> torch.Tensor:
    assert num_windows > 0 or window_size > 0, "Either num_windows or window_size must be positive"
    if events.ts.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    tss = events.ts 
    uniq_tss = torch.unique(tss)
    if skip_time > 0 and skip_time < uniq_tss.size(0):
        uniq_tss = uniq_tss[skip_time:]
    start_ts = uniq_tss[0]
    time_itv = (uniq_tss[-1] - uniq_tss[0]) / num_windows if num_windows > 0 else window_size
    num_windows = num_windows if num_windows > 0 else max(1, int((uniq_tss[-1] - uniq_tss[0]) / window_size))
    start = []
    end = []
    for i in range(num_windows - lags):
        window_start = start_ts + i * time_itv
        window_end = start_ts + (i + lags + 1) * time_itv
        if i == num_windows - lags - 1:
            mask = ((tss >= window_start) & (tss <= window_end)).nonzero(as_tuple=False)
        else:
            mask = ((tss >= window_start) & (tss < window_end)).nonzero(as_tuple=False)
        mask = mask if mask.numel() > 0 else None
        start.append(mask.min().item() if mask is not None else 0)
        end.append(mask.max().item() + 1 if mask is not None else 0)
    return torch.tensor(start), torch.tensor(end)
            
def _get_masked_slices(events: RawTemporalEvents, windows_slices: list[tuple[int, int]]) -> torch.Tensor:
    start = []
    end = []
    for window_start, window_end in windows_slices:
        window_start = int(window_start)
        window_end = int(window_end)
        if window_start >= window_end:
            start.append(0)
            end.append(0)
            continue
        mask = ((events.ts >= events.ts[window_start]) & (events.ts < events.ts[window_end - 1])).nonzero(as_tuple=False) if (events.ts >= events.ts[window_start]).any() else None
        start.append(mask.min().item() if mask is not None else 0)
        end.append(mask.max().item() + 1 if mask is not None else 0)
    return torch.tensor(start), torch.tensor(end)        

def _get_batch_slices(events: RawTemporalEvents, num_edges:int,  batch_size: int, skip_time: int = 0    ) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start_pos = 0
    tss = events.ts
    uniq_tss = torch.unique(tss)
    if skip_time > 0 and skip_time < uniq_tss.size(0):
        start_pos = (tss >= uniq_tss[skip_time]).nonzero(as_tuple=False).min().item()
    windows_slices = torch.arange(start_pos, num_edges, batch_size, dtype=torch.long)
    if windows_slices.numel() == 0 or windows_slices[-1].item() != num_edges:
        windows_slices = torch.cat([
            windows_slices,
            torch.tensor([num_edges], dtype=torch.long),
        ])
    return windows_slices


def _normalize_windows_slices(windows_slices: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(windows_slices, tuple) and len(windows_slices) == 2:
        starts = torch.as_tensor(windows_slices[0], dtype=torch.long)
        ends = torch.as_tensor(windows_slices[1], dtype=torch.long)
    else:
        slices = torch.as_tensor(windows_slices, dtype=torch.long)
        if slices.dim() == 1:
            if slices.numel() < 2:
                raise ValueError("1-D windows_slices must contain at least two boundaries")
            starts = slices[:-1]
            ends = slices[1:]
        elif slices.dim() == 2 and slices.size(1) == 2:
            starts = slices[:, 0]
            ends = slices[:, 1]
        else:
            raise ValueError("windows_slices must be boundaries [N+1], pairs [N,2], or (starts, ends)")
    if starts.numel() != ends.numel():
        raise ValueError("windows_slices starts and ends must have the same length")
    return starts.long(), ends.long()


def _build_windows_slices(
    events: RawTemporalEvents,
    *,
    snaps: int | None,
    windows_slices: Any = None,
    slice_config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_edges = int(events.num_edges)
    if windows_slices is not None:
        if slice_config is not None and bool(slice_config.get("use_masked_slices", slice_config.get("masked_slices", False))):
            return _get_masked_slices(events, windows_slices)
        return _normalize_windows_slices(windows_slices)

    cfg = slice_config or {}
    if not cfg and snaps is not None:
        cfg = {"num_windows": int(snaps)}
    use_batch_split = bool(cfg.get("use_batch_split", False))
    skip_time = int(cfg.get("skip_time", 0))
    if use_batch_split:
        batch_size = cfg.get("batch_size")
        if batch_size is None:
            raise ValueError("slice_config.batch_size is required when use_batch_split=True")
        boundaries = _get_batch_slices(events, num_edges, int(batch_size), skip_time=skip_time)
        return _normalize_windows_slices(boundaries)

    num_windows = int(cfg.get("num_windows", cfg.get("windows", snaps or 0)))
    window_size = float(cfg.get("window_size", cfg.get("windows_size", 0.0)) or 0.0)
    lags = int(cfg.get("lags", 0))
    if num_windows > 0 or window_size > 0:
        return _get_snapshots_slice(
            events,
            num_windows=num_windows,
            window_size=window_size,
            lags=lags,
            skip_time=skip_time,
        )

    ranges = _snapshot_ranges(num_edges, max(1, int(snaps or 1)))
    return _normalize_windows_slices(torch.tensor(ranges, dtype=torch.long))


def _snapshot_end_timestamps_from_slices(events: RawTemporalEvents, starts: torch.Tensor, ends: torch.Tensor) -> list[float]:
    if events.ts.numel() == 0:
        return [0.0 for _ in range(int(starts.numel()))]
    result: list[float] = []
    prev_ts = float(events.ts[0].item())
    for start_t, end_t in zip(starts.tolist(), ends.tolist()):
        start = int(start_t)
        end = int(end_t)
        if start < end:
            prev_ts = float(events.ts[end - 1].item())
        result.append(prev_ts)
    return result


def _cfg_bool(config: dict[str, Any] | None, path: str, default: bool) -> bool:
    cur: Any = config or {}
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return bool(cur)


def _cfg_dict(config: dict[str, Any] | None, path: str) -> dict[str, Any] | None:
    cur: Any = config or {}
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return dict(cur) if isinstance(cur, dict) else None


def _empty_snapshot_dataset(events: RawTemporalEvents, source: str = "disabled") -> dict[str, Any]:
    return {
        "num_nodes": int(events.num_nodes),
        "num_edges": int(events.num_edges),
        "num_snapshots": 0,
        "num_life_edges": 0,
        "dataset": [],
        "source": events.source,
        "node_feat_dim": 0,
        "label_dim": 0,
        "node_feature_source": source,
    }


def build_snapshot_dataset_from_events(events: RawTemporalEvents, snaps: int | None = None, input_x = None, input_y = None,
                                       windows_slices=None, slice_config=None, config: dict[str, Any] | None = None,
                                       copy_features: bool | None = None) -> dict[str, Any]:
    if not _cfg_bool(config, "data.build_snapshot_dataset", True):
        return _empty_snapshot_dataset(events)
    if slice_config is None:
        slice_config = _cfg_dict(config, "data.slice_config")
    if copy_features is None:
        copy_features = _cfg_bool(config, "data.copy_snapshot_features", True)
    num_edges = int(events.num_edges)
    starts, ends = _build_windows_slices(
        events,
        snaps=snaps,
        windows_slices=windows_slices,
        slice_config=slice_config,
    )
    snaps = int(starts.numel())
    build_features = bool(copy_features) or input_x is not None or input_y is not None
    if input_x is not None:
        if len(input_x) < snaps:
            raise ValueError(f"input_x must contain at least {snaps} snapshots, got {len(input_x)}")
        snapshot_features = input_x
        node_feature_source = "input_x"
    elif events.node_feat is not None and build_features:
        snapshot_features = events.node_feat
        node_feature_source = "static_node_feat"
    elif not build_features:
        snapshot_features = None
        node_feature_source = "not_copied"
    else:
        snapshot_features = None
        node_feature_source = "degree_fallback"
    if input_y is not None and len(input_y) < snaps:
        raise ValueError(f"input_y must contain at least {snaps} snapshots, got {len(input_y)}")
    dataset: list[dict[str, Any]] = []
    in_degrees: list[torch.Tensor] = []
    for snap_idx in range(snaps):
        start = int(starts[snap_idx].item())
        end = int(ends[snap_idx].item())
        if start >= end:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.stack([events.src[start:end], events.dst[start:end]], dim=0).long()
            edge_weight = events.weight[start:end].float()
        in_deg, out_deg = _compute_degrees(edge_index, edge_weight, events.num_nodes)
        in_degrees.append(in_deg)
        item: dict[str, Any] = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }
        if build_features:
            if snapshot_features is None:
                item["x"] = torch.stack([in_deg, out_deg], dim=1)
            else:
                item["x"] = snapshot_features[snap_idx] if isinstance(snapshot_features, list) else snapshot_features
            item["y"] = None
        dataset.append(item)
    for snap_idx, item in enumerate(dataset):
        if not build_features:
            continue
        if input_y is not None:
            item["y"] = input_y[snap_idx]
        elif events.node_label is not None:
            item["y"] = events.node_label
        else:
            item["y"] = None if snap_idx == snaps - 1 else torch.log(in_degrees[snap_idx + 1] + 1.0)
    label_tensor = input_y[0] if input_y is not None and len(input_y) > 0 else events.node_label
    return {
        "num_nodes": int(events.num_nodes),
        "num_edges": int(events.num_edges),
        "num_snapshots": int(len(dataset)),
        "num_life_edges": int(sum(int(item["edge_weight"].numel()) for item in dataset)),
        "dataset": dataset,
        "source": events.source,
        "node_feat_dim": int(dataset[0]["x"].size(-1)) if dataset and "x" in dataset[0] else 0,
        "label_dim": int(label_tensor.size(-1)) if build_features and isinstance(label_tensor, torch.Tensor) and label_tensor.dim() > 1 else (1 if build_features and label_tensor is not None else 0),
        "node_feature_source": node_feature_source,
    }
