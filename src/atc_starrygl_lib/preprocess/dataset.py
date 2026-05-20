from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


DATASET_FORMAT = "atc_dataset_v1"


def build_dataset(
    *,
    data: Any,
    mode: str = "event",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int | None = None,
    num_windows: int | None = None,
    train_batch_size: int | None = None,
    train_num_windows: int | None = None,
    val_batch_size: int | None = None,
    val_num_windows: int | None = None,
    test_batch_size: int | None = None,
    test_num_windows: int | None = None,
    lags: int = 1,
) -> dict[str, Any]:
    graph = _load_graph_like(data)
    src = graph["src"].long().cpu().contiguous()
    dst = graph["dst"].long().cpu().contiguous()
    ts = graph["ts"].float().cpu().contiguous()
    edge_ids = graph.get("edge_ids", torch.arange(src.numel(), dtype=torch.long)).long().cpu().contiguous()
    num_edges = int(src.numel())
    order = torch.argsort(ts, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    ts = ts.index_select(0, order)
    edge_ids = edge_ids.index_select(0, order)

    edge_feat = _reorder_opt(graph.get("edge_feat"), order)
    edge_label = _reorder_opt(graph.get("edge_label"), order)
    edge_weight = _reorder_opt(graph.get("edge_weight"), order)

    split = _event_split(num_edges=num_edges, train_ratio=train_ratio, val_ratio=val_ratio)
    snapshot_ptr = graph.get("snapshot_ptr")
    if snapshot_ptr is not None:
        snapshot_ptr = snapshot_ptr.long().cpu().contiguous()

    split_time_ptr: dict[str, Tensor] | None = None
    if mode == "event":
        split_time_ptr = _event_split_time_ptr(
            split=split,
            batch_size=batch_size,
            num_windows=num_windows,
            train_batch_size=train_batch_size,
            train_num_windows=train_num_windows,
            val_batch_size=val_batch_size,
            val_num_windows=val_num_windows,
            test_batch_size=test_batch_size,
            test_num_windows=test_num_windows,
        )
        time_ptr_2 = torch.cat([split_time_ptr["train"], split_time_ptr["val"], split_time_ptr["test"]], dim=0)
    elif mode == "snapshot":
        if snapshot_ptr is None:
            raise ValueError("snapshot mode requires snapshot_ptr")
        time_ptr_2 = _snapshot_time_ptr(snapshot_ptr=snapshot_ptr, lags=lags)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    num_nodes = int(graph.get("num_nodes", torch.cat([src, dst]).max().item() + 1 if num_edges > 0 else 0))

    return {
        "format": DATASET_FORMAT,
        "src": src,
        "dst": dst,
        "ts": ts,
        "edge_ids": edge_ids,
        "num_nodes": num_nodes,
        "node_feat": _cpu_opt(graph.get("node_feat")),
        "edge_feat": edge_feat,
        "node_label": _cpu_opt(graph.get("node_label")),
        "edge_label": edge_label,
        "edge_weight": edge_weight,
        "time_ptr_2": time_ptr_2.long().cpu().contiguous(),
        "split_time_ptr": None if split_time_ptr is None else {k: v.long().cpu().contiguous() for k, v in split_time_ptr.items()},
        "split": split,
        "snapshot_ptr": snapshot_ptr,
    }


def _load_graph_like(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        if isinstance(data.get("snapshots"), list):
            return _from_snapshots(data["snapshots"], data)
        return _normalize_keys(data)
    path = Path(data)
    if path.suffix == ".pth":
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(".pth payload must be dict")
        if isinstance(obj.get("snapshots"), list):
            return _from_snapshots(obj["snapshots"], obj)
        return _normalize_keys(obj)
    if path.name == "edges.csv":
        rows = _read_csv_like(path, sep=",")
    elif path.suffix == ".edges":
        rows = _read_csv_like(path, sep=" ")
    elif path.name == "edges.txt":
        rows = _read_csv_like(path, sep=" ")
    else:
        raise ValueError(f"unsupported dataset source: {path}")
    return {
        "src": torch.tensor([r[0] for r in rows], dtype=torch.long),
        "dst": torch.tensor([r[1] for r in rows], dtype=torch.long),
        "ts": torch.tensor([r[2] for r in rows], dtype=torch.float32),
    }


def _read_csv_like(path: Path, sep: str) -> list[tuple[int, int, float]]:
    import pandas as pd

    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    s_col = cols.get("src") or cols.get("u")
    d_col = cols.get("dst") or cols.get("i")
    t_col = cols.get("ts") or cols.get("time")
    if s_col is None or d_col is None or t_col is None:
        raw = pd.read_csv(path, sep=sep, header=None)
        return [(int(a), int(b), float(c)) for a, b, c in raw.iloc[:, :3].itertuples(index=False, name=None)]
    return [(int(a), int(b), float(c)) for a, b, c in df[[s_col, d_col, t_col]].itertuples(index=False, name=None)]


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if "src" not in out and "u" in out:
        out["src"] = out["u"]
    if "dst" not in out and "i" in out:
        out["dst"] = out["i"]
    if "ts" not in out and "time" in out:
        out["ts"] = out["time"]
    return out


def _from_snapshots(snaps: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    src_parts, dst_parts, ts_parts = [], [], []
    ptr = [0]
    for sid, snap in enumerate(snaps):
        s = torch.as_tensor(snap["src"], dtype=torch.long)
        d = torch.as_tensor(snap["dst"], dtype=torch.long)
        t = torch.as_tensor(snap.get("ts", torch.full((int(s.numel()),), float(sid))), dtype=torch.float32)
        src_parts.append(s)
        dst_parts.append(d)
        ts_parts.append(t)
        ptr.append(ptr[-1] + int(s.numel()))
    out = {
        "src": torch.cat(src_parts, dim=0) if src_parts else torch.empty(0, dtype=torch.long),
        "dst": torch.cat(dst_parts, dim=0) if dst_parts else torch.empty(0, dtype=torch.long),
        "ts": torch.cat(ts_parts, dim=0) if ts_parts else torch.empty(0, dtype=torch.float32),
        "snapshot_ptr": torch.tensor(ptr, dtype=torch.long),
    }
    for k in ("num_nodes", "node_feat", "node_label", "edge_feat", "edge_label", "edge_weight"):
        if k in meta:
            out[k] = meta[k]
    return out


def _event_split(*, num_edges: int, train_ratio: float, val_ratio: float) -> Tensor:
    n_train = int(num_edges * float(train_ratio))
    n_val = int(num_edges * float(val_ratio))
    split = torch.full((num_edges,), 2, dtype=torch.uint8)
    split[:n_train] = 0
    split[n_train : n_train + n_val] = 1
    return split


def _event_time_ptr(*, num_edges: int, batch_size: int | None, num_windows: int | None) -> Tensor:
    if batch_size is None and num_windows is None:
        batch_size = max(1, num_edges)
    if batch_size is None:
        batch_size = max(1, int(math.ceil(num_edges / max(1, int(num_windows)))))
    ptr = []
    for begin in range(0, num_edges, int(batch_size)):
        end = min(num_edges, begin + int(batch_size))
        ptr.append([begin, end])
    return torch.tensor(ptr, dtype=torch.long) if ptr else torch.zeros((0, 2), dtype=torch.long)


def _event_split_time_ptr(
    *,
    split: Tensor,
    batch_size: int | None,
    num_windows: int | None,
    train_batch_size: int | None,
    train_num_windows: int | None,
    val_batch_size: int | None,
    val_num_windows: int | None,
    test_batch_size: int | None,
    test_num_windows: int | None,
) -> dict[str, Tensor]:
    split = split.to(torch.uint8).cpu()
    split_cfg = {
        0: (train_batch_size if train_batch_size is not None else batch_size, train_num_windows if train_num_windows is not None else num_windows),
        1: (val_batch_size if val_batch_size is not None else batch_size, val_num_windows if val_num_windows is not None else num_windows),
        2: (test_batch_size if test_batch_size is not None else batch_size, test_num_windows if test_num_windows is not None else num_windows),
    }
    out: dict[str, Tensor] = {}
    names = {0: "train", 1: "val", 2: "test"}
    for sid in (0, 1, 2):
        idx = (split == sid).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            out[names[sid]] = torch.zeros((0, 2), dtype=torch.long)
            continue
        bsz, nw = split_cfg[sid]
        local_ptr = _event_time_ptr(num_edges=int(idx.numel()), batch_size=bsz, num_windows=nw)
        start = int(idx[0])
        rows: list[list[int]] = []
        for begin, end in local_ptr.tolist():
            rows.append([start + int(begin), start + int(end)])
        out[names[sid]] = torch.tensor(rows, dtype=torch.long) if rows else torch.zeros((0, 2), dtype=torch.long)
    return out


def _snapshot_time_ptr(*, snapshot_ptr: Tensor, lags: int) -> Tensor:
    lags = max(1, int(lags))
    t = int(snapshot_ptr.numel()) - 1
    out = []
    for sid in range(t):
        begin_sid = max(0, sid - lags + 1)
        out.append([int(snapshot_ptr[begin_sid]), int(snapshot_ptr[sid + 1])])
    return torch.tensor(out, dtype=torch.long) if out else torch.zeros((0, 2), dtype=torch.long)


def _reorder_opt(x: Any, order: Tensor) -> Tensor | None:
    if x is None:
        return None
    t = torch.as_tensor(x)
    return t.index_select(0, order).cpu().contiguous()


def _cpu_opt(x: Any) -> Tensor | None:
    if x is None:
        return None
    return torch.as_tensor(x).cpu().contiguous()
