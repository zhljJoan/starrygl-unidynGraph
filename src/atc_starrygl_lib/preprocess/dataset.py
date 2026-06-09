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
    split_mode: str = "fixed",
    adaptive_split: bool = False,
    adaptive_split_graph_feature: float = 0.5,
    adaptive_split_alpha: float = 1.0,
    adaptive_split_beta: float = 0.5,
    adaptive_split_aggl: float | None = None,
    adaptive_split_enable_drop: bool = False,
    adaptive_split_drop_rate: float = 0.8,
    adaptive_split_window_size: int = 1000,
    adaptive_split_min_batch_size: int | None = None,
    adaptive_split_max_batch_size: int | None = None,
    adaptive_split_coherence_chunks: int = 0,
    adaptive_split_max_chunk_entropy_ratio: float | None = None,
    adaptive_split_fallback: bool = True,
    lags: int = 1,
    random_node_feat_dim: int = 0,
    random_node_feat_seed: int = 0,
    random_edge_feat_dim: int = 0,
    random_edge_feat_seed: int = 0,
    random_feature_seed_mode: str = "independent",
) -> dict[str, Any]:
    graph = _load_graph_like(data)
    if graph.get("split") is None:
        roll = graph.get("ext_roll", graph.get("int_roll"))
        if roll is not None:
            graph["split"] = torch.as_tensor(roll, dtype=torch.long).clamp(0, 2).to(torch.uint8)
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

    split = _reorder_split(
        graph,
        order=order,
        num_edges=num_edges,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    snapshot_ptr = graph.get("snapshot_ptr")
    if snapshot_ptr is not None:
        snapshot_ptr = snapshot_ptr.long().cpu().contiguous()

    split_time_ptr: dict[str, Tensor] | None = None
    if mode == "event":
        split_time_ptr = _event_split_time_ptr(
            src=src,
            dst=dst,
            ts=ts,
            split=split,
            batch_size=batch_size,
            num_windows=num_windows,
            train_batch_size=train_batch_size,
            train_num_windows=train_num_windows,
            val_batch_size=val_batch_size,
            val_num_windows=val_num_windows,
            test_batch_size=test_batch_size,
            test_num_windows=test_num_windows,
            split_mode=split_mode,
            adaptive_split=adaptive_split,
            adaptive_split_graph_feature=adaptive_split_graph_feature,
            adaptive_split_alpha=adaptive_split_alpha,
            adaptive_split_beta=adaptive_split_beta,
            adaptive_split_aggl=adaptive_split_aggl,
            adaptive_split_enable_drop=adaptive_split_enable_drop,
            adaptive_split_drop_rate=adaptive_split_drop_rate,
            adaptive_split_window_size=adaptive_split_window_size,
            adaptive_split_min_batch_size=adaptive_split_min_batch_size,
            adaptive_split_max_batch_size=adaptive_split_max_batch_size,
            adaptive_split_coherence_chunks=adaptive_split_coherence_chunks,
            adaptive_split_max_chunk_entropy_ratio=adaptive_split_max_chunk_entropy_ratio,
            adaptive_split_fallback=adaptive_split_fallback,
        )
        time_ptr_2 = torch.cat([split_time_ptr["train"], split_time_ptr["val"], split_time_ptr["test"]], dim=0)
    elif mode == "snapshot":
        if snapshot_ptr is None:
            raise ValueError("snapshot mode requires snapshot_ptr")
        time_ptr_2 = _snapshot_time_ptr(snapshot_ptr=snapshot_ptr, lags=lags)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    num_nodes = int(graph.get("num_nodes", torch.cat([src, dst]).max().item() + 1 if num_edges > 0 else 0))
    node_feat = _cpu_opt(graph.get("node_feat"))
    seed_mode = str(random_feature_seed_mode).strip().lower()
    shared_seed_mode = seed_mode in {"shared", "sequential", "shared_sequential", "bts"}
    if shared_seed_mode:
        gen = torch.Generator()
        gen.manual_seed(int(random_node_feat_seed if int(random_node_feat_dim) > 0 else random_edge_feat_seed))
        if node_feat is None and int(random_node_feat_dim) > 0:
            node_feat = torch.randn((num_nodes, int(random_node_feat_dim)), generator=gen, dtype=torch.float32)
        if edge_feat is None and int(random_edge_feat_dim) > 0:
            edge_feat = torch.randn((num_edges, int(random_edge_feat_dim)), generator=gen, dtype=torch.float32)
    else:
        if node_feat is None and int(random_node_feat_dim) > 0:
            gen = torch.Generator()
            gen.manual_seed(int(random_node_feat_seed))
            node_feat = torch.randn((num_nodes, int(random_node_feat_dim)), generator=gen, dtype=torch.float32)
        if edge_feat is None and int(random_edge_feat_dim) > 0:
            gen = torch.Generator()
            gen.manual_seed(int(random_edge_feat_seed))
            edge_feat = torch.randn((num_edges, int(random_edge_feat_dim)), generator=gen, dtype=torch.float32)

    return {
        "format": DATASET_FORMAT,
        "src": src,
        "dst": dst,
        "ts": ts,
        "edge_ids": edge_ids,
        "num_nodes": num_nodes,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "node_feat": node_feat,
        "node_feat_time_varying": bool(graph.get("node_feat_time_varying", False)),
        "node_feat_source": graph.get("node_feat_source"),
        "edge_feat": edge_feat,
        "node_label": _cpu_opt(graph.get("node_label")),
        "node_label_time_varying": bool(graph.get("node_label_time_varying", False)),
        "node_label_source": graph.get("node_label_source"),
        "node_label_nodes": _cpu_opt(graph.get("node_label_nodes")),
        "node_label_ts": _cpu_opt(graph.get("node_label_ts")),
        "node_label_split": None if graph.get("node_label_split") is None else torch.as_tensor(graph["node_label_split"], dtype=torch.uint8).cpu().contiguous(),
        "edge_label": edge_label,
        "edge_weight": edge_weight,
        "time_ptr_2": time_ptr_2.long().cpu().contiguous(),
        "split_time_ptr": None if split_time_ptr is None else {k: v.long().cpu().contiguous() for k, v in split_time_ptr.items()},
        "split": split,
        "snapshot_ptr": snapshot_ptr,
    }


def _load_graph_like(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        if isinstance(data.get("dataset"), list):
            return _from_flare_snapshots(data["dataset"], data)
        if isinstance(data.get("snapshots"), list):
            return _from_snapshots(data["snapshots"], data)
        return _normalize_keys(data)
    path = Path(str(data)).expanduser()
    if path.is_dir():
        return _load_dataset_dir(path)
    if path.name == "edges.csv":
        return _load_dataset_dir(path.parent)
    if path.suffix == ".pth":
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(".pth payload must be dict")
        if isinstance(obj.get("dataset"), list):
            return _from_flare_snapshots(obj["dataset"], obj)
        if isinstance(obj.get("snapshots"), list):
            return _from_snapshots(obj["snapshots"], obj)
        return _normalize_keys(obj)
    if path.suffix == ".csv":
        rows = _read_csv_like(path, sep=",")
    elif path.suffix == ".edges":
        rows = _read_csv_like(path, sep=r"\s+")
    elif path.name == "edges.txt":
        rows = _read_csv_like(path, sep=r"\s+")
    else:
        raise ValueError(f"unsupported dataset source: {path}")
    return rows


def _load_dataset_dir(path: Path) -> dict[str, Any]:
    edge_path = path / "edges.csv"
    if not edge_path.exists():
        matches = sorted(path.glob("*.edges"))
        if not matches:
            raise ValueError(f"dataset directory does not contain edges.csv or *.edges: {path}")
        edge_path = matches[0]
    graph = _read_csv_like(edge_path, sep="," if edge_path.suffix == ".csv" else r"\s+")
    node_feat = _load_first_tensor(path, ("node_features.pt", "node_feat.pt", "nodes.pt"))
    edge_feat = _load_first_tensor(path, ("edge_features.pt", "edge_feat.pt", "edge_features_e0.pt"))
    if node_feat is not None:
        graph["node_feat"] = _feature_tensor(node_feat)
    if edge_feat is not None:
        graph["edge_feat"] = _align_edge_tensor(_feature_tensor(edge_feat), int(graph["src"].numel()))
    labels = path / "labels.csv"
    if labels.exists():
        graph.update(_load_node_labels(labels))
    return graph


def _load_first_tensor(path: Path, names: tuple[str, ...]) -> Tensor | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            value = torch.load(candidate, map_location="cpu", weights_only=False)
            if isinstance(value, dict):
                for key in ("feat", "features", "data", "x"):
                    if key in value:
                        value = value[key]
                        break
            return torch.as_tensor(value)
    return None


def _feature_tensor(value: Tensor) -> Tensor:
    tensor = torch.as_tensor(value).cpu().contiguous()
    if tensor.dtype == torch.bool or not tensor.dtype.is_floating_point:
        tensor = tensor.float()
    return tensor


def _align_edge_tensor(value: Tensor, num_edges: int) -> Tensor:
    if int(value.size(0)) == int(num_edges) + 1:
        value = value[1:]
    if int(value.size(0)) > int(num_edges):
        value = value[:num_edges]
    if int(value.size(0)) < int(num_edges):
        pad = torch.zeros((int(num_edges) - int(value.size(0)), *value.shape[1:]), dtype=value.dtype)
        value = torch.cat([value, pad], dim=0)
    return value.contiguous()


def _load_node_labels(path: Path) -> dict[str, Tensor]:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {str(c).lower(): c for c in df.columns}
    node_col = cols.get("node") or cols.get("node_id") or cols.get("nid")
    label_col = cols.get("label") or cols.get("y")
    if node_col is None or label_col is None:
        return {}
    out: dict[str, Tensor] = {
        "node_label_nodes": torch.as_tensor(df[node_col].to_numpy(), dtype=torch.long),
        "node_label": torch.as_tensor(df[label_col].to_numpy()),
    }
    time_col = cols.get("time") or cols.get("ts")
    if time_col is not None:
        out["node_label_ts"] = torch.as_tensor(df[time_col].to_numpy(), dtype=torch.float32)
    split_col = cols.get("int_roll")
    if split_col is not None:
        out["node_label_split"] = torch.as_tensor(df[split_col].to_numpy(), dtype=torch.uint8).clamp(0, 2)
    return out


def _read_csv_like(path: Path, sep: str) -> dict[str, Tensor]:
    import pandas as pd

    df = pd.read_csv(path, sep=sep, comment="%", engine="python")
    cols = {str(c).lower(): c for c in df.columns}
    s_col = cols.get("src") or cols.get("u")
    d_col = cols.get("dst") or cols.get("i")
    t_col = cols.get("ts") or cols.get("time")
    label_col = cols.get("label") or cols.get("weight") or cols.get("rating")
    if s_col is None or d_col is None or t_col is None:
        raw = pd.read_csv(path, sep=sep, comment="%", header=None, engine="python")
        graph = {
            "src": torch.as_tensor(raw.iloc[:, 0].to_numpy(), dtype=torch.long),
            "dst": torch.as_tensor(raw.iloc[:, 1].to_numpy(), dtype=torch.long),
            "ts": torch.as_tensor(raw.iloc[:, -1].to_numpy(), dtype=torch.float32),
        }
        if raw.shape[1] >= 4:
            label = torch.as_tensor(raw.iloc[:, -2].to_numpy(), dtype=torch.float32)
            graph["edge_weight"] = label
            graph["edge_label"] = label
        return graph
    graph = {
        "src": torch.as_tensor(df[s_col].to_numpy(), dtype=torch.long),
        "dst": torch.as_tensor(df[d_col].to_numpy(), dtype=torch.long),
        "ts": torch.as_tensor(df[t_col].to_numpy(), dtype=torch.float32),
    }
    if label_col is not None:
        label = torch.as_tensor(df[label_col].to_numpy(), dtype=torch.float32)
        graph["edge_weight"] = label
        graph["edge_label"] = label
    for name in ("ext_roll", "int_roll"):
        col = cols.get(name)
        if col is not None:
            graph[name] = torch.as_tensor(df[col].to_numpy(), dtype=torch.long)
    return graph


def _normalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if isinstance(out.get("dataset"), dict):
        nested = dict(out["dataset"])
        for key in ("num_nodes", "num_edges", "num_snapshots"):
            if key in out and key not in nested:
                nested[key] = out[key]
        out = nested
    edge_index = out.get("edge_index")
    if edge_index is not None and ("src" not in out or "dst" not in out):
        edges = torch.as_tensor(edge_index)
        if edges.dim() != 2:
            raise ValueError("edge_index must be a rank-2 tensor")
        if int(edges.size(0)) < 2 and int(edges.size(1)) >= 2:
            edges = edges.t()
        out["src"] = edges[0].long()
        out["dst"] = edges[1].long()
        if "ts" not in out and int(edges.size(0)) >= 3:
            out["ts"] = edges[2].float()
    if "src" not in out and "u" in out:
        out["src"] = out["u"]
    if "dst" not in out and "i" in out:
        out["dst"] = out["i"]
    if "ts" not in out and "time" in out:
        out["ts"] = out["time"]
    if "ts" not in out and "src" in out:
        out["ts"] = torch.arange(int(torch.as_tensor(out["src"]).numel()), dtype=torch.float32)
    if "node_feat" not in out and "x" in out:
        out["node_feat"] = out["x"]
    if "edge_feat" not in out and "efeat" in out:
        out["edge_feat"] = out["efeat"]
    if "split" not in out:
        roll = out.get("ext_roll", out.get("int_roll"))
        if roll is not None:
            roll_t = torch.as_tensor(roll, dtype=torch.long).clamp(0, 2)
            out["split"] = roll_t.to(torch.uint8)
    return out


def _reorder_split(
    graph: dict[str, Any],
    *,
    order: Tensor,
    num_edges: int,
    train_ratio: float,
    val_ratio: float,
) -> Tensor:
    if graph.get("split") is not None:
        return torch.as_tensor(graph["split"], dtype=torch.uint8).cpu().index_select(0, order).contiguous()
    masks = []
    for name in ("train_mask", "val_mask", "test_mask"):
        value = graph.get(name)
        if value is None:
            masks = []
            break
        masks.append(torch.as_tensor(value, dtype=torch.bool).cpu().index_select(0, order))
    if masks:
        split = torch.full((int(num_edges),), 2, dtype=torch.uint8)
        split[masks[0]] = 0
        split[masks[1]] = 1
        split[masks[2]] = 2
        return split
    return _event_split(num_edges=num_edges, train_ratio=train_ratio, val_ratio=val_ratio)


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
    for k in (
        "num_nodes",
        "node_feat",
        "node_feat_time_varying",
        "node_feat_source",
        "node_label",
        "node_label_time_varying",
        "node_label_source",
        "node_label_nodes",
        "node_label_ts",
        "node_label_split",
        "edge_feat",
        "edge_label",
        "edge_weight",
    ):
        if k in meta:
            out[k] = meta[k]
    return out


def _from_flare_snapshots(snaps: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    node_feat_parts: list[Tensor] = []
    node_label_parts: list[Tensor] = []
    all_have_feat = True
    all_have_label = True
    usable_snaps = [snap for snap in snaps if snap.get("y", snap.get("node_label")) is not None]
    if not usable_snaps:
        usable_snaps = snaps
    for sid, snap in enumerate(usable_snaps):
        edge_index = torch.as_tensor(snap["edge_index"])
        if edge_index.dim() != 2:
            raise ValueError("snapshot edge_index must be rank-2")
        if int(edge_index.size(0)) < 2 and int(edge_index.size(1)) >= 2:
            edge_index = edge_index.t()
        num_edges = int(edge_index.size(1))
        normalized.append({
            "src": edge_index[0].long(),
            "dst": edge_index[1].long(),
            "ts": torch.full((num_edges,), float(sid), dtype=torch.float32),
        })
        feat = snap.get("node_feat", snap.get("x"))
        label = snap.get("node_label", snap.get("y"))
        if feat is None:
            all_have_feat = False
        elif all_have_feat:
            node_feat_parts.append(_feature_tensor(torch.as_tensor(feat)))
        if label is None:
            all_have_label = False
        elif all_have_label:
            node_label_parts.append(torch.as_tensor(label).cpu().contiguous())

    out = _from_snapshots(normalized, meta)
    edge_weights = [torch.as_tensor(snap["edge_weight"]).cpu().contiguous() for snap in usable_snaps if snap.get("edge_weight") is not None]
    if len(edge_weights) == len(usable_snaps):
        out["edge_weight"] = torch.cat(edge_weights, dim=0) if edge_weights else torch.empty(0, dtype=torch.float32)
        out["edge_label"] = out["edge_weight"]
    if all_have_feat and len(node_feat_parts) == len(usable_snaps) and _same_shape(node_feat_parts):
        out["node_feat"] = torch.stack(node_feat_parts, dim=0).contiguous()
        out["node_feat_time_varying"] = True
        out["node_feat_source"] = meta.get("node_feat_source", meta.get("x_source", "snapshot_x"))
    if all_have_label and len(node_label_parts) == len(usable_snaps) and _same_shape(node_label_parts):
        out["node_label"] = torch.stack(node_label_parts, dim=0).contiguous()
        out["node_label_time_varying"] = True
        out["node_label_source"] = meta.get("node_label_source", meta.get("y_source", "snapshot_y"))
    if "num_nodes" not in out and meta.get("num_nodes") is not None:
        out["num_nodes"] = int(meta["num_nodes"])
    return out


def _same_shape(items: list[Tensor]) -> bool:
    if not items:
        return False
    shape = tuple(items[0].shape)
    return all(tuple(item.shape) == shape for item in items)


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
    src: Tensor,
    dst: Tensor,
    ts: Tensor,
    split: Tensor,
    batch_size: int | None,
    num_windows: int | None,
    train_batch_size: int | None,
    train_num_windows: int | None,
    val_batch_size: int | None,
    val_num_windows: int | None,
    test_batch_size: int | None,
    test_num_windows: int | None,
    split_mode: str = "fixed",
    adaptive_split: bool = False,
    adaptive_split_graph_feature: float = 0.5,
    adaptive_split_alpha: float = 1.0,
    adaptive_split_beta: float = 0.5,
    adaptive_split_aggl: float | None = None,
    adaptive_split_enable_drop: bool = False,
    adaptive_split_drop_rate: float = 0.8,
    adaptive_split_window_size: int = 1000,
    adaptive_split_min_batch_size: int | None = None,
    adaptive_split_max_batch_size: int | None = None,
    adaptive_split_coherence_chunks: int = 0,
    adaptive_split_max_chunk_entropy_ratio: float | None = None,
    adaptive_split_fallback: bool = True,
) -> dict[str, Tensor]:
    split = split.to(torch.uint8).cpu()
    src = src.long().cpu().contiguous()
    dst = dst.long().cpu().contiguous()
    ts = ts.cpu().contiguous()
    use_adaptive = bool(adaptive_split) or str(split_mode).strip().lower() in {"adaptive", "adaptive_split", "auto"}
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
        if use_adaptive:
            local_ptr = _event_adaptive_time_ptr(
                src=src.index_select(0, idx),
                dst=dst.index_select(0, idx),
                ts=ts.index_select(0, idx),
                batch_size=bsz,
                num_windows=nw,
                graph_feature=float(adaptive_split_graph_feature),
                alpha=float(adaptive_split_alpha),
                beta=float(adaptive_split_beta),
                aggl=adaptive_split_aggl,
                enable_drop=bool(adaptive_split_enable_drop),
                drop_rate=float(adaptive_split_drop_rate),
                window_size=int(adaptive_split_window_size),
                min_batch_size=adaptive_split_min_batch_size,
                max_batch_size=adaptive_split_max_batch_size,
                coherence_chunks=int(adaptive_split_coherence_chunks),
                max_chunk_entropy_ratio=adaptive_split_max_chunk_entropy_ratio,
                fallback=bool(adaptive_split_fallback),
            )
        else:
            local_ptr = _event_time_ptr(num_edges=int(idx.numel()), batch_size=bsz, num_windows=nw)
        start = int(idx[0])
        rows: list[list[int]] = []
        for begin, end in local_ptr.tolist():
            rows.append([start + int(begin), start + int(end)])
        out[names[sid]] = torch.tensor(rows, dtype=torch.long) if rows else torch.zeros((0, 2), dtype=torch.long)
    return out


def _event_adaptive_time_ptr(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor,
    batch_size: int | None,
    num_windows: int | None,
    graph_feature: float,
    alpha: float,
    beta: float,
    aggl: float | None,
    enable_drop: bool,
    drop_rate: float,
    window_size: int,
    min_batch_size: int | None,
    max_batch_size: int | None,
    coherence_chunks: int,
    max_chunk_entropy_ratio: float | None,
    fallback: bool,
) -> Tensor:
    num_edges = int(src.numel())
    if num_edges <= 1:
        return _event_time_ptr(num_edges=num_edges, batch_size=batch_size, num_windows=num_windows)
    if batch_size is None:
        batch_size = max(1, int(math.ceil(num_edges / max(1, int(num_windows or 1)))))
    try:
        from atc_starrygl_lib.lib.loader import load_adaptive_split_module

        native = load_adaptive_split_module()
        aggl_value = _adaptive_aggl(src=src, dst=dst) if aggl is None else float(aggl)
        result = native.adaptive_split(
            src.long().contiguous(),
            dst.long().contiguous(),
            ts.to(torch.float64).contiguous(),
            int(batch_size),
            float(graph_feature),
            float(alpha),
            float(beta),
            float(aggl_value),
            bool(enable_drop),
            float(drop_rate),
            int(window_size),
        )
        group_index = result.group_index.long().cpu().contiguous()
        if bool(enable_drop):
            keep = result.keep_indices.long().cpu().contiguous()
            expected = torch.arange(num_edges, dtype=torch.long)
            if int(keep.numel()) != num_edges or not torch.equal(keep, expected):
                raise ValueError("adaptive split edge dropping is not supported by split_time_ptr artifacts")
        ptr = _time_ptr_from_group_index(group_index, num_edges=num_edges)
        return _refine_adaptive_time_ptr(
            ptr,
            src=src,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            coherence_chunks=coherence_chunks,
            max_chunk_entropy_ratio=max_chunk_entropy_ratio,
        )
    except Exception:
        if not fallback:
            raise
        return _event_time_ptr(num_edges=num_edges, batch_size=batch_size, num_windows=num_windows)


def _refine_adaptive_time_ptr(
    ptr: Tensor,
    *,
    src: Tensor,
    min_batch_size: int | None,
    max_batch_size: int | None,
    coherence_chunks: int,
    max_chunk_entropy_ratio: float | None,
) -> Tensor:
    min_size = 1 if min_batch_size is None else max(1, int(min_batch_size))
    max_size = 0 if max_batch_size is None else int(max_batch_size)
    num_chunks = max(0, int(coherence_chunks))
    entropy_limit = None if max_chunk_entropy_ratio is None else float(max_chunk_entropy_ratio)
    if max_size <= 0 and (num_chunks <= 1 or entropy_limit is None or entropy_limit <= 0.0):
        return ptr.long().cpu().contiguous()
    if max_size > 0 and max_size < min_size:
        raise ValueError("adaptive_split_max_batch_size must be >= adaptive_split_min_batch_size")

    src = src.long().cpu().contiguous()
    out: list[list[int]] = []
    for raw_begin, raw_end in ptr.long().cpu().tolist():
        begin = int(raw_begin)
        raw_end = int(raw_end)
        if raw_end <= begin:
            continue
        counts = torch.zeros(num_chunks, dtype=torch.long) if num_chunks > 1 else None
        start = begin
        for pos in range(begin, raw_end):
            if counts is not None:
                chunk = int(src[pos]) % num_chunks
                counts[chunk] += 1
            size = pos - start + 1
            should_cut = False
            if max_size > 0 and size >= max_size:
                should_cut = True
            elif size >= min_size and counts is not None and entropy_limit is not None:
                if _chunk_entropy_ratio(counts) > entropy_limit:
                    should_cut = True
            if should_cut:
                out.append([start, pos + 1])
                start = pos + 1
                if counts is not None:
                    counts.zero_()
        if start < raw_end:
            if out and raw_end - start < min_size and (max_size <= 0 or out[-1][1] - out[-1][0] + raw_end - start <= max_size):
                out[-1][1] = raw_end
            else:
                out.append([start, raw_end])
    return torch.tensor(out, dtype=torch.long) if out else torch.zeros((0, 2), dtype=torch.long)


def _chunk_entropy_ratio(counts: Tensor) -> float:
    total = int(counts.sum().item())
    active = int((counts > 0).sum().item())
    if total <= 0 or active <= 1:
        return 0.0
    probs = counts[counts > 0].float() / float(total)
    entropy = float((-(probs * torch.log(probs))).sum().item())
    return entropy / math.log(float(active))


def _adaptive_aggl(*, src: Tensor, dst: Tensor) -> float:
    if src.numel() == 0:
        return 0.0
    touched = torch.unique(torch.cat([src.long(), dst.long()], dim=0), sorted=False)
    return float(touched.numel()) / float(max(1, int(src.numel()) * 2))


def _time_ptr_from_group_index(group_index: Tensor, *, num_edges: int) -> Tensor:
    if int(group_index.numel()) != int(num_edges):
        raise ValueError("adaptive split group_index length does not match edge count")
    if num_edges == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    changes = (group_index[1:] != group_index[:-1]).nonzero(as_tuple=True)[0] + 1
    boundaries = torch.cat(
        [
            torch.zeros((1,), dtype=torch.long),
            changes.long(),
            torch.tensor([int(num_edges)], dtype=torch.long),
        ]
    )
    return torch.stack([boundaries[:-1], boundaries[1:]], dim=1).long().contiguous()


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
