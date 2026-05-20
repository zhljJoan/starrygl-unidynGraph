from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.dtdg.runtime import FlareDTDGBackend
from atc_starrygl_lib.dtdg.runtime.stgraph_loader import STGraphWindow
from atc_starrygl_lib.dtdg.train_loop import evaluate, train_epoch
from atc_starrygl_lib.models.dtdg import TGCN
from atc_starrygl_lib.tasks import NodeRegressionTask


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_dist(args.device)
    artifact_root = Path(args.artifact_root)
    device = _device(args.device, local_rank)
    source = _dataset_source(args)

    ctx = _ctx(args, source=source, artifact_root=artifact_root, rank=rank, world_size=world_size, device=str(device))
    backend = FlareDTDGBackend()
    if args.prepare and rank == 0:
        backend.prepare(ctx)
    if dist.is_initialized():
        dist.barrier()
    artifacts = _bundle(artifact_root, world_size=world_size)
    backend.build_runtime(ctx, artifacts)

    graph = torch.load(artifacts.require("graph"), map_location="cpu", weights_only=False)
    route_summary = _validate_routes(backend, split="train", device=device, max_batches=int(args.route_check_batches))
    model = TGCN(
        input_size=_node_feature_dim(graph),
        hidden_size=int(args.hidden_dim),
        output_size=_node_label_dim(graph),
        num_gcn_layers=int(args.gcn_layers),
    ).to(device)
    task = NodeRegressionTask()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    summary: dict[str, Any] = {
        "artifact_root": str(artifact_root),
        "device": str(device),
        "rank": rank,
        "world_size": world_size,
        "route": route_summary,
        "epochs": [],
    }
    for epoch in range(int(args.epochs)):
        train_metrics = train_epoch(backend, model, task, optimizer, split="train")
        val_metrics = evaluate(backend, model, task, split="val")
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        summary["epochs"].append(row)
        print(json.dumps({"rank": rank, **row}, sort_keys=True), flush=True)

    test_metrics = evaluate(backend, model, task, split="test")
    summary["test"] = test_metrics
    print(json.dumps({"rank": rank, "test": test_metrics, "route": route_summary}, sort_keys=True), flush=True)

    if args.output and rank == 0:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-process DTDG STGraphLoader node-regression smoke.")
    parser.add_argument("--data", default=None, help="Optional .pth graph source or DyGLib-style ml_*.csv event file. Defaults to a synthetic snapshot graph.")
    parser.add_argument("--node-feat", default=None, help="Optional node feature .npy for CSV input. Defaults to sibling ml_*_node.npy when present.")
    parser.add_argument("--event-snapshot-bins", type=int, default=16, help="Number of time snapshots to build from CSV event data.")
    parser.add_argument("--max-events", type=int, default=50000, help="Maximum CSV events used by this smoke; use 0 for all events.")
    parser.add_argument("--artifact-root", default="/tmp/atc_dtdg_stgraph_smoke", help="Artifact output/input directory.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--gcn-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--num-snapshots", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--partition-algorithm", default="chunk_load_balance")
    parser.add_argument("--chunks-per-rank", type=int, default=2)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--chunk-order", choices=("none", "identity", "rand"), default="identity")
    parser.add_argument("--chunk-decay", default="1", help="Comma-separated chunk decay list, or empty for none.")
    parser.add_argument("--num-full-snapshots", type=int, default=1)
    parser.add_argument("--disable-states", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-routes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route-check-batches", type=int, default=2, help="Number of train batches used to validate route all_to_all.")
    return parser.parse_args()


def _ctx(
    args: argparse.Namespace,
    *,
    source: Any,
    artifact_root: Path,
    rank: int,
    world_size: int,
    device: str,
) -> RuntimeContext:
    runtime_cfg: dict[str, Any] = {
        "num_full_snapshots": int(args.num_full_snapshots),
        "disable_states": bool(args.disable_states),
        "disable_routes": bool(args.disable_routes),
    }
    if args.chunk_order != "none":
        runtime_cfg["chunk_order"] = args.chunk_order
    chunk_decay = _chunk_decay(args.chunk_decay)
    if chunk_decay is not None:
        runtime_cfg["chunk_decay"] = chunk_decay
    return RuntimeContext(
        config={
            "graph": {"mode": "dtdg", "source": source},
            "task": {"name": "node_regression"},
            "preprocess": {
                "use_new_pipeline": True,
                "partition_algorithm": args.partition_algorithm,
                "chunks_per_rank": int(args.chunks_per_rank),
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "lags": int(args.lags),
            },
            "runtime": runtime_cfg,
        },
        artifact_root=artifact_root,
        rank=rank,
        world_size=world_size,
        device=device,
    )


def _init_dist(device_arg: str) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return rank, world_size, local_rank
    backend = "nccl" if str(device_arg).startswith("cuda") and torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _device(device_arg: str, local_rank: int) -> torch.device:
    if str(device_arg).startswith("cuda"):
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _dataset_source(args: argparse.Namespace) -> Any:
    if args.data:
        path = Path(args.data)
        if path.suffix == ".csv":
            return _dyglib_csv_to_snapshot_graph(
                path,
                node_feat_path=None if args.node_feat is None else Path(args.node_feat),
                num_snapshots=int(args.event_snapshot_bins),
                max_events=int(args.max_events),
            )
        return args.data
    return _synthetic_snapshot_graph(num_nodes=int(args.num_nodes), num_snapshots=int(args.num_snapshots))


def _synthetic_snapshot_graph(*, num_nodes: int, num_snapshots: int) -> dict[str, Any]:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    snapshots = []
    for sid in range(int(num_snapshots)):
        src = torch.arange(num_nodes, dtype=torch.long)
        dst = (src + sid + 1) % num_nodes
        snapshots.append({
            "src": src,
            "dst": dst,
            "ts": torch.full((num_nodes,), float(sid), dtype=torch.float32),
        })
    node_id = torch.arange(num_nodes, dtype=torch.float32)
    node_feat = torch.stack(
        [
            node_id / max(num_nodes - 1, 1),
            torch.sin(node_id),
            torch.cos(node_id),
        ],
        dim=1,
    )
    node_label = (node_feat[:, :1] * 0.5 + node_feat[:, 1:2] * 0.25).contiguous()
    return {
        "snapshots": snapshots,
        "num_nodes": int(num_nodes),
        "node_feat": node_feat,
        "node_label": node_label,
    }


def _dyglib_csv_to_snapshot_graph(
    path: Path,
    *,
    node_feat_path: Path | None,
    num_snapshots: int,
    max_events: int,
) -> dict[str, Any]:
    import pandas as pd

    if num_snapshots <= 0:
        raise ValueError("--event-snapshot-bins must be positive for CSV input")
    df = pd.read_csv(path)
    if max_events > 0:
        df = df.iloc[: int(max_events)]
    cols = {str(col).lower(): col for col in df.columns}
    src_col = cols.get("src") or cols.get("u")
    dst_col = cols.get("dst") or cols.get("i")
    ts_col = cols.get("ts") or cols.get("time")
    if src_col is None or dst_col is None or ts_col is None:
        raise ValueError(f"CSV input must include src/dst/ts or u/i/ts columns: {path}")

    src = torch.as_tensor(df[src_col].to_numpy(), dtype=torch.long)
    dst = torch.as_tensor(df[dst_col].to_numpy(), dtype=torch.long)
    ts = torch.as_tensor(df[ts_col].to_numpy(), dtype=torch.float32)
    order = torch.argsort(ts, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    ts = ts.index_select(0, order)
    num_nodes = int(torch.cat([src, dst]).max().item()) + 1 if src.numel() else 0

    snapshots = []
    boundaries = torch.linspace(0, int(src.numel()), steps=int(num_snapshots) + 1, dtype=torch.long)
    for sid in range(int(num_snapshots)):
        begin = int(boundaries[sid])
        end = int(boundaries[sid + 1])
        snapshots.append({
            "src": src[begin:end],
            "dst": dst[begin:end],
            "ts": ts[begin:end],
        })

    node_feat = _load_or_build_node_features(
        csv_path=path,
        node_feat_path=node_feat_path,
        src=src,
        dst=dst,
        num_nodes=num_nodes,
    )
    indeg = torch.bincount(dst, minlength=num_nodes).float()
    node_label = torch.log1p(indeg).unsqueeze(1)
    if node_label.numel() > 0 and float(node_label.max().item()) > 0.0:
        node_label = node_label / node_label.max()
    return {
        "snapshots": snapshots,
        "num_nodes": num_nodes,
        "node_feat": node_feat,
        "node_label": node_label,
    }


def _load_or_build_node_features(
    *,
    csv_path: Path,
    node_feat_path: Path | None,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if node_feat_path is None:
        inferred = csv_path.with_name(csv_path.stem + "_node.npy")
        node_feat_path = inferred if inferred.exists() else None
    if node_feat_path is not None and node_feat_path.exists():
        import numpy as np

        base = torch.as_tensor(np.load(node_feat_path), dtype=torch.float32)
        if int(base.size(0)) < num_nodes:
            pad = torch.zeros((num_nodes - int(base.size(0)), int(base.size(1))), dtype=base.dtype)
            base = torch.cat([base, pad], dim=0)
        base = base[:num_nodes].contiguous()
        if bool((base.abs().sum(dim=1) > 0).any()):
            return base

    node_id = torch.arange(num_nodes, dtype=torch.float32)
    denom = max(num_nodes - 1, 1)
    indeg = torch.bincount(dst, minlength=num_nodes).float()
    outdeg = torch.bincount(src, minlength=num_nodes).float()
    if indeg.numel() and float(indeg.max().item()) > 0.0:
        indeg = indeg / indeg.max()
    if outdeg.numel() and float(outdeg.max().item()) > 0.0:
        outdeg = outdeg / outdeg.max()
    return torch.stack([node_id / denom, indeg, outdeg], dim=1).contiguous()


def _bundle(root: Path, *, world_size: int = 1) -> ArtifactBundle:
    files = {
        "graph": root / "graph.pt",
        "dist": root / "dist.pt",
        "meta": root / "meta.json",
    }
    for rank in range(int(world_size)):
        files[f"rank_{rank:03d}"] = root / f"rank_{rank:03d}.pt"
        files[f"partition_data_{rank:03d}"] = root / f"partition_data_{rank:03d}.pt"
        feature = root / f"feature_{rank:03d}.pt"
        if feature.exists():
            files[f"feature_{rank:03d}"] = feature
    return ArtifactBundle(root=root, graph_mode="dtdg", files=files)


def _validate_routes(backend: FlareDTDGBackend, *, split: str, device: torch.device, max_batches: int) -> dict[str, Any]:
    checked = 0
    routed = 0
    send_total = 0
    recv_total = 0
    if max_batches <= 0:
        return {"checked": checked, "routed": routed, "send_total": send_total, "recv_total": recv_total}
    for batch in backend.iter_batches(split):
        for graph in _graphs_from_batch_graph(batch.graph):
            route = getattr(graph, "route", None)
            if route is None or route.send_index is None:
                continue
            rows = int(graph.num_dst_nodes()) if hasattr(graph, "num_dst_nodes") else int(graph.num_nodes())
            x = torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1)
            y = graph.flare_apply_route(x)
            expected = rows + int(route.recv_len)
            if int(y.size(0)) != expected:
                raise RuntimeError(f"route output rows mismatch: got {int(y.size(0))}, expected {expected}")
            checked += 1
            routed += int(route.send_len > 0 or route.recv_len > 0)
            send_total += int(route.send_len)
            recv_total += int(route.recv_len)
        if checked >= int(max_batches):
            break
    return {
        "checked": checked,
        "routed": routed,
        "send_total": send_total,
        "recv_total": recv_total,
    }


def _graphs_from_batch_graph(graph: Any) -> list[Any]:
    if isinstance(graph, STGraphWindow):
        return list(graph)
    if isinstance(graph, (list, tuple)):
        out = []
        for item in graph:
            out.extend(_graphs_from_batch_graph(item))
        return out
    return [graph]


def _node_feature_dim(graph: dict[str, Any]) -> int:
    feat = graph.get("node_feat")
    if feat is None:
        raise ValueError("DTDG STGraph smoke requires node_feat")
    return int(torch.as_tensor(feat).size(-1))


def _node_label_dim(graph: dict[str, Any]) -> int:
    labels = graph.get("node_label")
    if labels is None:
        raise ValueError("DTDG STGraph smoke requires node_label")
    labels = torch.as_tensor(labels)
    if labels.dim() == 1:
        return 1
    return int(labels.size(-1))


def _chunk_decay(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    return [int(item) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
