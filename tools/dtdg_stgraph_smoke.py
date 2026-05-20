from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.dtdg.runtime import FlareDTDGBackend
from atc_starrygl_lib.dtdg.train_loop import evaluate, train_epoch
from atc_starrygl_lib.models.dtdg import TGCN
from atc_starrygl_lib.tasks import NodeRegressionTask


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root)
    device = torch.device(args.device)
    source = _dataset_source(args)

    ctx = _ctx(args, source=source, artifact_root=artifact_root, device=str(device))
    backend = FlareDTDGBackend()
    artifacts = backend.prepare(ctx) if args.prepare else _bundle(artifact_root)
    backend.build_runtime(ctx, artifacts)

    graph = torch.load(artifacts.require("graph"), map_location="cpu", weights_only=False)
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
        "epochs": [],
    }
    for epoch in range(int(args.epochs)):
        train_metrics = train_epoch(backend, model, task, optimizer, split="train")
        val_metrics = evaluate(backend, model, task, split="val")
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        summary["epochs"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    test_metrics = evaluate(backend, model, task, split="test")
    summary["test"] = test_metrics
    print(json.dumps({"test": test_metrics}, sort_keys=True), flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


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
    return parser.parse_args()


def _ctx(args: argparse.Namespace, *, source: Any, artifact_root: Path, device: str) -> RuntimeContext:
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
        rank=0,
        world_size=1,
        device=device,
    )


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


def _bundle(root: Path) -> ArtifactBundle:
    files = {
        "graph": root / "graph.pt",
        "dist": root / "dist.pt",
        "meta": root / "meta.json",
        "rank_000": root / "rank_000.pt",
        "partition_data_000": root / "partition_data_000.pt",
    }
    feature = root / "feature_000.pt"
    if feature.exists():
        files["feature_000"] = feature
    return ArtifactBundle(root=root, graph_mode="dtdg", files=files)


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
