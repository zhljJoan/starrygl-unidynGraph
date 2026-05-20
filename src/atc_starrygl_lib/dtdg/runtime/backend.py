from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, Batch, RuntimeContext
from atc_starrygl_lib.sampling.negative import NegativeSampler, NegativeSamplingRequest, RandomNegativeSampler

from .stgraph_loader import STGraphLoader, STGraphSnapshot, STGraphWindow


class FlareDTDGBackend:
    """Thin bridge for the existing FlareDTDG/STGraphLoader runtime."""

    def __init__(self) -> None:
        self._loader = None
        self._graph: dict[str, Any] | None = None
        self._task_name = "node_regression"
        self._negative_sampler: NegativeSampler | None = None
        self._negative_ratio = 0
        self._runtime_cfg: dict[str, Any] = {}

    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
        prep_cfg = dict(ctx.config.get("preprocess", {}))
        if bool(prep_cfg.get("use_new_pipeline", False)):
            return self._prepare_new_pipeline(ctx, prep_cfg=prep_cfg)

        from starry_unigraph.backends.dtdg import FlareDTDGPreprocessor
        from starry_unigraph.types import SessionContext

        session_ctx = SessionContext(
            config=dict(ctx.config),
            project_root=ctx.artifact_root.parent,
            artifact_root=ctx.artifact_root,
        )
        prepared = FlareDTDGPreprocessor().run(session_ctx)
        return ArtifactBundle(
            root=ctx.artifact_root,
            graph_mode="dtdg",
            files={name: path for name, path in prepared.directories.items()},
            meta=dict(prepared.provider_meta),
        )

    def _prepare_new_pipeline(self, ctx: RuntimeContext, *, prep_cfg: dict[str, Any]) -> ArtifactBundle:
        from atc_starrygl_lib.preprocess.pipeline import run_preprocess_pipeline

        graph_cfg = dict(ctx.config.get("graph", {}))
        source = graph_cfg.get("source") or graph_cfg.get("path") or prep_cfg.get("source")
        if source is None:
            raise ValueError("new DTDG preprocess pipeline requires graph.source (or graph.path/preprocess.source)")
        out_dir = Path(ctx.artifact_root)
        result = run_preprocess_pipeline(
            data=source,
            out_dir=out_dir,
            world_size=int(ctx.world_size),
            algorithm=str(prep_cfg.get("partition_algorithm", "chunk_load_balance")),
            chunks_per_rank=int(prep_cfg.get("chunks_per_rank", 1)),
            mode="snapshot",
            build_feature=bool(prep_cfg.get("build_feature", True)),
            build_partition_data=True,
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            lags=int(prep_cfg.get("lags", 1)),
        )
        files = {"graph": out_dir / "graph.pt", "dist": out_dir / "dist.pt", "meta": out_dir / "meta.json"}
        for rank in range(len(result["ranks"])):
            files[f"rank_{rank:03d}"] = out_dir / f"rank_{rank:03d}.pt"
            files[f"partition_data_{rank:03d}"] = out_dir / f"partition_data_{rank:03d}.pt"
            feature = out_dir / f"feature_{rank:03d}.pt"
            if feature.exists():
                files[f"feature_{rank:03d}"] = feature
        return ArtifactBundle(root=out_dir, graph_mode="dtdg", files=files, meta=dict(result["meta"]))

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
        if f"partition_data_{ctx.rank:03d}" in artifacts.files:
            rank = int(ctx.rank)
            task_cfg = dict(ctx.config.get("task", {}))
            runtime_cfg = dict(ctx.config.get("runtime", {}))
            partition_data = torch.load(artifacts.require(f"partition_data_{rank:03d}"), map_location="cpu", weights_only=False)
            self._graph = torch.load(artifacts.require("graph"), map_location="cpu", weights_only=False)
            self._task_name = str(task_cfg.get("name", "node_regression")).strip().lower()
            self._runtime_cfg = runtime_cfg
            self._negative_ratio = int(runtime_cfg.get("negative_ratio", 0))
            self._negative_sampler = runtime_cfg.get("negative_sampler")
            if self._negative_sampler is None and self._negative_ratio > 0:
                self._negative_sampler = RandomNegativeSampler()
            self._loader = STGraphLoader(
                partition_data=partition_data,
                device=ctx.device,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
            return

        from starry_unigraph.backends.dtdg import FlareRuntimeLoader

        flare_dir = artifacts.require("flare")
        part_id = min(ctx.rank, ctx.world_size - 1)
        partition_path = flare_dir / f"part_{part_id:03d}.pth"
        partition_data = torch.load(partition_path, weights_only=False)
        self._loader = FlareRuntimeLoader.from_partition_data(
            data=partition_data,
            device=ctx.device,
            rank=ctx.rank,
            world_size=ctx.world_size,
            config=dict(ctx.config),
        )

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if self._loader is None:
            raise RuntimeError("build_runtime() must run before iter_batches()")
        if isinstance(self._loader, STGraphLoader):
            yield from self._iter_new_batches(split)
            return
        iterator = self._loader.iter_train(split) if split == "train" else self._loader.iter_eval(split)
        for old_batch in iterator:
            yield Batch(
                split=split,
                roots=torch.empty(0, dtype=torch.long),
                graph=getattr(old_batch, "graph", old_batch),
            )

    def _iter_new_batches(self, split: str) -> Iterator[Batch]:
        assert isinstance(self._loader, STGraphLoader)
        snapshot_ids = _split_snapshot_indices(len(self._loader), split, self._graph)
        if self._task_name in {"edge_prediction", "edge_predict", "link_prediction"}:
            for sid in snapshot_ids:
                snapshot = self._loader.fetch_snapshot(sid)
                yield self._edge_prediction_batch(split=split, snapshot=snapshot, snapshot_id=sid)
            return

        if split == "train":
            for item in self._loader.iter_sliding_windows(
                snapshot_ids=snapshot_ids,
                chunk_order=_chunk_order(self._runtime_cfg, self._loader.chunk_count),
                chunk_decay=_chunk_decay(self._runtime_cfg),
                num_full_snapshots=int(self._runtime_cfg.get("num_full_snapshots", 1)),
                disable_states=bool(self._runtime_cfg.get("disable_states", True)),
                disable_routes=bool(self._runtime_cfg.get("disable_routes", False)),
            ):
                yield _node_batch(split=split, item=item)
            return

        for sid in snapshot_ids:
            snapshot = self._loader.fetch_snapshot(sid)
            yield _node_batch(split=split, item=snapshot)

    def _edge_prediction_batch(self, *, split: str, snapshot: STGraphSnapshot, snapshot_id: int) -> Batch:
        pos_src = snapshot.edge_src.long()
        pos_dst = snapshot.edge_dst.long()
        neg_dst = None
        if self._negative_sampler is not None and self._negative_ratio > 0 and pos_src.numel() > 0:
            result = self._negative_sampler.sample(
                NegativeSamplingRequest(
                    pos_src=pos_src,
                    pos_dst=pos_dst,
                    num_nodes=max(1, int(snapshot.graph.num_dst_nodes())),
                    ratio=int(self._negative_ratio),
                    split=split,
                )
            )
            neg_dst = result.neg_dst.to(pos_dst.device).long().clamp_max(max(0, int(snapshot.graph.num_dst_nodes()) - 1))
        return Batch(
            split=split,
            roots=snapshot.src_ids,
            timestamps=torch.full((int(snapshot.src_ids.numel()),), float(snapshot_id), device=snapshot.src_ids.device),
            graph=snapshot.graph,
            eids=snapshot.edge_ids,
            src=snapshot.src_ids.index_select(0, pos_src) if pos_src.numel() else snapshot.src_ids.new_empty(0),
            dst=snapshot.dst_ids.index_select(0, pos_dst) if pos_dst.numel() else snapshot.dst_ids.new_empty(0),
            ts=torch.full((int(pos_src.numel()),), float(snapshot_id), device=snapshot.src_ids.device),
            pos_src=pos_src,
            pos_dst=pos_dst,
            neg_dst=neg_dst,
        )


def _split_snapshot_indices(num_snapshots: int, split: str, graph: dict[str, Any] | None) -> range:
    split = str(split)
    train_ratio = 0.7
    val_ratio = 0.15
    if graph is not None:
        train_ratio = float(graph.get("train_ratio", train_ratio))
        val_ratio = float(graph.get("val_ratio", val_ratio))
    train_end = int(num_snapshots * train_ratio)
    val_end = train_end + int(num_snapshots * val_ratio)
    if split == "train":
        return range(0, train_end)
    if split == "val":
        return range(train_end, val_end)
    if split == "test":
        return range(val_end, num_snapshots)
    raise ValueError(f"unknown split: {split!r}")


def _node_batch(*, split: str, item: STGraphSnapshot | STGraphWindow) -> Batch:
    if isinstance(item, STGraphWindow):
        graph = item
        latest_graph = item.latest_graph
    else:
        graph = item.graph
        latest_graph = item.graph
    labels = _node_labels(latest_graph)
    node_ids = _node_ids(latest_graph)
    timestamps = torch.full((int(node_ids.numel()),), float(getattr(latest_graph, "flare_snapshot_id", -1)), device=node_ids.device)
    return Batch(
        split=split,
        roots=node_ids,
        timestamps=timestamps,
        graph=graph,
        labels=labels,
        node_ids=node_ids,
    )


def _node_labels(graph: Any) -> torch.Tensor | None:
    if getattr(graph, "is_block", False):
        if "y" in graph.dstdata:
            return graph.dstdata["y"]
        if "y" in graph.srcdata:
            return graph.srcdata["y"]
        return None
    return graph.ndata["y"] if "y" in graph.ndata else None


def _node_ids(graph: Any) -> torch.Tensor:
    if getattr(graph, "is_block", False):
        if "ID" in graph.dstdata:
            return graph.dstdata["ID"]
        return torch.arange(int(graph.num_dst_nodes()), dtype=torch.long, device=graph.device)
    if "ID" in graph.ndata:
        return graph.ndata["ID"]
    return torch.arange(int(graph.num_nodes()), dtype=torch.long, device=graph.device)


def _chunk_order(runtime_cfg: dict[str, Any], chunk_count: int) -> torch.Tensor | None:
    raw = runtime_cfg.get("chunk_order")
    if raw is None:
        if runtime_cfg.get("chunk_decay") is None:
            return None
        return torch.arange(int(chunk_count), dtype=torch.long)
    if isinstance(raw, torch.Tensor):
        return raw.long().cpu()
    if str(raw).lower() == "identity":
        return torch.arange(int(chunk_count), dtype=torch.long)
    if str(raw).lower() == "rand":
        return torch.randperm(int(chunk_count), dtype=torch.long)
    return torch.tensor(list(raw), dtype=torch.long)


def _chunk_decay(runtime_cfg: dict[str, Any]) -> list[int] | None:
    raw = runtime_cfg.get("chunk_decay")
    if raw is None:
        return None
    if isinstance(raw, str):
        return [int(item) for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]
