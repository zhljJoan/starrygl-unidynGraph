from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, Batch, RuntimeContext


class MemShareTemporalSamplingBackend:
    """Thin bridge for the existing MemShare-compatible CTDG runtime."""

    def __init__(self) -> None:
        self._session = None
        self._prepared_by = "legacy"

    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
        graph_cfg = dict(ctx.config.get("graph", {}))
        prep_cfg = dict(ctx.config.get("preprocess", {}))
        if bool(prep_cfg.get("use_new_pipeline", False)):
            mode = str(prep_cfg.get("mode", "event")).strip().lower()
            if mode != "event":
                raise ValueError("CTDG new preprocess pipeline currently supports event mode only")
            bundle = self._prepare_new_pipeline(ctx, graph_cfg=graph_cfg, prep_cfg=prep_cfg)
            self._prepared_by = "new_pipeline"
            return bundle

        from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
        from starry_unigraph.types import SessionContext

        session_config = self._build_session_config_with_event_time_ptr(ctx.config, graph_cfg=graph_cfg, prep_cfg=prep_cfg)

        session_ctx = SessionContext(
            config=session_config,
            project_root=ctx.artifact_root.parent,
            artifact_root=ctx.artifact_root,
        )
        session = CTDGSession()
        prepared = session.prepare_data(session_ctx)
        self._session = session
        self._prepared_by = "legacy"
        return ArtifactBundle(
            root=ctx.artifact_root,
            graph_mode="ctdg",
            files={name: path for name, path in prepared.directories.items()},
            meta=dict(prepared.provider_meta),
        )

    def _build_session_config_with_event_time_ptr(self, config: object, *, graph_cfg: dict, prep_cfg: dict) -> dict:
        session_config = dict(config)
        if str(prep_cfg.get("mode", "event")).strip().lower() != "event":
            return session_config
        if prep_cfg.get("batch_size") is None and prep_cfg.get("num_windows") is None:
            return session_config
        source = graph_cfg.get("source") or graph_cfg.get("path") or prep_cfg.get("source")
        if source is None:
            return session_config
        from atc_starrygl_lib.preprocess.dataset import build_dataset

        dataset = build_dataset(
            data=source,
            mode="event",
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            batch_size=prep_cfg.get("batch_size"),
            num_windows=prep_cfg.get("num_windows"),
        )
        next_prep = dict(prep_cfg)
        next_prep["time_ptr_2"] = dataset["time_ptr_2"].tolist()
        next_prep["split"] = dataset["split"].tolist()
        session_config["preprocess"] = next_prep
        return session_config

    def _prepare_new_pipeline(self, ctx: RuntimeContext, *, graph_cfg: dict, prep_cfg: dict) -> ArtifactBundle:
        from atc_starrygl_lib.preprocess.pipeline import run_preprocess_pipeline

        source = graph_cfg.get("source") or graph_cfg.get("path") or prep_cfg.get("source")
        if source is None:
            raise ValueError("new CTDG preprocess pipeline requires graph.source (or graph.path/preprocess.source)")
        out_dir = Path(ctx.artifact_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_preprocess_pipeline(
            data=source,
            out_dir=out_dir,
            world_size=int(ctx.world_size),
            algorithm=str(prep_cfg.get("partition_algorithm", "speed_partition")),
            chunks_per_rank=int(prep_cfg.get("chunks_per_rank", 1)),
            mode="event",
            build_feature=bool(prep_cfg.get("build_feature", True)),
            build_partition_data=bool(prep_cfg.get("build_partition_data", False)),
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            batch_size=prep_cfg.get("batch_size"),
            num_windows=prep_cfg.get("num_windows"),
        )
        files = {
            "graph": out_dir / "graph.pt",
            "dist": out_dir / "dist.pt",
            "meta": out_dir / "meta.json",
        }
        for rank in range(len(result["ranks"])):
            files[f"rank_{rank:03d}"] = out_dir / f"rank_{rank:03d}.pt"
            if bool(prep_cfg.get("build_feature", True)):
                files[f"feature_{rank:03d}"] = out_dir / f"feature_{rank:03d}.pt"
            if bool(prep_cfg.get("build_partition_data", False)):
                files[f"partition_data_{rank:03d}"] = out_dir / f"partition_data_{rank:03d}.pt"
        return ArtifactBundle(root=out_dir, graph_mode="ctdg", files=files, meta=dict(result["meta"]))

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
        if self._prepared_by == "new_pipeline":
            raise NotImplementedError("CTDG runtime reader for new preprocess artifacts is not wired yet")
        from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
        from starry_unigraph.types import SessionContext

        session_ctx = SessionContext(
            config=dict(ctx.config),
            project_root=artifacts.root.parent,
            artifact_root=artifacts.root,
        )
        self._session = self._session or CTDGSession()
        self._session.build_runtime(session_ctx)

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if self._session is None:
            raise RuntimeError("build_runtime() must run before iter_batches()")
        iterator = self._session.iter_train(self._session.online_runtime.ctx) if split == "train" else self._session.iter_eval(self._session.online_runtime.ctx, split)
        for old_batch in iterator:
            yield _coerce_ctdg_batch(old_batch, split)


def _coerce_ctdg_batch(old_batch: object, split: str) -> Batch:
    pos_src = getattr(old_batch, "pos_src", None)
    pos_dst = getattr(old_batch, "pos_dst", None)
    ts = getattr(old_batch, "timestamps", None)
    if pos_src is None:
        pos_src = torch.empty(0, dtype=torch.long)
    roots = pos_src if pos_src is not None else torch.empty(0, dtype=torch.long)
    return Batch(
        split=split,
        roots=roots,
        timestamps=ts,
        graph=getattr(old_batch, "graph", old_batch),
        pos_src=pos_src,
        pos_dst=pos_dst,
        neg_src=getattr(old_batch, "neg_src", None),
        neg_dst=getattr(old_batch, "neg_dst", None),
        node_ids=getattr(old_batch, "node_ids", None),
    )
