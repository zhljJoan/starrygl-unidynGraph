from __future__ import annotations

from typing import Iterator

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, Batch, RuntimeContext


class MemShareCTDGBackend:
    """Thin bridge for the existing MemShare-compatible CTDG runtime."""

    def __init__(self) -> None:
        self._session = None

    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
        from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
        from starry_unigraph.types import SessionContext

        session_ctx = SessionContext(
            config=dict(ctx.config),
            project_root=ctx.artifact_root.parent,
            artifact_root=ctx.artifact_root,
        )
        session = CTDGSession()
        prepared = session.prepare_data(session_ctx)
        self._session = session
        return ArtifactBundle(
            root=ctx.artifact_root,
            graph_mode="ctdg",
            files={name: path for name, path in prepared.directories.items()},
            meta=dict(prepared.provider_meta),
        )

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
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
