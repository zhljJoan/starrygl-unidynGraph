from __future__ import annotations

from typing import Iterator

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, Batch, RuntimeContext


class FlareDTDGBackend:
    """Thin bridge for the existing FlareDTDG/STGraphLoader runtime."""

    def __init__(self) -> None:
        self._loader = None

    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
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

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
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
        iterator = self._loader.iter_train(split) if split == "train" else self._loader.iter_eval(split)
        for old_batch in iterator:
            yield Batch(
                split=split,
                roots=torch.empty(0, dtype=torch.long),
                graph=getattr(old_batch, "graph", old_batch),
            )
