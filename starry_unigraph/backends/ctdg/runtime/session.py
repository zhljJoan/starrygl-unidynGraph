"""Lightweight CTDG session for TGN-style training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from starry_unigraph.backends.ctdg.preprocess import CTDGPreprocessor
from starry_unigraph.types import PreparedArtifacts, RuntimeBundle, SessionContext
from .factory import build_ctdg_runtime
from .runtime import CTDGOnlineRuntime


class CTDGSession:
    """Thin CTDG session replacing the old CTDGProvider.

    Usage::

        session = CTDGSession()
        session.prepare_data(ctx)
        session.build_runtime(ctx)
        for batch in session.iter_train(ctx):
            result = session.train_step(batch)
        session.save_checkpoint("ckpt.pt")
    """

    def __init__(self) -> None:
        self.online_runtime: CTDGOnlineRuntime | None = None
        self.prepared: PreparedArtifacts | None = None
        self._runtime: RuntimeBundle = RuntimeBundle()

    def prepare_data(self, session_ctx: SessionContext) -> PreparedArtifacts:
        preprocessor = CTDGPreprocessor()
        self.prepared = preprocessor.run(session_ctx)
        return self.prepared

    def build_runtime(self, session_ctx: SessionContext) -> None:
        self.online_runtime, self._runtime = build_ctdg_runtime(session_ctx)

    def iter_train(self, session_ctx: SessionContext):
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(
            split="train",
            batch_size=int(session_ctx.config["train"]["batch_size"]),
        )

    def iter_eval(self, session_ctx: SessionContext, split: str = "val"):
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(
            split=split,
            batch_size=int(session_ctx.config["train"]["batch_size"]),
        )

    def iter_predict(self, session_ctx: SessionContext, split: str = "test"):
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(
            split=split,
            batch_size=int(session_ctx.config["train"]["batch_size"]),
        )

    def train_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        return self.online_runtime.train_step(batch)

    def eval_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        return self.online_runtime.eval_step(batch, split=batch.split)

    def predict_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        return self.online_runtime.predict_step(batch)

    def save_checkpoint(self, path: str | Path) -> None:
        assert self.online_runtime is not None
        model = self.online_runtime.model
        model_state = (model.module if hasattr(model, "module") else model).state_dict()
        ckpt: dict[str, Any] = {"model": model_state}
        if self.online_runtime.memory_updater is not None:
            ckpt["memory_updater"] = self.online_runtime.memory_updater.state_dict()
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        assert self.online_runtime is not None
        ckpt = torch.load(str(path), map_location="cpu")
        model = self.online_runtime.model
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        if self.online_runtime.memory_updater is not None and "memory_updater" in ckpt:
            self.online_runtime.memory_updater.load_state_dict(ckpt["memory_updater"])
