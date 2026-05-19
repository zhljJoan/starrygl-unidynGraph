from __future__ import annotations

from typing import Iterator, Optional

from .config import build_context
from .errors import BackendError, ConfigError
from .registry import BackendRegistry, TaskRegistry
from .types import ArtifactBundle, Batch, DataBackend, RuntimeContext, TaskAdapter


class TrainingSession:
    """One public data entry point for CTDG and DTDG."""

    def __init__(self, ctx: RuntimeContext, backend: DataBackend, task: TaskAdapter) -> None:
        self.ctx = ctx
        self.backend = backend
        self.task = task
        self.artifacts: Optional[ArtifactBundle] = None

    @classmethod
    def from_config(cls, ctx: RuntimeContext) -> "TrainingSession":
        try:
            backend_name = str(ctx.config["graph"]["mode"])
            task_name = str(ctx.config["task"]["name"])
        except KeyError as exc:
            raise ConfigError("config must contain graph.mode and task.name") from exc
        backend_factory = BackendRegistry.get(backend_name)
        task_factory = TaskRegistry.get(task_name)
        return cls(ctx=ctx, backend=backend_factory(), task=task_factory())

    @classmethod
    def from_config_file(cls, config_path: str, *, artifact_root: str) -> "TrainingSession":
        return cls.from_config(build_context(config_path, artifact_root=artifact_root))

    def prepare(self) -> ArtifactBundle:
        try:
            self.artifacts = self.backend.prepare(self.ctx)
        except Exception as exc:
            raise BackendError("backend prepare failed") from exc
        return self.artifacts

    def build_runtime(self, artifacts: Optional[ArtifactBundle] = None) -> None:
        self.artifacts = artifacts or self.artifacts
        if self.artifacts is None:
            raise BackendError("prepare() must run before build_runtime() unless artifacts are provided")
        try:
            self.backend.build_runtime(self.ctx, self.artifacts)
        except Exception as exc:
            raise BackendError("backend runtime build failed") from exc

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if not split:
            raise ConfigError("split must be non-empty")
        try:
            yield from self.backend.iter_batches(split)
        except Exception as exc:
            raise BackendError(f"backend batch iteration failed for split={split!r}") from exc
