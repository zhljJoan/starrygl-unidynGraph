from __future__ import annotations

from abc import ABC, abstractmethod
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from starry_unigraph.types import PreparedArtifacts, SessionContext


@dataclass
class ArtifactLayout:
    root: Path
    directories: dict[str, Path]

    @property
    def meta_path(self) -> Path:
        return self.directories["meta"] / "artifacts.json"


@dataclass
class ArtifactOutput:
    relative_path: str
    payload: Any
    serializer: str = "json"


@dataclass
class ArtifactPayload:
    provider_meta: dict[str, Any]
    outputs: list[ArtifactOutput]


class GraphPreprocessor(ABC):
    artifact_dirs: tuple[str, ...] = ("meta", "partitions", "routes", "sampling", "snapshots")

    @abstractmethod
    def prepare_raw(self, session_ctx: SessionContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_partitions(self, session_ctx: SessionContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        raise NotImplementedError

    def ensure_artifact_layout(self, session_ctx: SessionContext) -> ArtifactLayout:
        directories = {name: session_ctx.artifact_root / name for name in self.artifact_dirs}
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
        return ArtifactLayout(root=session_ctx.artifact_root, directories=directories)

    def emit_artifacts(self, session_ctx: SessionContext, payload: ArtifactPayload) -> PreparedArtifacts:
        layout = self.ensure_artifact_layout(session_ctx)
        self._write_json(layout.meta_path, payload.provider_meta)
        for output in payload.outputs:
            self._write_output(layout.root / output.relative_path, output.payload, output.serializer)
        return PreparedArtifacts(
            meta_path=layout.meta_path,
            directories=layout.directories,
            provider_meta=payload.provider_meta,
        )

    def run(self, session_ctx: SessionContext) -> PreparedArtifacts:
        self.prepare_raw(session_ctx)
        self.build_partitions(session_ctx)
        return self.build_runtime_artifacts(session_ctx)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def _write_output(cls, path: Path, payload: Any, serializer: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if serializer == "json":
            cls._write_json(path, payload)
            return
        if serializer == "torch":
            torch.save(payload, path)
            return
        raise ValueError(f"Unsupported artifact serializer: {serializer}")
