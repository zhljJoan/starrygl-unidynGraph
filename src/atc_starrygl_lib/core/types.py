from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Protocol

from torch import Tensor

from .errors import ArtifactError


@dataclass(frozen=True)
class RuntimeContext:
    config: Mapping[str, Any]
    artifact_root: Path
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"


@dataclass
class ArtifactBundle:
    root: Path
    graph_mode: str
    files: dict[str, Path] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Path:
        try:
            path = self.files[name]
        except KeyError as exc:
            raise ArtifactError(f"missing artifact entry: {name}") from exc
        if not path.exists():
            raise ArtifactError(f"artifact path does not exist: {path}")
        return path


@dataclass(slots=True)
class Batch:
    split: str
    roots: Tensor
    timestamps: Optional[Tensor] = None
    graph: Any = None
    eids: Optional[Tensor] = None
    src: Optional[Tensor] = None
    dst: Optional[Tensor] = None
    ts: Optional[Tensor] = None
    edge_feat: Optional[Tensor] = None
    pos_src: Optional[Tensor] = None
    pos_dst: Optional[Tensor] = None
    commit_src_rows: Optional[Tensor] = None
    commit_dst_rows: Optional[Tensor] = None
    neg_src: Optional[Tensor] = None
    neg_dst: Optional[Tensor] = None
    neg_weight: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    node_ids: Optional[Tensor] = None


# --- Model output types (slots for zero-overhead attribute access) ---

@dataclass(slots=True)
class EdgePredOutput:
    pos_score: Tensor
    neg_score: Tensor


@dataclass(slots=True)
class ClassifyOutput:
    logits: Tensor


@dataclass(slots=True)
class RegressionOutput:
    pred: Tensor


class DataBackend(Protocol):
    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
        ...

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
        ...

    def iter_batches(self, split: str) -> Iterator[Batch]:
        ...


class TaskAdapter(Protocol):
    def compute_loss(self, output: Any, batch: Batch) -> Tensor:
        ...

    def compute_metrics(self, output: Any, batch: Batch) -> dict[str, float]:
        ...
