from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from starry_unigraph.types import PredictionResult, PreparedArtifacts, RuntimeBundle, SessionContext


class RuntimeAdapter(ABC):
    @abstractmethod
    def init_model(self, session_ctx: SessionContext) -> Any:
        raise NotImplementedError

    @abstractmethod
    def init_optimizer(self, session_ctx: SessionContext, model: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load_runtime_state(self, runtime: RuntimeBundle, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def dump_runtime_state(self, runtime: RuntimeBundle) -> dict[str, Any]:
        raise NotImplementedError


class ExecutionAdapter(ABC):
    @abstractmethod
    def train_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        raise NotImplementedError


class GraphProvider(ABC):
    graph_mode = "base"

    def __init__(self, task_adapter: Any):
        self.task_adapter = task_adapter
        self.prepared: PreparedArtifacts | None = None
        self.runtime = RuntimeBundle()

    @abstractmethod
    def prepare_data(self, session_ctx: SessionContext) -> PreparedArtifacts:
        raise NotImplementedError

    @abstractmethod
    def build_runtime(self, session_ctx: SessionContext) -> RuntimeBundle:
        raise NotImplementedError

    @abstractmethod
    def build_train_iterator(self, session_ctx: SessionContext, split: str = "train") -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def build_eval_iterator(self, session_ctx: SessionContext, split: str = "val") -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def build_predict_iterator(self, session_ctx: SessionContext, split: str = "test") -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def run_train_step(self, batch: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_eval_step(self, batch: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_predict_step(self, batch: Any) -> dict[str, Any]:
        raise NotImplementedError
