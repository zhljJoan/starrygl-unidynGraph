from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Iterable, TypeVar


BatchT = TypeVar("BatchT")
ResultT = TypeVar("ResultT")


@dataclass
class AsyncStageHandle:
    token: str
    name: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStageRecord:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    stages: list[PipelineStageRecord] = field(default_factory=list)
    async_ops: list[AsyncStageHandle] = field(default_factory=list)
    _token_counter: int = 0

    def add(self, name: str, payload: dict[str, Any]) -> None:
        self.stages.append(PipelineStageRecord(name=name, payload=payload))

    def begin_async(self, name: str, payload: dict[str, Any] | None = None, depends_on: list[str] | None = None) -> str:
        token = f"{name}:{self._token_counter}"
        self._token_counter += 1
        self.async_ops.append(
            AsyncStageHandle(
                token=token,
                name=name,
                status="pending",
                payload=payload or {},
                depends_on=depends_on or [],
            )
        )
        return token

    def complete_async(self, token: str, payload: dict[str, Any] | None = None) -> None:
        handle = self._get_async_handle(token)
        handle.status = "completed"
        if payload:
            handle.payload.update(payload)

    def fail_async(self, token: str, error: str) -> None:
        handle = self._get_async_handle(token)
        handle.status = "failed"
        handle.payload["error"] = error

    def _get_async_handle(self, token: str) -> AsyncStageHandle:
        for handle in self.async_ops:
            if handle.token == token:
                return handle
        raise KeyError(f"Unknown async token: {token}")

    def as_meta(self) -> dict[str, Any]:
        return {
            "pipeline": [stage.name for stage in self.stages],
            "stage_payloads": {stage.name: stage.payload for stage in self.stages},
            "async_ops": [handle.describe() for handle in self.async_ops],
        }


@dataclass
class StateHandle:
    family: str
    scope: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateDelta:
    family: str
    transition: str
    values: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateWriteback:
    handle: StateHandle
    delta: StateDelta
    version: int | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "handle": self.handle.describe(),
            "delta": self.delta.describe(),
            "version": self.version,
        }


class KernelBatch(ABC):
    index: int
    split: str
    chain: str

    @abstractmethod
    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError


class KernelRuntimeState(ABC):
    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


class KernelResult(ABC):
    @abstractmethod
    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError


class KernelExecutor(ABC, Generic[BatchT, ResultT]):
    @abstractmethod
    def iter_batches(self, split: str, count: int) -> Iterable[BatchT]:
        raise NotImplementedError

    @abstractmethod
    def execute_train(self, batch: BatchT) -> ResultT:
        raise NotImplementedError

    @abstractmethod
    def execute_eval(self, batch: BatchT) -> ResultT:
        raise NotImplementedError

    @abstractmethod
    def execute_predict(self, batch: BatchT) -> ResultT:
        raise NotImplementedError

    @abstractmethod
    def dump_state(self) -> dict[str, Any]:
        raise NotImplementedError
