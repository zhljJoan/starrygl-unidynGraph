from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .protocol import (
    KernelBatch,
    KernelExecutor,
    KernelResult,
    KernelRuntimeState,
    PipelineTrace,
    StateDelta,
    StateHandle,
    StateWriteback,
)


@dataclass
class SnapshotRoutePlan:
    route_type: str
    cache_policy: str

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "SnapshotRoutePlan"}


@dataclass
class DTDGPartitionBook:
    num_parts: int
    partition_algo: str
    snapshot_count: int

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"graph_mode": "dtdg"}


@dataclass
class DTDGWindowState:
    window_size: int
    last_snapshot: int | None = None
    stored_windows: int = 0

    def store(self, snapshot_index: int) -> None:
        self.last_snapshot = snapshot_index
        self.stored_windows += 1

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DTDGBatch(KernelBatch):
    index: int
    split: str
    window_size: int
    route_plan: SnapshotRoutePlan
    adjacency: list[list[float]]
    dense_features: list[float]
    graph: Any = None
    graph_meta: dict[str, Any] = field(default_factory=dict)
    chain: str = "load_snapshot->route_apply->state_fetch->state_transition->state_writeback"

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "chain": self.chain,
            "window": {"snapshot_id": self.index, "window_size": self.window_size},
            "route_plan": self.route_plan.describe(),
            "adjacency": self.adjacency,
            "dense_features": self.dense_features,
            "graph_meta": self.graph_meta,
        }


@dataclass
class DTDGStepResult(KernelResult):
    loss: float | None
    predictions: list[float]
    targets: list[float] | None
    meta: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "predictions": self.predictions,
            "meta": self.meta,
        }
        if self.loss is not None:
            payload["loss"] = self.loss
        if self.targets is not None:
            payload["targets"] = self.targets
        return payload


@dataclass
class DTDGRuntimeState(KernelRuntimeState):
    snapshot_cursor: int = 0
    last_split: str | None = None
    last_prediction: float | None = None

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DTDGSnapshotCore:
    snaps: int
    window_state: DTDGWindowState
    route_plan: SnapshotRoutePlan
    cursor: int = 0
    backend_loader: Any = None

    def build_snapshot_index(self) -> dict[str, Any]:
        return {
            "snaps": self.snaps,
            "window_size": self.window_state.window_size,
            "index_ready": True,
        }

    def load_snapshot(self, snapshot_index: int, split: str) -> DTDGBatch:
        if self.backend_loader is not None:
            return self.backend_loader.load_snapshot(snapshot_index, split=split)
        return DTDGBatch(
            index=snapshot_index,
            split=split,
            window_size=self.window_state.window_size,
            route_plan=self.route_plan,
            adjacency=[[1.0, 0.0], [0.5, 1.0]],
            dense_features=[snapshot_index + 1.0, snapshot_index + 2.0],
        )

    def windows(self, split: str, count: int = 3) -> list[DTDGBatch]:
        batches = [self.load_snapshot(self.cursor + offset, split=split) for offset in range(count)]
        self.cursor += count
        return batches

    def dump_state(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "snapshot_index": self.build_snapshot_index(),
            "window_state": self.window_state.describe(),
            "route_plan": self.route_plan.describe(),
        }


@dataclass
class DTDGKernel(KernelExecutor[DTDGBatch, DTDGStepResult]):
    snapshot_core: DTDGSnapshotCore
    runtime_state: DTDGRuntimeState = field(default_factory=DTDGRuntimeState)

    def iter_batches(self, split: str, count: int) -> Iterable[DTDGBatch]:
        yield from self.snapshot_core.windows(split=split, count=count)

    def _apply_route(self, batch: DTDGBatch) -> dict[str, Any]:
        return {
            "route": batch.route_plan.describe(),
            "routed_features": [value * 2 for value in batch.dense_features],
        }

    def _state_fetch(self, batch: DTDGBatch) -> dict[str, Any]:
        handle = StateHandle(
            family="dtdg",
            scope="window",
            name="temporal_window_state",
            metadata={"window_size": self.snapshot_core.window_state.window_size},
        )
        return {
            "handle": handle.describe(),
            "window_state": self.snapshot_core.window_state.describe(),
            "fetch_kind": "temporal_state_fetch",
        }

    def _snapshot_propagation(self, batch: DTDGBatch, routed: dict[str, Any]) -> list[float]:
        dense = routed["routed_features"]
        adjacency = batch.adjacency
        return [
            adjacency[0][0] * dense[0] + adjacency[0][1] * dense[1],
            adjacency[1][0] * dense[0] + adjacency[1][1] * dense[1],
        ]

    def _state_transition(self, spmm_output: list[float], batch: DTDGBatch) -> dict[str, Any]:
        aggregated = {
            "aggregated_value": sum(spmm_output) / len(spmm_output),
            "window_size": batch.window_size,
        }
        return {
            "transition_kind": "temporal_window_fusion",
            "snapshot_propagation": {"spmm_output": spmm_output},
            "temporal_fusion": aggregated,
        }

    def _state_writeback(self, batch: DTDGBatch) -> dict[str, Any]:
        self.snapshot_core.window_state.store(batch.index)
        self.runtime_state.snapshot_cursor = self.snapshot_core.cursor
        self.runtime_state.last_split = batch.split
        handle = StateHandle(
            family="dtdg",
            scope="window",
            name="temporal_window_state",
            metadata={"window_size": self.snapshot_core.window_state.window_size},
        )
        delta = StateDelta(
            family="dtdg",
            transition="temporal_state_writeback",
            values={"window_state": self.snapshot_core.window_state.describe()},
        )
        writeback = StateWriteback(
            handle=handle,
            delta=delta,
            version=self.snapshot_core.window_state.stored_windows,
        )
        return writeback.describe()

    def _run_pipeline(self, batch: DTDGBatch, include_targets: bool, include_loss: bool) -> DTDGStepResult:
        trace = PipelineTrace()
        load_payload = batch.to_payload()["window"] | {
            "graph_meta": batch.graph_meta,
            "flare_is_full_snapshot": batch.graph_meta.get("flare_is_full_snapshot", True),
            "flare_remap": batch.graph_meta.get("flare_remap", {}),
        }
        trace.add("load_snapshot", load_payload)
        routed = self._apply_route(batch)
        routed["graph_meta"] = batch.graph_meta
        trace.add("route_apply", routed)
        state_fetch = self._state_fetch(batch)
        state_fetch["graph_meta"] = batch.graph_meta
        trace.add("state_fetch", state_fetch)
        spmm_output = self._snapshot_propagation(batch, routed)
        state_transition = self._state_transition(spmm_output, batch)
        trace.add("state_transition", state_transition)
        stored = self._state_writeback(batch)
        trace.add("state_writeback", stored)
        aggregated = state_transition["temporal_fusion"]
        self.runtime_state.last_prediction = aggregated["aggregated_value"]
        meta = {
            "chain": batch.chain,
            "graph_meta": batch.graph_meta,
            **trace.as_meta(),
            "state": stored,
            "spmm_output": spmm_output,
            "aggregated": aggregated,
            "state_fetch": state_fetch,
            "state_transition": state_transition,
        }
        return DTDGStepResult(
            loss=float(batch.index + 1) if include_loss else None,
            predictions=[aggregated["aggregated_value"]],
            targets=[float(batch.index)] if include_targets else None,
            meta=meta,
        )

    def execute_train(self, batch: DTDGBatch) -> DTDGStepResult:
        return self._run_pipeline(batch, include_targets=True, include_loss=True)

    def execute_eval(self, batch: DTDGBatch) -> DTDGStepResult:
        return self._run_pipeline(batch, include_targets=True, include_loss=True)

    def execute_predict(self, batch: DTDGBatch) -> DTDGStepResult:
        return self._run_pipeline(batch, include_targets=False, include_loss=False)

    def dump_state(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_state.describe(),
            "snapshot": self.snapshot_core.dump_state(),
            "window": self.snapshot_core.window_state.describe(),
        }
