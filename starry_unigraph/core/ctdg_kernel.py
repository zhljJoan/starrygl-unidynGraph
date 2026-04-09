from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

from starry_unigraph.native import BTSNativeSampler, build_temporal_neighbor_block, is_bts_sampler_available

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
class FeatureRoutePlan:
    route_type: str
    fanout: list[int]
    feature_keys: list[str] = field(default_factory=lambda: ["x", "edge_attr"])

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "FeatureRoutePlan"}


@dataclass
class StateSyncPlan:
    mode: str
    version: int = 0

    def advance(self) -> None:
        self.version += 1

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "StateSyncPlan"}


@dataclass
class CTDGPartitionBook:
    num_parts: int
    partition_algo: str
    event_count: int

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"graph_mode": "ctdg"}


@dataclass
class CTDGMailboxState:
    memory_type: str
    memory_version: int = 0
    mailbox_version: int = 0
    last_batch: int | None = None

    def update(self, batch_index: int) -> None:
        self.last_batch = batch_index
        self.memory_version += 1
        self.mailbox_version += 1

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CTDGBatch(KernelBatch):
    index: int
    split: str
    roots: list[int]
    neighbors: list[int]
    strategy: str
    history: int
    route_plan: FeatureRoutePlan
    chain: str = (
        "sample->feature_fetch->state_fetch->memory_updater->"
        "neighbor_attention_aggregate->message_generate->state_writeback"
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "chain": self.chain,
            "sample": {
                "roots": self.roots,
                "neighbors": self.neighbors,
                "strategy": self.strategy,
                "history": self.history,
            },
            "route_plan": self.route_plan.describe(),
        }


@dataclass
class CTDGPreparedBatch:
    batch: CTDGBatch
    node_features: list[float]
    edge_features: list[float]

    def describe(self) -> dict[str, Any]:
        return {
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "feature_keys": self.batch.route_plan.feature_keys,
        }


@dataclass
class CTDGStepResult(KernelResult):
    loss: float | None
    predictions: list[float]
    targets: list[int] | None
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
class CTDGRuntimeState(KernelRuntimeState):
    sampler_cursor: int = 0
    last_split: str | None = None
    last_prediction: float | None = None

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CTDGSamplerCore:
    neighbor_limit: list[int]
    strategy: str
    history: int
    feature_route: FeatureRoutePlan
    state_sync: StateSyncPlan
    cursor: int = 0
    native_sampler: BTSNativeSampler | None = None

    def build_lookup(self) -> dict[str, Any]:
        return {
            "neighbor_limit": self.neighbor_limit,
            "strategy": self.strategy,
            "history": self.history,
            "lookup_ready": True,
        }

    def sample_batch(self, batch_index: int, split: str) -> CTDGBatch:
        return CTDGBatch(
            index=batch_index,
            split=split,
            roots=[batch_index],
            neighbors=list(self.neighbor_limit),
            strategy=self.strategy,
            history=self.history,
            route_plan=self.feature_route,
        )

    def sample(self, split: str, count: int = 3) -> list[CTDGBatch]:
        batches = [self.sample_batch(self.cursor + offset, split=split) for offset in range(count)]
        self.cursor += count
        return batches

    def attach_native_sampler(
        self,
        graph_name: str,
        row: torch.Tensor,
        col: torch.Tensor,
        num_nodes: int,
        eid: torch.Tensor,
        timestamp: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        workers: int = 1,
        local_part: int = -1,
        edge_part: torch.Tensor | None = None,
        node_part: torch.Tensor | None = None,
        probability: float = 1.0,
    ) -> None:
        if not is_bts_sampler_available():
            raise RuntimeError("BTS native sampler library is not available in this environment")
        tnb = build_temporal_neighbor_block(
            graph_name=graph_name,
            row=row,
            col=col,
            num_nodes=num_nodes,
            eid=eid,
            edge_weight=edge_weight,
            timestamp=timestamp,
        )
        self.native_sampler = BTSNativeSampler(
            tnb=tnb,
            num_nodes=num_nodes,
            num_edges=eid.numel(),
            num_layers=self.history,
            fanout=list(self.neighbor_limit),
            workers=workers,
            policy=self.strategy,
            local_part=local_part,
            edge_part=edge_part,
            node_part=node_part,
            probability=probability,
        )

    def dump_state(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "lookup": self.build_lookup(),
            "state_sync": self.state_sync.describe(),
            "feature_route": self.feature_route.describe(),
            "native_sampler_attached": self.native_sampler is not None,
        }


@dataclass
class CTDGKernel(KernelExecutor[CTDGBatch, CTDGStepResult]):
    sampler: CTDGSamplerCore
    mailbox_state: CTDGMailboxState
    runtime_state: CTDGRuntimeState = field(default_factory=CTDGRuntimeState)

    def iter_batches(self, split: str, count: int) -> Iterable[CTDGBatch]:
        yield from self.sampler.sample(split=split, count=count)

    def _state_fetch(self, batch: CTDGBatch) -> dict[str, Any]:
        handle = StateHandle(
            family="ctdg",
            scope="node",
            name="memory_mailbox",
            metadata={"memory_type": self.mailbox_state.memory_type},
        )
        return {
            "handle": handle.describe(),
            "memory_state": self.mailbox_state.describe(),
            "fetch_kind": "memory_fetch",
        }

    def _materialize(self, batch: CTDGBatch) -> CTDGPreparedBatch:
        node_features = [float(root * 10) for root in batch.roots]
        edge_features = [float(sum(batch.neighbors))]
        return CTDGPreparedBatch(batch=batch, node_features=node_features, edge_features=edge_features)

    def _memory_updater(self, prepared: CTDGPreparedBatch) -> dict[str, Any]:
        return {
            "transition_kind": "memory_updater_gru",
            "features_seen": len(prepared.node_features),
        }

    def _neighbor_attention_aggregate(self, prepared: CTDGPreparedBatch) -> dict[str, Any]:
        aggregated = [feature + prepared.edge_features[0] for feature in prepared.node_features]
        return {
            "aggregation_kind": "temporal_neighbor_attention",
            "aggregated_embeddings": aggregated,
        }

    def _message_generate(self, attention_state: dict[str, Any]) -> dict[str, Any]:
        messages = [value * 0.5 for value in attention_state["aggregated_embeddings"]]
        return {
            "message_kind": "edge_message_generate",
            "messages": messages,
        }

    def _state_writeback(self, prepared: CTDGPreparedBatch) -> dict[str, Any]:
        self.mailbox_state.update(prepared.batch.index)
        self.sampler.state_sync.advance()
        self.runtime_state.sampler_cursor = self.sampler.cursor
        self.runtime_state.last_split = prepared.batch.split
        handle = StateHandle(
            family="ctdg",
            scope="node",
            name="memory_mailbox",
            metadata={"memory_type": self.mailbox_state.memory_type},
        )
        delta = StateDelta(
            family="ctdg",
            transition="memory_writeback",
            values={
                "memory_state": self.mailbox_state.describe(),
                "state_sync": self.sampler.state_sync.describe(),
            },
        )
        writeback = StateWriteback(
            handle=handle,
            delta=delta,
            version=self.mailbox_state.memory_version,
        )
        return writeback.describe()

    def _run_pipeline(self, batch: CTDGBatch, include_targets: bool, include_loss: bool) -> CTDGStepResult:
        trace = PipelineTrace()
        trace.add("sample", batch.to_payload()["sample"])
        feature_fetch_token = trace.begin_async("feature_fetch", {"mode": "async_candidate"})
        prepared = self._materialize(batch)
        trace.add("feature_fetch", prepared.describe())
        trace.complete_async(feature_fetch_token, prepared.describe())
        state_fetch = self._state_fetch(batch)
        trace.add("state_fetch", state_fetch)
        memory_updater = self._memory_updater(prepared)
        trace.add("memory_updater", memory_updater)
        attention_state = self._neighbor_attention_aggregate(prepared)
        trace.add("neighbor_attention_aggregate", attention_state)
        message_state = self._message_generate(attention_state)
        trace.add("message_generate", message_state)
        state_transition = {
            "memory_updater": memory_updater,
            "attention": attention_state,
            "message_generate": message_state,
        }
        trace.add("state_transition", state_transition)
        writeback_token = trace.begin_async(
            "state_writeback_commit",
            {"mode": "async_candidate"},
            depends_on=[feature_fetch_token],
        )
        state = self._state_writeback(prepared)
        trace.add("state_writeback", state)
        trace.complete_async(writeback_token, state)
        prediction = float(sum(prepared.node_features) + sum(prepared.edge_features))
        self.runtime_state.last_prediction = prediction
        meta = {
            "chain": batch.chain,
            **trace.as_meta(),
            "state": state,
            "state_fetch": state_fetch,
            "state_transition": state_transition,
        }
        return CTDGStepResult(
            loss=float(batch.index + 1) if include_loss else None,
            predictions=[prediction],
            targets=[batch.index] if include_targets else None,
            meta=meta,
        )

    def execute_train(self, batch: CTDGBatch) -> CTDGStepResult:
        return self._run_pipeline(batch, include_targets=True, include_loss=True)

    def execute_eval(self, batch: CTDGBatch) -> CTDGStepResult:
        return self._run_pipeline(batch, include_targets=True, include_loss=True)

    def execute_predict(self, batch: CTDGBatch) -> CTDGStepResult:
        return self._run_pipeline(batch, include_targets=False, include_loss=False)

    def dump_state(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_state.describe(),
            "mailbox": self.mailbox_state.describe(),
            "sampler": self.sampler.dump_state(),
        }
