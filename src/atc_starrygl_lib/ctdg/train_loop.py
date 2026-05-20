from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
from torch import Tensor

from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.memory import AsyncMemoryUpdateSpec


def train_epoch(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    task: Any,
    optimizer: torch.optim.Optimizer,
    *,
    split: str = "train",
    memory_commit: Any = None,
) -> dict[str, float]:
    """Run one explicit CTDG train epoch over session.iter_batches(split)."""
    encoder.train()
    head.train()

    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for batch in session.iter_batches(split):
        emb = encode_batch(encoder, batch)
        output = head(emb, batch)
        loss = task.compute_loss(output, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if memory_commit is not None:
            memory_commit(encoder, batch)

        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


@torch.no_grad()
def evaluate(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    task: Any,
    *,
    split: str = "val",
    memory_commit: Any = None,
) -> dict[str, float]:
    """Evaluate CTDG batches, optionally advancing memory after each prediction."""
    encoder.eval()
    head.eval()

    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for batch in session.iter_batches(split):
        emb = encode_batch(encoder, batch)
        output = head(emb, batch)
        loss = task.compute_loss(output, batch)
        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))
        if memory_commit is not None:
            memory_commit(encoder, batch)

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


@torch.no_grad()
def predict(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    *,
    split: str = "test",
    memory_commit: Any = None,
) -> list[tuple[Any, Batch]]:
    """Return raw outputs, optionally updating memory after each emitted output."""
    encoder.eval()
    head.eval()
    outputs: list[tuple[Any, Batch]] = []
    for batch in session.iter_batches(split):
        output = head(encode_batch(encoder, batch), batch)
        outputs.append((output, batch))
        if memory_commit is not None:
            memory_commit(encoder, batch)
    return outputs


def encode_batch(encoder: torch.nn.Module, batch: Batch) -> Tensor:
    if hasattr(encoder, "encode"):
        return encoder.encode(batch.graph)
    return encoder(batch.graph)


class CTDGMemoryCommitHook:
    """Explicit post-step memory/mailbox writeback hook for CTDG training."""

    def __init__(self, updater: Any = None, *, wait_apply: bool = True) -> None:
        self.updater = updater
        self.wait_apply = bool(wait_apply)
        self.last_handle = None

    def __call__(self, encoder: torch.nn.Module, batch: Batch) -> None:
        updater = self.updater or _find_memory_updater(encoder)
        if updater is None:
            return
        spec = AsyncMemoryUpdateSpec.from_edges(
            batch.src,
            batch.dst,
            batch.ts,
            edge_feat=_first_edge_feature(batch.graph),
            wait_apply=self.wait_apply,
        ) if batch.src is not None and batch.dst is not None and batch.ts is not None else AsyncMemoryUpdateSpec(
            wait_apply=self.wait_apply,
        )
        if hasattr(updater, "submit_commit"):
            handle = updater.submit_commit(spec)
        elif hasattr(updater, "commit"):
            handle = updater.commit(spec)
        else:
            return
        self.last_handle = handle
        if self.wait_apply and handle is not None and hasattr(handle, "wait_apply"):
            handle.wait_apply()


def _append_metrics(dst: defaultdict[str, list[float]], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        dst[key].append(float(value))


def _mean_metrics(metrics: defaultdict[str, list[float]]) -> dict[str, float]:
    return {key: _mean(values) for key, values in metrics.items()}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _find_memory_updater(module: torch.nn.Module) -> Any:
    current: Any = module
    if hasattr(current, "module"):
        current = current.module
    for name in ("memory_updater", "updater", "memory"):
        candidate = getattr(current, name, None)
        if candidate is not None and (hasattr(candidate, "submit_commit") or hasattr(candidate, "commit")):
            return candidate
    for child in current.modules() if hasattr(current, "modules") else ():
        if child is current:
            continue
        if hasattr(child, "submit_commit") or hasattr(child, "commit"):
            return child
    return None


def _first_edge_feature(graph: Any) -> Tensor | None:
    block = _first_block(graph)
    edata = getattr(block, "edata", None)
    if not isinstance(edata, dict):
        return None
    feature = edata.get("f")
    if feature is None:
        feature = edata.get("feat")
    return feature


def _first_block(graph: Any) -> Any:
    if isinstance(graph, (list, tuple)):
        if not graph:
            return None
        first = graph[0]
        if isinstance(first, (list, tuple)):
            return first[0] if first else None
        return first
    return graph
