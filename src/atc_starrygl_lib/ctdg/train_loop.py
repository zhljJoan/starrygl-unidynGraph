from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
from torch import Tensor

from atc_starrygl_lib.core.types import Batch


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
) -> dict[str, float]:
    """Evaluate a CTDG encoder/head pair over session.iter_batches(split)."""
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
) -> list[tuple[Any, Batch]]:
    """Return raw head outputs with their source batches."""
    encoder.eval()
    head.eval()
    return [(head(encode_batch(encoder, batch), batch), batch) for batch in session.iter_batches(split)]


def encode_batch(encoder: torch.nn.Module, batch: Batch) -> Tensor:
    if hasattr(encoder, "encode"):
        return encoder.encode(batch.graph)
    return encoder(batch.graph)


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
