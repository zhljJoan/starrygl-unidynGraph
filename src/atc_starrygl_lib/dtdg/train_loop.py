from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch

from atc_starrygl_lib.core.types import Batch, ClassifyOutput, RegressionOutput


def train_epoch(
    session: Any,
    model: torch.nn.Module,
    task: Any,
    optimizer: torch.optim.Optimizer,
    *,
    split: str = "train",
) -> dict[str, float]:
    """Run one DTDG training epoch over sliding-window batches."""
    model.train()

    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for batch in session.iter_batches(split):
        raw_output = model(batch.graph)
        output = task_output(raw_output, task=task, training=True)
        loss = task.compute_loss(output, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


@torch.no_grad()
def evaluate(
    session: Any,
    model: torch.nn.Module,
    task: Any,
    *,
    split: str = "val",
) -> dict[str, float]:
    """Evaluate a DTDG recurrent model, carrying state across snapshots."""
    model.eval()

    state = None
    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for batch in session.iter_batches(split):
        raw_output = model(batch.graph, state)
        raw_pred, state = _split_model_output(raw_output)
        output = task_output(raw_pred, task=task, training=False)
        loss = task.compute_loss(output, batch)
        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


@torch.no_grad()
def evaluate_edge_prediction(
    session: Any,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    task: Any,
    *,
    split: str = "test",
) -> dict[str, float]:
    """Evaluate DTDG edge prediction batches with an encoder/head pair."""
    encoder.eval()
    head.eval()

    state = None
    losses: list[float] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for batch in session.iter_batches(split):
        raw_output = _call_stateful(encoder, batch.graph, state)
        embeddings, state = _split_model_output(raw_output)
        if isinstance(embeddings, list):
            embeddings = embeddings[-1]
        output = head(embeddings, batch)
        loss = task.compute_loss(output, batch)
        losses.append(float(loss.detach().item()))
        _append_metrics(metrics, task.compute_metrics(output, batch))

    out = _mean_metrics(metrics)
    out["loss"] = _mean(losses)
    return out


def task_output(raw_output: Any, *, task: Any, training: bool) -> Any:
    raw_pred, _ = _split_model_output(raw_output)
    if training and isinstance(raw_pred, list):
        raw_pred = raw_pred[-1]
    task_type = getattr(getattr(task, "spec", None), "task_type", "")
    if task_type == "node_regression":
        return RegressionOutput(pred=raw_pred)
    if task_type == "node_classification":
        return ClassifyOutput(logits=raw_pred)
    return raw_pred


def _split_model_output(raw_output: Any) -> tuple[Any, Any]:
    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        return raw_output
    return raw_output, None


def _call_stateful(model: torch.nn.Module, graph: Any, state: Any) -> Any:
    if state is None:
        return model(graph)
    try:
        return model(graph, state)
    except TypeError:
        return model(graph)


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
