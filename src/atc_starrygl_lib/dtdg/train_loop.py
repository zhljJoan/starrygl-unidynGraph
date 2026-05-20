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
        embeddings = prepare_edge_prediction_embeddings(embeddings, batch)
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


def prepare_edge_prediction_embeddings(embeddings: Any, batch: Batch) -> Any:
    """Make DTDG block embeddings indexable by edge endpoint row ids.

    Recurrent DTDG encoders such as TGCN naturally return dst-node rows for a
    DGL block. Edge prediction batches index into block src rows because source
    endpoints may live in the src tail. Expand dst rows to src rows so shared
    heads can consume the normal Batch.pos_src/pos_dst/neg_dst contract.
    """
    if not isinstance(embeddings, torch.Tensor):
        return embeddings
    graph = batch.graph
    if graph is None or not getattr(graph, "is_block", False):
        return embeddings
    if int(embeddings.size(0)) >= _required_embedding_rows(batch):
        return embeddings
    if int(embeddings.size(0)) != int(graph.num_dst_nodes()):
        return embeddings
    return _expand_dst_embeddings_to_src_rows(embeddings, graph)


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


def _required_embedding_rows(batch: Batch) -> int:
    max_row = -1
    for value in (batch.pos_src, batch.pos_dst, batch.neg_src, batch.neg_dst):
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            max_row = max(max_row, int(value.max().item()))
    return max_row + 1


def _expand_dst_embeddings_to_src_rows(embeddings: torch.Tensor, graph: Any) -> torch.Tensor:
    num_src = int(graph.num_src_nodes())
    num_dst = int(graph.num_dst_nodes())
    out = embeddings.new_zeros((num_src, *embeddings.shape[1:]))
    out[:num_dst] = embeddings

    route = getattr(graph, "route", None)
    send_index = getattr(route, "send_index", None)
    if route is not None and send_index is not None and send_index.numel() > 0:
        if int(send_index.max().item()) < int(embeddings.size(0)):
            routed = graph.flare_apply_route(embeddings)
            recv_rows = getattr(graph, "flare_route_recv_src_rows", None)
            recv_len = int(getattr(route, "recv_len", 0))
            if isinstance(recv_rows, torch.Tensor) and recv_len > 0:
                rows = recv_rows.to(device=out.device, dtype=torch.long)
                out.index_copy_(0, rows, routed[num_dst : num_dst + int(rows.numel())])

    src_ids = graph.srcdata.get("ID") if hasattr(graph, "srcdata") else None
    dst_ids = graph.dstdata.get("ID") if hasattr(graph, "dstdata") else None
    if isinstance(src_ids, torch.Tensor) and isinstance(dst_ids, torch.Tensor) and num_src > num_dst:
        local = _dst_row_lookup(src_ids[num_dst:], dst_ids)
        if local.numel() > 0:
            tail_rows = torch.arange(num_dst, num_src, dtype=torch.long, device=out.device)
            keep = local >= 0
            if bool(keep.any()):
                out.index_copy_(0, tail_rows[keep], embeddings.index_select(0, local[keep].to(embeddings.device)))
    return out


def _dst_row_lookup(nodes: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
    mapping = {int(nid): row for row, nid in enumerate(dst_ids.detach().cpu().tolist())}
    return torch.tensor([mapping.get(int(nid), -1) for nid in nodes.detach().cpu().tolist()], dtype=torch.long, device=nodes.device)
