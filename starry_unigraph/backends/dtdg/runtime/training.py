"""Flare DTDG training, evaluation, and prediction step functions.

Provides:

- :func:`init_flare_training` — build model + optimizer + DDP wrapper and
  store them into the shared :class:`RuntimeBundle`.
- :func:`run_flare_train_step` — one forward + backward + optimizer step.
- :func:`run_flare_eval_step` — one no-grad forward for validation.
- :func:`run_flare_predict_step` — one no-grad forward for inference.

All step functions accept either :class:`STGraphBlob` (training, multi-frame)
or :class:`DTDGBatch` (eval/predict, single-frame with stateful RNN state
carried in ``runtime.state["eval_rnn_state"]``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .models import build_flare_model, extract_graph_labels
from .session_loader import DTDGBatch
from .state import RNNStateManager, STGraphBlob
from starry_unigraph.types import RuntimeBundle, SessionContext


def _infer_dims(partition_data: Any) -> tuple[int, int]:
    input_size = int(partition_data.node_data["x"].data.size(-1))
    labels = partition_data.node_data.get("y")
    if labels is None:
        return input_size, 1
    y = labels.data
    if y.dtype == torch.long:
        return input_size, int(y.max().item() + 1 if y.dim() == 1 else y.size(-1))
    return input_size, int(1 if y.dim() == 1 else y.size(-1))


def _compute_part_loss_scale(local_nodes: int) -> float:
    """Compute loss scale = local_nodes / total_nodes across all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return 1.0
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    t = torch.tensor([local_nodes], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total = float(t.item())
    return local_nodes / total if total > 0 else 1.0


def init_flare_training(runtime: RuntimeBundle, session_ctx: SessionContext, partition_data: Any, device: str | None = None) -> None:
    """Initialize model, optimizer, and DDP for Flare DTDG training.

    Builds a Flare model (via :func:`build_flare_model`), wraps it in
    ``DistributedDataParallel`` when distributed training is active,
    creates an Adam optimizer, computes a per-partition loss scale factor,
    and stores everything into the given *runtime*.

    Args:
        runtime: Shared :class:`RuntimeBundle` to populate.
        session_ctx: Session context with config and distributed info.
        partition_data: :class:`PartitionData` used to infer feature dims.
        device: Target device string (e.g. ``"cuda:0"``).  If ``None``,
            read from ``session_ctx.config["runtime"]["device"]``.

    Side effects:
        Sets ``runtime.model``, ``runtime.optimizer``, and
        ``runtime.state["flare_model"]`` / ``runtime.state["part_loss_scale"]``.
    """
    input_size, output_size = _infer_dims(partition_data)
    if device is None:
        device = session_ctx.config["runtime"]["device"]
    model = build_flare_model(
        model_name=session_ctx.config["model"]["name"],
        input_size=input_size,
        hidden_size=int(session_ctx.config["model"]["hidden_dim"]),
        output_size=output_size,
    ).to(device)

    # Wrap with DDP when distributed is active; specify device_ids for correct GPU binding
    if dist.is_available() and dist.is_initialized():
        local_rank = int(torch.device(device).index) if torch.device(device).index is not None else 0
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(session_ctx.config.get("train", {}).get("lr", 1e-3)),
    )

    # part_loss_scale: compensate for unequal partition sizes across ranks
    local_nodes = int(partition_data.num_dst_nodes)
    part_loss_scale = _compute_part_loss_scale(local_nodes)

    runtime.model = model
    runtime.optimizer = optimizer
    runtime.state["flare_model"] = {
        "name": session_ctx.config["model"]["name"],
        "input_size": input_size,
        "hidden_size": int(session_ctx.config["model"]["hidden_dim"]),
        "output_size": output_size,
        "part_loss_scale": part_loss_scale,
    }
    runtime.state["part_loss_scale"] = part_loss_scale


def _get_labels(graph: Any) -> torch.Tensor | None:
    """Extract labels from a single DGL graph (block or plain)."""
    if graph.is_block:
        if "y" in graph.dstdata:
            y = graph.dstdata["y"]
        elif "y" in graph.srcdata:
            y = graph.srcdata["y"][: graph.num_dst_nodes()]
        else:
            return None
    else:
        y = graph.ndata.get("y")
        if y is None:
            return None
    return y.view(-1, 1) if y.dim() == 1 else y


def _forward_single_graph(
    runtime: RuntimeBundle,
    graph: Any,
    state: Any,
) -> tuple[torch.Tensor, Any]:
    """Run model forward on a single graph with external RNN state. Returns (pred, new_state)."""
    model = runtime.model
    # Patch dummy state methods so model.forward can call flare_fetch/store_state on it
    graph = RNNStateManager.patch_dummy_methods(graph)
    # DDP wraps module; access inner model for single-graph stateful eval
    inner = model.module if hasattr(model, "module") else model
    pred, new_state = inner(graph, state)
    return pred, new_state


def _loss_and_predictions(
    runtime: RuntimeBundle,
    blob_or_batch: STGraphBlob | DTDGBatch,
    training: bool,
) -> tuple[torch.Tensor | None, list[float], list[float] | None]:
    model = runtime.model
    device = next(model.parameters()).device

    if training:
        model.train()
    else:
        model.eval()

    # Resolve input: training receives STGraphBlob, eval/predict receives DTDGBatch
    if isinstance(blob_or_batch, STGraphBlob):
        blob = blob_or_batch
        preds, _ = model(blob)
    else:
        # Stateful single-graph eval/predict: use persisted RNN state across snapshots
        graph = blob_or_batch.graph
        if graph is None:
            raise RuntimeError("flare_native batch.graph is not available")
        state = runtime.state.get("eval_rnn_state", None)
        pred, new_state = _forward_single_graph(runtime, graph, state)
        runtime.state["eval_rnn_state"] = new_state
        preds = pred

    # preds: list[Tensor] when blob is STGraphBlob, single Tensor otherwise
    if isinstance(preds, list):
        pred_tensor = preds[-1]
    else:
        pred_tensor = preds

    if isinstance(blob_or_batch, STGraphBlob):
        last_graph = blob_or_batch.current_graph
    else:
        last_graph = blob_or_batch.graph
    labels = _get_labels(last_graph)

    pred_values = pred_tensor.detach().view(-1).cpu().tolist()
    if labels is None:
        return None, pred_values, None

    labels = labels.to(device)
    if pred_tensor.size(-1) != labels.size(-1):
        labels = labels[..., : pred_tensor.size(-1)]
    part_loss_scale = float(runtime.state.get("part_loss_scale", 1.0))
    loss = F.mse_loss(pred_tensor, labels) * part_loss_scale
    return loss, pred_values, labels.detach().view(-1).cpu().tolist()


def run_flare_train_step(
    runtime: RuntimeBundle,
    batch: Any,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Flare training step (forward + backward + optimizer).

    Args:
        runtime: :class:`RuntimeBundle` with ``model`` and ``optimizer``.
        batch: An :class:`STGraphBlob` (multi-frame training blob).
        kernel_output: Extra metadata dict (passed through to output).

    Returns:
        Dict with keys ``"loss"`` (float), ``"predictions"`` (list),
        ``"targets"`` (list or None), ``"meta"`` (dict).

    Raises:
        RuntimeError: If no labels are found in the batch.
    """
    loss, predictions, targets = _loss_and_predictions(runtime, batch, training=True)
    if loss is None:
        raise RuntimeError("Training labels are required for flare_native train_step")
    loss.backward()
    runtime.optimizer.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().item()),
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"], "sequence_state": "updated"},
    }


def run_flare_eval_step(
    runtime: RuntimeBundle,
    batch: Any,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Flare evaluation step (no-grad forward).

    For warmup batches (``batch.chain == "snapshot_warmup"``), only updates
    the persistent RNN state without computing loss.  Returns ``{}`` for
    warmup batches so the session loop skips them.

    Args:
        runtime: :class:`RuntimeBundle` with ``model``.
        batch: A :class:`DTDGBatch` with ``batch.graph``.
        kernel_output: Extra metadata dict.

    Returns:
        Dict with ``"loss"``, ``"predictions"``, ``"targets"``, ``"meta"``,
        or empty dict ``{}`` for warmup batches.
    """
    is_warmup = getattr(batch, "chain", "") == "snapshot_warmup"
    with torch.no_grad():
        if is_warmup:
            # Warmup: run forward to update RNN state, no loss computed
            graph = batch.graph
            state = runtime.state.get("eval_rnn_state", None)
            _, new_state = _forward_single_graph(runtime, graph, state)
            runtime.state["eval_rnn_state"] = new_state
            return {}  # empty: session.run_epoch skips items with no "loss"
        loss, predictions, targets = _loss_and_predictions(runtime, batch, training=False)
    payload: dict[str, Any] = {
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"]},
    }
    if loss is not None:
        payload["loss"] = float(loss.detach().item())
    return payload


def run_flare_predict_step(
    runtime: RuntimeBundle,
    batch: Any,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Flare prediction step (no-grad forward, no loss).

    Same warmup handling as :func:`run_flare_eval_step`.

    Args:
        runtime: :class:`RuntimeBundle` with ``model``.
        batch: A :class:`DTDGBatch` with ``batch.graph``.
        kernel_output: Extra metadata dict.

    Returns:
        Dict with ``"predictions"``, ``"targets"``, ``"meta"``, or ``{}``
        for warmup batches.
    """
    is_warmup = getattr(batch, "chain", "") == "snapshot_warmup"
    with torch.no_grad():
        if is_warmup:
            graph = batch.graph
            state = runtime.state.get("eval_rnn_state", None)
            _, new_state = _forward_single_graph(runtime, graph, state)
            runtime.state["eval_rnn_state"] = new_state
            return {}
        _, predictions, targets = _loss_and_predictions(runtime, batch, training=False)
    return {
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"]},
    }
