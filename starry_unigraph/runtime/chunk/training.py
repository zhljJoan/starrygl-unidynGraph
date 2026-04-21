"""Chunk DTDG training, evaluation, and prediction step functions.

Provides chunk-specific step functions that handle chunk-materialized graphs
and their state management, separate from Flare's pre-built snapshot path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from starry_unigraph.backends.dtdg.runtime.state import RNNStateManager, STGraphBlob
from starry_unigraph.backends.dtdg.runtime.training import _get_labels, _loss_and_predictions
from starry_unigraph.types import RuntimeBundle


def run_chunk_train_step(
    runtime: RuntimeBundle,
    blob: STGraphBlob,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one chunk training step (forward + backward + optimizer).

    Args:
        runtime: :class:`RuntimeBundle` with ``model`` and ``optimizer``.
        blob: An :class:`STGraphBlob` (multi-frame chunk training blob).
        kernel_output: Extra metadata dict (passed through to output).

    Returns:
        Dict with keys ``"loss"``, ``"predictions"``, ``"targets"``, ``"meta"``.
    """
    loss, predictions, targets = _loss_and_predictions(runtime, blob, training=True)
    if loss is None:
        raise RuntimeError("Training labels are required for chunk train_step")
    loss.backward()
    runtime.optimizer.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().item()),
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"], "sequence_state": "updated"},
    }


def run_chunk_eval_step(
    runtime: RuntimeBundle,
    batch: Any,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one chunk evaluation step (no-grad forward).

    Args:
        runtime: :class:`RuntimeBundle` with ``model``.
        batch: A chunk batch object.
        kernel_output: Extra metadata dict.

    Returns:
        Dict with ``"loss"``, ``"predictions"``, ``"targets"``, ``"meta"``.
    """
    with torch.no_grad():
        loss, predictions, targets = _loss_and_predictions(runtime, batch, training=False)
    payload: dict[str, Any] = {
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"]},
    }
    if loss is not None:
        payload["loss"] = float(loss.detach().item())
    return payload


def run_chunk_predict_step(
    runtime: RuntimeBundle,
    batch: Any,
    kernel_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute one chunk prediction step (no-grad forward, no loss).

    Args:
        runtime: :class:`RuntimeBundle` with ``model``.
        batch: A chunk batch object.
        kernel_output: Extra metadata dict.

    Returns:
        Dict with ``"predictions"``, ``"targets"``, ``"meta"``.
    """
    with torch.no_grad():
        _, predictions, targets = _loss_and_predictions(runtime, batch, training=False)
    return {
        "predictions": predictions,
        "targets": targets,
        "meta": kernel_output.get("meta", {}) | {"model": runtime.state["flare_model"]},
    }
