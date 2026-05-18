"""Training/eval step utilities for chunk runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData
from starry_unigraph.backends.chunk.runtime.task_adapter import ChunkTaskAdapter


def run_batch(
    *,
    model: Any,
    batch: BatchData,
    task_adapter: ChunkTaskAdapter,
    optimizer: Optional[torch.optim.Optimizer] = None,
    prediction_head: Optional[Any] = None,
    model_state: Optional[Dict[str, Any]] = None,
    train: bool = True,
) -> Dict[str, Any]:
    """Run one batch with aligned task + evaluation interface.

    Returns:
        {
            "loss": Tensor,
            "metrics": Dict[str, float],
            "output": Dict[str, Tensor]
        }
    """
    model.train(mode=train)
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = None
    if device is not None:
        batch = batch.to(device)
    if optimizer is not None and train:
        optimizer.zero_grad(set_to_none=True)

    # Forward:
    # - If model already returns prediction dict, use directly.
    # - If model returns embeddings, route through prediction_head.
    if model_state is None:
        raw_out = model(batch)
    else:
        raw_out = model(model_state, batch)

    if isinstance(raw_out, dict):
        output: Dict[str, Tensor] = raw_out
    else:
        if prediction_head is None:
            raise ValueError("model returned embeddings/tensor but prediction_head is None")
        output = prediction_head(raw_out, batch)

    loss = task_adapter.compute_loss(output, batch)
    if not isinstance(loss, Tensor):
        loss = torch.as_tensor(loss)

    if train and optimizer is not None and loss.requires_grad:
        loss.backward()
        optimizer.step()

    metrics = task_adapter.compute_metrics(output, batch)

    return {
        "loss": loss.detach(),
        "metrics": metrics,
        "output": output,
    }
