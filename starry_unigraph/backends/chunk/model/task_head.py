"""Backward-compatible chunk task-head imports."""

from starry_unigraph.models.task_head import (
    EdgePredictHead,
    EdgeRegressHead,
    NodeClassifyHead,
    NodeRegressHead,
)

__all__ = ["EdgePredictHead", "EdgeRegressHead", "NodeRegressHead", "NodeClassifyHead"]
