"""Base task adapter with common utilities."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor

from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.sample_config import SampleConfig


class BaseTaskAdapter:
    """Base class for task adapters. Subclass for specific tasks."""

    task_type = "base"

    def build_sample_config(
        self,
        chunk: Any,
        model: Any,
        split: str,
    ) -> SampleConfig:
        """Build sampling config. Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__}.build_sample_config()")

    def compute_loss(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Tensor:
        """Compute loss. Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__}.compute_loss()")

    def compute_metrics(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, float]:
        """Compute evaluation metrics. Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__}.compute_metrics()")

    def format_output(
        self,
        model_output: Dict[str, Tensor],
        batch: BatchData,
    ) -> Dict[str, Any]:
        """Format output for logging/saving. Default: return model_output."""
        return model_output

    # --- Backward compatibility layer ---
    # For existing code that calls these old methods, provide fallback

    def compute_loss_old_api(self, output: Dict[str, Any]) -> float:
        """Old API: extract loss from output dict directly.
        This is called if new compute_loss() fails.
        """
        return float(output.get("loss", 0.0))

    def format_prediction(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Old API: format prediction. Kept for backward compatibility."""
        return output
