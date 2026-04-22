"""PipelineEngine: Orchestrates training/eval loop independent of graph mode and task.

Composes GraphBackend + TaskAdapter + StateManager + Model to produce
a unified training loop that works for all (graph_mode, task) combinations.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from starry_unigraph.data.batch_data import BatchData
from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.registry.task_adapter import TaskAdapter
from starry_unigraph.runtime.backend import GraphBackend, StateManager


class PipelineEngine:
    """Core training loop engine for all graph modes and tasks.

    Eliminates per-graph-mode branching in session.py:
    - Task Type (EdgePredict, NodeRegress, NodeClassify) is orthogonal
    - Graph Mode (CTDG, DTDG, Chunk) is orthogonal
    - Model architecture adapts via task heads
    - Loss/metrics compute via task adapter
    - State management is abstracted
    """

    def __init__(
        self,
        backend: GraphBackend,
        state_manager: StateManager,
        model: nn.Module,
        task_adapter: TaskAdapter,
        device: str = "cpu",
    ):
        """Initialize the pipeline engine.

        Args:
            backend: GraphBackend (provides data chunks)
            state_manager: StateManager (manages RNN/embedding state)
            model: Neural network model with task head
            task_adapter: TaskAdapter (sampling, loss, metrics)
            device: Device to run on ("cuda:0", "cpu", etc.)
        """
        self.backend = backend
        self.state_manager = state_manager
        self.model = model
        self.task_adapter = task_adapter
        self.device = device

    def run_epoch(
        self,
        split: str = "train",
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Run a full epoch (train or eval).

        Args:
            split: "train", "val", or "test"
            batch_size: Batch size

        Returns:
            Epoch metrics: {
                'split': str,
                'loss': float,
                'num_batches': int,
                'metrics': Dict[str, float],
                'outputs': List[Dict],
            }
        """
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()

        outputs = []
        losses = []
        metric_accumulator: Dict[str, list] = {}

        # Iterate over data
        for chunk in self.backend.iter_batches(split, batch_size):
            # Task defines what to sample
            sample_config = self.task_adapter.build_sample_config(
                chunk=chunk,
                model=self.model,
                split=split,
            )

            # Materialize batch from chunk
            # (In real system: calls C++/CUDA extension)
            batch = self._materialize_batch(chunk, sample_config)

            # State preparation
            state = self.state_manager.prepare(
                node_ids=batch.node_ids,
                timestamps=batch.timestamps,
            )

            # Forward pass
            if is_train:
                model_output = self.model.predict(state, batch)
            else:
                with torch.no_grad():
                    model_output = self.model.predict(state, batch)

            # Loss computation (only for train; eval computes metrics)
            if is_train:
                loss = self.task_adapter.compute_loss(model_output, batch)
                losses.append(loss.detach().item())

                # Backward pass
                loss.backward()
                # Optimizer step is handled externally

            # Metrics computation (all splits)
            metrics = self.task_adapter.compute_metrics(model_output, batch)
            for key, value in metrics.items():
                metric_accumulator.setdefault(key, []).append(value)

            # State update
            self.state_manager.update(model_output, chunk)

            # Format output
            formatted = self.task_adapter.format_output(model_output, batch)
            outputs.append(formatted)

        # Aggregate epoch statistics
        avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
        avg_metrics = {
            key: float(sum(values) / len(values))
            for key, values in metric_accumulator.items()
            if values
        }

        return {
            "split": split,
            "loss": avg_loss,
            "num_batches": len(outputs),
            "metrics": avg_metrics,
            "outputs": outputs,
        }

    def iter_batches_with_step(
        self,
        split: str = "train",
        batch_size: int = 64,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over batches and yield step results.

        Useful for custom training loops that want per-batch results.

        Args:
            split: "train", "val", or "test"
            batch_size: Batch size

        Yields:
            Step result: {
                'loss': float (or None for eval),
                'metrics': Dict[str, float],
                'batch_idx': int,
                'output': Dict,
            }
        """
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()

        batch_idx = 0
        for chunk in self.backend.iter_batches(split, batch_size):
            sample_config = self.task_adapter.build_sample_config(
                chunk=chunk,
                model=self.model,
                split=split,
            )
            batch = self._materialize_batch(chunk, sample_config)

            state = self.state_manager.prepare(
                node_ids=batch.node_ids,
                timestamps=batch.timestamps,
            )

            if is_train:
                model_output = self.model.predict(state, batch)
            else:
                with torch.no_grad():
                    model_output = self.model.predict(state, batch)

            loss = None
            if is_train:
                loss = self.task_adapter.compute_loss(model_output, batch)
                loss.backward()

            metrics = self.task_adapter.compute_metrics(model_output, batch)
            self.state_manager.update(model_output, chunk)

            formatted = self.task_adapter.format_output(model_output, batch)

            yield {
                "loss": loss.detach().item() if loss is not None else None,
                "metrics": metrics,
                "batch_idx": batch_idx,
                "output": formatted,
            }

            batch_idx += 1

    @staticmethod
    def _materialize_batch(
        chunk: ChunkAtomic,
        sample_config: Any,  # SampleConfig
    ) -> BatchData:
        """Convert ChunkAtomic + SampleConfig to BatchData.

        In this placeholder, creates a minimal BatchData.
        In real system, this would call C++/CUDA extension for sampling.

        Args:
            chunk: ChunkAtomic with graph structure
            sample_config: SampleConfig specifying sampling params

        Returns:
            BatchData ready for model.forward()
        """
        # Placeholder: real implementation would use C++ extension
        # For now, create empty BatchData with basic fields
        batch = BatchData(
            mfg=None,  # Would be actual MFG from C++ extension
            node_ids=chunk.node_set,
            timestamps=chunk.tcsr_ts if hasattr(chunk, "tcsr_ts") else None,
            chunk_id=chunk.chunk_id,
        )

        # Fill in task-specific fields based on sample_config
        if sample_config.pos_src is not None:
            batch.pos_src = sample_config.pos_src
            batch.pos_dst = sample_config.pos_dst

        if sample_config.target_nodes is not None:
            batch.target_nodes = sample_config.target_nodes
            batch.labels = sample_config.target_labels

        return batch

