"""Chunk runtime — chunked DTDG (Discrete-Time Dynamic Graph) backend.

This subpackage implements the chunked pipeline for snapshot-based temporal
graph neural networks with:

- **Adaptive chunk materialization** — loads chunked partition artifacts and
  reconstructs graphs on-demand.
- **RNN state management** — reuses :class:`RNNStateManager` / :class:`STGraphBlob`
  for per-snapshot hidden state padding/mixing across training windows.
- **Chunk-specific step functions** — :func:`run_chunk_train_step`,
  :func:`run_chunk_eval_step`, :func:`run_chunk_predict_step` for chunk-materialized batches.

Typical usage::

    from starry_unigraph.runtime.chunk import ChunkRuntimeLoader, run_chunk_train_step

    loader = ChunkRuntimeLoader.from_prepared_artifacts(
        prepared_dir="artifacts/my_dataset",
        device="cuda:0",
        rank=0,
        world_size=1,
        config=config,
    )
    for blob in loader.iter_train(split="train"):
        result = run_chunk_train_step(runtime, blob, {})
"""

from .session_loader import ChunkRuntimeLoader
from .training import (
    run_chunk_eval_step,
    run_chunk_predict_step,
    run_chunk_train_step,
)

__all__ = [
    "ChunkRuntimeLoader",
    "run_chunk_eval_step",
    "run_chunk_predict_step",
    "run_chunk_train_step",
]
