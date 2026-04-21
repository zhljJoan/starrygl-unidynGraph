"""Online runtime — CTDG (Continuous-Time Dynamic Graph) backend.

This subpackage implements the online pipeline for event-driven temporal
graph neural networks (e.g. TGN / MemShare) with:

- **Per-node memory + K-slot mailbox** — :class:`CTDGMemoryBank` stores
  and updates node embeddings and mailbox slots, with distributed async
  sync support via ``all_to_all``.
- **Native BTS temporal sampler** — :class:`NativeTemporalSampler` wraps
  the C++ BTS sampler for multi-hop temporal neighbor sampling.
- **Historical cache** — :class:`CTDGHistoricalCache` with adaptive
  threshold (:class:`AdaParameter`) to skip unnecessary syncs.
- **Temporal transformer attention** — :class:`CTDGLinkPredictor` with
  :class:`CTDGMemoryUpdater` (GRU over mailbox history).
- **End-to-end runtime** — :class:`CTDGOnlineRuntime` orchestrates the
  full train/eval/predict loop including negative sampling, memory update,
  conv, scoring, and async distributed sync.

Typical usage::

    from starry_unigraph.runtime.online import (
        TGTemporalDataset, NativeTemporalSampler, CTDGMemoryBank,
        CTDGLinkPredictor, CTDGOnlineRuntime,
    )

    dataset = TGTemporalDataset(root, name)
    runtime = CTDGOnlineRuntime(dataset=dataset, sampler=sampler,
                                 memory=memory, model=model, ...)
    for batch in runtime.iter_batches("train", batch_size=200):
        result = runtime.train_step(batch)
"""

from .cache import AdaParameter, CTDGHistoricalCache
from .data import CTDGDataBatch, TGTemporalDataset
from .factory import build_ctdg_runtime
from .memory import CTDGMemoryBank
from .models import CTDGLinkPredictor, CTDGMemoryUpdater, CTDGModelOutput
from .route import AsyncExchangeHandle, CTDGFeatureRoute
from .runtime import CTDGOnlineRuntime
from .sampler import CTDGSampleOutput, NativeTemporalSampler
from .session import CTDGSession

__all__ = [
    "AdaParameter",
    "AsyncExchangeHandle",
    "CTDGDataBatch",
    "CTDGFeatureRoute",
    "CTDGHistoricalCache",
    "CTDGLinkPredictor",
    "CTDGMemoryBank",
    "CTDGMemoryUpdater",
    "CTDGModelOutput",
    "CTDGOnlineRuntime",
    "CTDGSampleOutput",
    "CTDGSession",
    "NativeTemporalSampler",
    "TGTemporalDataset",
    "build_ctdg_runtime",
]
