"""Flare runtime — high-performance DTDG (Discrete-Time Dynamic Graph) backend.

This subpackage implements the Flare pipeline for snapshot-based temporal
graph neural networks with:

- **Async GPU prefetch** — :class:`STGraphLoader` pins partition data and
  streams snapshots to GPU via a dedicated CUDA stream.
- **RNN state management** — :class:`RNNStateManager` / :class:`STGraphBlob`
  handle per-snapshot hidden state padding, mixing, and persistence across
  the sliding window.
- **Gradient-aware distributed routing** — :class:`Route` / :class:`RouteAgent`
  perform differentiable all-to-all feature exchange with autograd support.
- **Flare model zoo** — :func:`build_flare_model` instantiates
  EvolveGCN / TGCN / MPNN-LSTM with built-in GCN message passing.
- **Training utilities** — :func:`init_flare_training` sets up DDP + loss
  scaling; ``run_flare_{train,eval,predict}_step`` execute one forward/backward.

Typical usage::

    from starry_unigraph.runtime.flare import (
        STGraphLoader, build_flare_model, init_flare_training,
        run_flare_train_step,
    )
    from starry_unigraph.data import PartitionData

    part = PartitionData.load("artifacts/flare/part_000.pth")
    loader = STGraphLoader.from_partition_data(part, device="cuda:0",
                                                chunk_index=chunk_idx)
    for blob in loader:
        result = run_flare_train_step(runtime, blob, {})
"""

from .loader import STGraphLoader
from .models import (
    FlareEvolveGCN,
    FlareMPNNLSTM,
    FlareTGCN,
    GCNStack,
    build_flare_model,
    extract_graph_labels,
)
from .route import Route, RouteAgent
from .session_loader import DTDGBatch, DTDGWindowState, FlareRuntimeLoader, SnapshotRoutePlan
from .state import RNNStateManager, STGraphBlob, STWindowState
from .training import (
    init_flare_training,
    run_flare_eval_step,
    run_flare_predict_step,
    run_flare_train_step,
)

__all__ = [
    "DTDGBatch",
    "DTDGWindowState",
    "FlareEvolveGCN",
    "FlareMPNNLSTM",
    "FlareTGCN",
    "FlareRuntimeLoader",
    "GCNStack",
    "RNNStateManager",
    "Route",
    "RouteAgent",
    "STGraphBlob",
    "STGraphLoader",
    "STWindowState",
    "SnapshotRoutePlan",
    "build_flare_model",
    "extract_graph_labels",
    "init_flare_training",
    "run_flare_eval_step",
    "run_flare_predict_step",
    "run_flare_train_step",
]
