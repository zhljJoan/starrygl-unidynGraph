# Flare backend moved to starry_unigraph.backends.dtdg
# This module is kept for backward compatibility re-exports.
from starry_unigraph.backends.dtdg.runtime import (
    PartitionData,
    RNNStateManager,
    Route,
    RouteData,
    STGraphBlob,
    STGraphLoader,
    STWindowState,
    TensorData,
    build_flare_model,
    extract_graph_labels,
)

__all__ = [
    "PartitionData",
    "RNNStateManager",
    "RouteData",
    "Route",
    "STGraphBlob",
    "STGraphLoader",
    "STWindowState",
    "TensorData",
    "build_flare_model",
    "extract_graph_labels",
]
