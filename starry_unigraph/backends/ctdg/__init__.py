# CTDG backend moved to starry_unigraph.runtime.online
# This module is kept for backward compatibility re-exports.
from starry_unigraph.runtime.online import (
    AdaParameter,
    AsyncExchangeHandle,
    CTDGDataBatch,
    CTDGFeatureRoute,
    CTDGHistoricalCache,
    CTDGLinkPredictor,
    CTDGMemoryBank,
    CTDGMemoryUpdater,
    CTDGModelOutput,
    CTDGOnlineRuntime,
    CTDGSampleOutput,
    NativeTemporalSampler,
    TGTemporalDataset,
)

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
    "NativeTemporalSampler",
    "TGTemporalDataset",
]
