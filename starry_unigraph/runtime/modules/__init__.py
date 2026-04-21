"""Reusable neural network model components for temporal graph learning.

This module contains stable, backend-agnostic components that are shared across
CTDG, DTDG, and Chunk backends:

- :class:`TimeEncode` — learnable time encoding using cosine basis
- :class:`GCNStack` — multi-layer GCN message passing (DGL-based)
- :class:`TemporalTransformerConv` — temporal multi-head attention (CTDG-specific)
- RNN cell references — MatGRUCell, _LSTMCell for stateful models

All components are PyTorch nn.Module subclasses and can be composed into
larger models or task-specific heads.
"""

from .time_encode import TimeEncode
from .gcn_layers import GCNStack
from .temporal_conv import TemporalTransformerConv
from .rnn_cells import MatGRUCell, _LSTMCell

__version__ = "0.1.0"
__all__ = [
    "TimeEncode",
    "GCNStack",
    "TemporalTransformerConv",
    "MatGRUCell",
    "_LSTMCell",
]
