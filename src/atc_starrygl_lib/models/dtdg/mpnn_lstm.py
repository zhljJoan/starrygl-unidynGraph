from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .gcn import GCN
from ..shared.lstm_cell import LSTMCell


class MPNN_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gcn = GCN(input_size, hidden_size, num_layers=2)
        self.rnn = nn.ModuleList([
            LSTMCell(hidden_size, hidden_size),
            LSTMCell(hidden_size, hidden_size),
        ])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, blob: Any, state: Optional[Any] = None):
        if self.training:
            outs: List[Tensor] = []
            inps: List[Tensor] = self.gcn.layerwise(blob)
            h = state
            for g, x in zip(blob, inps):
                h = g.flare_fetch_state(h)
                s1 = None if h is None else h[0:2]
                s2 = None if h is None else h[2:4]
                x, _ = s1 = self.rnn[0](x, s1)
                x, _ = s2 = self.rnn[1](x, s2)
                h = s1 + s2
                g.flare_store_state(h)
                outs.append(x)
            state = h
            outs = [self.out(x) for x in outs]
            return outs, state
        else:
            g = blob
            x = self.gcn(g)
            s1 = None if state is None else state[0:2]
            s2 = None if state is None else state[2:4]
            x, _ = s1 = self.rnn[0](x, s1)
            x, _ = s2 = self.rnn[1](x, s2)
            h = s1 + s2
            return self.out(x), h
