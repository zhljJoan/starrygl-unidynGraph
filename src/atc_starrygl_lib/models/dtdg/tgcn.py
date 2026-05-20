from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from .gcn import GCN
from ..shared.lstm_cell import LSTMCell

if TYPE_CHECKING:
    from dgl import DGLGraph


class TGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, num_gcn_layers: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gcn = GCN(input_size, hidden_size * 3, num_layers=num_gcn_layers, bias=False)
        self.u_t = nn.Linear(hidden_size * 2, hidden_size)
        self.r_t = nn.Linear(hidden_size * 2, hidden_size)
        self.c_t = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, blob: Any, state: Optional[Tensor] = None):
        if self.training:
            # blob is STGraphBlob — iterable of DGLGraph with flare_* methods
            outs: List[Tensor] = []
            urcs: List[Tensor] = self.gcn.layerwise(blob)
            h = state
            for g, x in zip(blob, urcs):
                h = g.flare_fetch_state(h)
                if h is None:
                    h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
                u, r, c = x.chunk(3, dim=-1)
                u = torch.sigmoid(self.u_t(torch.cat([u, h], dim=-1)))
                r = torch.sigmoid(self.r_t(torch.cat([r, h], dim=-1)))
                c = torch.tanh(self.c_t(torch.cat([c, r * h], dim=-1)))
                h = u * h + (1 - u) * c
                g.flare_store_state(h)
                outs.append(h)
            state = h
            outs = [self.out(x) for x in outs]
            return outs, state
        else:
            # blob is a single DGLGraph at eval time
            g = blob
            x = self.gcn(g)
            if state is None or int(state.size(0)) != int(x.size(0)):
                h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
            else:
                h = state
            u, r, c = x.chunk(3, dim=-1)
            u = torch.sigmoid(self.u_t(torch.cat([u, h], dim=-1)))
            r = torch.sigmoid(self.r_t(torch.cat([r, h], dim=-1)))
            c = torch.tanh(self.c_t(torch.cat([c, r * h], dim=-1)))
            h = u * h + (1 - u) * c
            return self.out(h), h
