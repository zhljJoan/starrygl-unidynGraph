from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_t = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor] | None = None
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = state
        xh = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.i_t(xh))
        f = torch.sigmoid(self.f_t(xh))
        g = torch.tanh(self.g_t(xh))
        o = torch.sigmoid(self.o_t(xh))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c
