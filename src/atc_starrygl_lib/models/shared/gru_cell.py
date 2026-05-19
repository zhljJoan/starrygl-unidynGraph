from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.z_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.x_t = nn.Linear(input_size, hidden_size)
        self.h_t = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        if h is None:
            h = torch.zeros(*x.shape[:-1], self.hidden_size, dtype=x.dtype, device=x.device)
        xh = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.r_t(xh))
        z = torch.sigmoid(self.z_t(xh))
        n = torch.tanh(self.x_t(x) + r * self.h_t(h))
        return (1 - z) * n + z * h
