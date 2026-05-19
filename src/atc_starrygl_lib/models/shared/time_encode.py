from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class TimeEncode(nn.Module):
    def __init__(self, dim: int, parameter_requires_grad: bool = True):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.w.bias = nn.Parameter(torch.zeros(dim))
        if parameter_requires_grad:
            self.w.weight = nn.Parameter(
                torch.from_numpy(
                    1.0 / 10 ** np.linspace(0, 9, dim, dtype=np.float32)
                ).reshape(dim, -1)
            )
        else:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False
            alpha = math.sqrt(dim)
            beta = math.sqrt(dim)
            self.w.weight = nn.Parameter(
                torch.from_numpy(
                    1.0 / alpha ** np.linspace(0, dim / beta, dim, dtype=np.float32)
                ).reshape(dim, -1),
                requires_grad=False,
            )

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(self.w(t.float().reshape(-1, 1)))
