"""Learnable time encoding using cosine basis.

Stable API for encoding continuous time values into fixed-dimensional vectors.
Used across CTDG online models and DTDG temporal components.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEncode(nn.Module):
    """Learnable cosine-basis time encoding. Stable API - v0.1.0+

    Encodes a scalar time value or batch of times into a fixed-dimensional
    vector using learnable cosine basis functions. The time dimension is then
    concatenated with node/edge features in attention and memory mechanisms.

    Mathematical formulation:
        TimeEncode(t) = cos(W @ t)
        where W = diag(1/10^[0,1,...,dim-1])

    This encoding is universal across CTDG (online) and DTDG (snapshot) modes.

    Args:
        dim: Output embedding dimension (typically 64-128). Larger dims
            capture finer time granularity.

    Example::

        time_enc = TimeEncode(dim=100)
        t = torch.tensor([0.5, 1.2, 2.0])           # [B]
        h_t = time_enc(t)                            # [B, 100]

    Shape:
        - Input: ``[*]`` (any shape, treated as scalar times)
        - Output: ``[*, dim]`` (last dimension = dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.w.bias = nn.Parameter(torch.zeros(dim))
        self.w.weight = nn.Parameter(
            (1.0 / 10 ** torch.linspace(0, 9, dim)).reshape(dim, 1)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, any shape (treated as scalars)

        Returns:
            Time embeddings of shape [*original_shape..., dim]
        """
        return torch.cos(self.w(t.float().reshape(-1, 1))).reshape(*t.shape, self.dim)
