"""RNN cell primitives for stateful temporal models.

References and reference implementations used in DTDG backends:
- MatGRUCell: Weight-evolving GRU for EvolveGCN
- _LSTMCell: Custom LSTM cell for stacking in FlareMPNNLSTM
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatGRUCell(nn.Module):
    """GRU cell for evolving weight matrices (EvolveGCN-H). DTDG reference.

    Used in FlareEvolveGCN to evolve GCN weight matrices across snapshots.
    The cell evolves a matrix over time based on graph context input.

    Args:
        in_feats: Input (context) feature dimension.
        out_feats: Output (matrix) feature dimension.

    Example::

        mat_gru = MatGRUCell(in_feats=64, out_feats=256)
        prev_weight = torch.randn(64, 256)
        ctx = torch.randn(64)
        new_weight = mat_gru(prev_weight, ctx)          # [64, 256]
    """

    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.update = nn.Linear(in_feats + out_feats, out_feats)
        self.reset = nn.Linear(in_feats + out_feats, out_feats)
        self.htilda = nn.Linear(in_feats + out_feats, out_feats)

    def forward(self, prev_w: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_w: Previous weight matrix [out_feats, ...]
            inputs: Context input [in_feats], broadcast to match prev_w rows

        Returns:
            Updated weight matrix [out_feats, ...]
        """
        inputs = inputs.repeat(prev_w.size(0), 1)
        xc = torch.cat([inputs, prev_w], dim=1)
        z_t = torch.sigmoid(self.update(xc))
        r_t = torch.sigmoid(self.reset(xc))
        g_c = torch.cat([inputs, r_t * prev_w], dim=1)
        h_tilde = torch.tanh(self.htilda(g_c))
        return z_t * prev_w + (1 - z_t) * h_tilde


class _LSTMCell(nn.Module):
    """Custom LSTM cell matching FlareDTDG reference. DTDG reference.

    Supports tuple state (h, c) for use in stacked LSTM models.
    Returns state as tuple, enabling Python tuple concatenation ``s1 + s2``
    for multi-layer state composition.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.

    Example::

        lstm = _LSTMCell(input_size=64, hidden_size=128)
        x = torch.randn(32, 64)
        state = None
        for _ in range(seq_len):
            h, c = lstm(x, state)
            state = (h, c)

        # Multi-layer stacking
        lstm1 = _LSTMCell(64, 128)
        lstm2 = _LSTMCell(128, 128)
        h1, c1 = lstm1(x, None)
        h2, c2 = lstm2(h1, None)
        combined = (h1, c1) + (h2, c2)                  # (h1, c1, h2, c2)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.i_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_t = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_t = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LSTM cell.

        Args:
            x: Input [batch_size, input_size]
            state: Tuple (h, c) or None (initializes to zeros)

        Returns:
            Tuple (h_new, c_new) representing updated hidden and cell states
        """
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
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
