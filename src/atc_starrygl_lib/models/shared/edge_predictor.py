from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EdgePredictor(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.src_fc = nn.Linear(dim_in, dim_in)
        self.dst_fc = nn.Linear(dim_in, dim_in)
        self.out_fc = nn.Linear(dim_in, 1)

    def forward(
        self,
        h_pos_src: Tensor,
        h_pos_dst: Tensor,
        h_neg_src: Tensor | None = None,
        h_neg_dst: Tensor | None = None,
        neg_samples: int = 1,
        mode: str = "triplet",
    ) -> tuple[Tensor, Tensor]:
        h_pos_src = self.src_fc(h_pos_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_pos_edge = F.relu(h_pos_src + h_pos_dst)
        pos_score = self.out_fc(h_pos_edge)

        if mode == "triplet":
            assert h_neg_dst is not None
            h_neg_dst = self.dst_fc(h_neg_dst)
            h_neg_edge = F.relu(h_pos_src.tile(neg_samples, 1) + h_neg_dst)
        else:
            assert h_neg_src is not None and h_neg_dst is not None
            h_neg_src = self.src_fc(h_neg_src)
            h_neg_dst = self.dst_fc(h_neg_dst)
            h_neg_edge = F.relu(h_neg_src + h_neg_dst)

        neg_score = self.out_fc(h_neg_edge)
        return pos_score, neg_score
