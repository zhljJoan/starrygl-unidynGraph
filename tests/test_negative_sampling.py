from __future__ import annotations

import torch

from atc_starrygl_lib.sampling.negative import NegativeSamplingRequest, PoolNegativeSampler


def test_pool_negative_sampler_balances_local_remote_weight_mass() -> None:
    sampler = PoolNegativeSampler(train_remote_dst_prob=0.5)
    result = sampler.sample(
        NegativeSamplingRequest(
            pos_src=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            pos_dst=torch.tensor([10, 11, 12, 13], dtype=torch.long),
            num_nodes=32,
            ratio=1,
            split="train",
            local_dst_pool=torch.tensor([20, 21], dtype=torch.long),
            remote_dst_pool=torch.tensor([30, 31], dtype=torch.long),
            generator=torch.Generator().manual_seed(0),
        )
    )
    assert result.weight is not None
    remote_mask = result.neg_dst >= 30
    local_mask = ~remote_mask
    assert remote_mask.any()
    assert local_mask.any()
    remote_mass = result.weight[remote_mask].sum()
    local_mass = result.weight[local_mask].sum()
    assert torch.isclose(remote_mass, local_mass)
