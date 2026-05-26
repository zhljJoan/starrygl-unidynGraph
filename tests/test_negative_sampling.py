from __future__ import annotations

import torch

from atc_starrygl_lib.ctdg.runtime.backend import _rank_negative_dst_pools
from atc_starrygl_lib.sampling.negative import MemShareLocalNegativeSampler, NegativeSamplingRequest


def test_memshare_local_negative_sampler_returns_expected_weights() -> None:
    sampler = MemShareLocalNegativeSampler(beta=0.1)
    req = NegativeSamplingRequest(
        pos_src=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        pos_dst=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        num_nodes=10,
        ratio=1,
        split="train",
        dst_pool=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        local_dst_pool=torch.tensor([10, 11], dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    out = sampler.sample(req)

    assert out.weight is not None
    expected_local = 1.0 / (1.0 - 0.1 + 0.1 * (2.0 / 4.0))
    expected_remote = 1.0 / (0.1 * (2.0 / 4.0))
    local_mask = torch.isin(out.neg_dst.cpu(), req.local_dst_pool)
    if local_mask.any():
        assert torch.allclose(out.weight[local_mask], torch.full_like(out.weight[local_mask], expected_local))
    if (~local_mask).any():
        assert torch.allclose(out.weight[~local_mask], torch.full_like(out.weight[~local_mask], expected_remote))


def test_memshare_local_policy_builds_rank_dst_pools() -> None:
    graph = {"dst": torch.tensor([1, 2, 3, 3], dtype=torch.long)}
    dist = {"node_owner": torch.tensor([0, 0, 1, 1], dtype=torch.long)}

    local, remote = _rank_negative_dst_pools(
        graph=graph,
        dist=dist,
        rank=1,
        runtime_cfg={"negative_sampler_policy": "memshare_local"},
    )

    assert local is not None and local.tolist() == [2, 3]
    assert remote is not None and remote.tolist() == [1]
