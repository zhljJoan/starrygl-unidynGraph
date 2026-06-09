from __future__ import annotations

import torch

from atc_starrygl_lib.ctdg.runtime.backend import _rank_negative_dst_pools
from atc_starrygl_lib.sampling.negative import (
    MemShareLocalNegativeSampler,
    NegativeSamplingRequest,
    PoolNegativeSampler,
    _membership_mask,
    _membership_mask_sorted,
)


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


def test_membership_mask_sorted_matches_generic_helper() -> None:
    sampled = torch.tensor([10, 11, 14, 15, 10, 18], dtype=torch.long)
    sorted_pool = torch.tensor([10, 12, 15, 18], dtype=torch.long)

    generic = _membership_mask(sampled, sorted_pool)
    optimized = _membership_mask_sorted(sampled, sorted_pool)

    assert torch.equal(generic.cpu(), optimized.cpu())


def test_memshare_local_negative_sampler_assigns_local_weights_for_local_branch() -> None:
    sampler = MemShareLocalNegativeSampler(beta=0.25)
    req = NegativeSamplingRequest(
        pos_src=torch.arange(32, dtype=torch.long),
        pos_dst=torch.arange(32, dtype=torch.long),
        num_nodes=128,
        ratio=1,
        split="train",
        dst_pool=torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long),
        local_dst_pool=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    out = sampler.sample(req)

    assert out.weight is not None
    expected_local = 1.0 / (1.0 - 0.25 + 0.25 * (4.0 / 8.0))
    expected_remote = 1.0 / (0.25 * (4.0 / 8.0))
    local_mask = torch.isin(out.neg_dst.cpu(), req.local_dst_pool)
    assert local_mask.any()
    assert (~local_mask).any()
    assert torch.allclose(out.weight[local_mask], torch.full_like(out.weight[local_mask], expected_local))
    assert torch.allclose(out.weight[~local_mask], torch.full_like(out.weight[~local_mask], expected_remote))


def test_pool_negative_sampler_importance_corrects_local_remote_mixture() -> None:
    sampler = PoolNegativeSampler(train_remote_dst_prob=0.25, correction="importance")
    req = NegativeSamplingRequest(
        pos_src=torch.arange(64, dtype=torch.long),
        pos_dst=torch.arange(64, dtype=torch.long),
        num_nodes=100,
        ratio=1,
        split="train",
        dst_pool=torch.arange(8, dtype=torch.long),
        local_dst_pool=torch.tensor([0, 1], dtype=torch.long),
        remote_dst_pool=torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    out = sampler.sample(req)

    assert out.weight is not None
    remote_mask = out.neg_dst >= 2
    assert remote_mask.any()
    assert (~remote_mask).any()
    expected_local = (2.0 / 8.0) / 0.75
    expected_remote = (6.0 / 8.0) / 0.25
    assert torch.allclose(out.weight[~remote_mask], torch.full_like(out.weight[~remote_mask], expected_local))
    assert torch.allclose(out.weight[remote_mask], torch.full_like(out.weight[remote_mask], expected_remote))


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


def test_memshare_local_policy_includes_replica_hot_dst_in_local_pool() -> None:
    graph = {"dst": torch.tensor([1, 2, 3, 4], dtype=torch.long)}
    dist = {
        "node_owner": torch.tensor([0, 0, 1, 1, 0], dtype=torch.long),
        "replica_node_ids_by_part": [
            torch.tensor([3], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ],
    }

    local, remote = _rank_negative_dst_pools(
        graph=graph,
        dist=dist,
        rank=1,
        runtime_cfg={"negative_sampler_policy": "memshare_local"},
    )

    assert local is not None and local.tolist() == [1, 2, 3]
    assert remote is not None and remote.tolist() == [4]


def test_memshare_local_policy_can_use_owned_only_rank_dst_pool() -> None:
    graph = {"dst": torch.tensor([1, 2, 3, 4], dtype=torch.long)}
    dist = {
        "node_owner": torch.tensor([0, 0, 1, 1, 0], dtype=torch.long),
        "replica_node_ids_by_part": [
            torch.tensor([3], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ],
    }

    local, remote = _rank_negative_dst_pools(
        graph=graph,
        dist=dist,
        rank=1,
        runtime_cfg={
            "negative_sampler_policy": "memshare_local",
            "negative_local_pool_scope": "owned",
        },
    )

    assert local is not None and local.tolist() == [2, 3]
    assert remote is not None and remote.tolist() == [1, 4]
