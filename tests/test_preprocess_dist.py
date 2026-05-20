import torch
import pytest

import atc_starrygl_lib.preprocess.dist as dist_mod
from atc_starrygl_lib.preprocess.dist import (
    assign_chunks_by_load,
    build_chunk_csr,
    build_dist_plan,
    build_local_metis_chunks,
    compute_chunk_load,
    normalize_replica,
)


def test_build_chunk_csr_keeps_global_chunk_order() -> None:
    node_to_chunk = torch.tensor([0, 0, 2, 2, 1, 3])
    chunk_ptr, chunk_nodes = build_chunk_csr(node_to_chunk=node_to_chunk, num_chunks=4)
    assert chunk_ptr.tolist() == [0, 2, 3, 5, 6]
    assert chunk_nodes.tolist() == [0, 1, 4, 2, 3, 5]


def test_local_metis_chunks_partition_each_rank_subgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_metis(*, src, dst, num_nodes, world_size):
        calls.append((src.clone(), dst.clone(), int(num_nodes), int(world_size)))
        return torch.arange(int(num_nodes), dtype=torch.long) % int(world_size)

    monkeypatch.setattr(dist_mod, "run_metis_partition", fake_metis)
    node_owner = torch.tensor([0, 0, 1, 1, 0, 1])
    node_to_chunk, chunk_ptr, chunk_nodes, chunk_owner = build_local_metis_chunks(
        src=torch.tensor([0, 1, 2, 3, 4, 5]),
        dst=torch.tensor([1, 4, 3, 5, 0, 2]),
        node_owner=node_owner,
        chunks_per_rank=2,
        world_size=2,
    )
    assert node_to_chunk.tolist() == [0, 1, 2, 3, 0, 2]
    assert chunk_owner.tolist() == [0, 0, 1, 1]
    assert chunk_ptr.tolist() == [0, 2, 3, 5, 6]
    assert chunk_nodes.tolist() == [0, 4, 1, 2, 5, 3]
    assert [call[2:] for call in calls] == [(3, 2), (3, 2)]


def test_chunk_load_counts_events_and_unique_nodes() -> None:
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])
    node_to_chunk = torch.tensor([0, 0, 1, 1])
    time_ptr_2 = torch.tensor([[0, 2], [2, 4]])
    load = compute_chunk_load(
        src=src,
        dst=dst,
        node_to_chunk=node_to_chunk,
        time_ptr_2=time_ptr_2,
        num_chunks=2,
        node_count_weight=1.0,
    )
    assert load.tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_assign_chunks_by_mean_std_round_robin() -> None:
    load = torch.tensor([[10.0, 1.0, 8.0, 2.0], [10.0, 7.0, 0.0, 2.0]])
    owner = assign_chunks_by_load(load, world_size=2)
    assert owner.tolist() == [0, 0, 1, 1]


def test_normalize_replica_promotes_hot_global_ids() -> None:
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 2, 3])
    node_owner = torch.tensor([0, 1, 0, 1])
    replica_mask, hot_node_ids, node_master = normalize_replica(
        src=src,
        dst=dst,
        node_owner=node_owner,
        hot_topk=1,
    )
    hot = int(hot_node_ids[0])
    assert replica_mask[hot]
    assert node_master[hot] == node_owner[hot]
    assert torch.equal(node_master[~replica_mask], node_owner[~replica_mask])


def test_build_dist_plan_chunk_load_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_metis(*, src, dst, num_nodes, world_size):
        return torch.arange(int(num_nodes), dtype=torch.long) % int(world_size)

    monkeypatch.setattr(dist_mod, "run_metis_partition", fake_metis)
    src = torch.tensor([0, 1, 2, 3, 0, 2])
    dst = torch.tensor([1, 2, 3, 0, 2, 1])
    plan = build_dist_plan(
        src=src,
        dst=dst,
        ts=None,
        num_nodes=4,
        world_size=2,
        time_ptr_2=torch.tensor([[0, 3], [3, 6]]),
        algorithm="chunk_load_balance",
        chunks_per_rank=2,
        hot_topk=1,
    )
    assert plan["format"] == "atc_dist_v1"
    assert plan["node_to_chunk"].numel() == 4
    assert plan["chunk_load"].shape == (2, 4)
    assert plan["replica_mask"].sum().item() == 1
