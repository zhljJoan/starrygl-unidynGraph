import math

import torch

from atc_starrygl_lib.preprocess.partition_metrics import (
    aggregate_rank_loads,
    count_cross_partition_edges,
    summarize_rank_loads,
)


def test_count_cross_partition_edges_counts_boundary_edges() -> None:
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 2, 3, 3])
    node_owner = torch.tensor([0, 0, 1, 1])
    assert count_cross_partition_edges(src=src, dst=dst, node_owner=node_owner) == 2


def test_aggregate_rank_loads_sums_chunks_by_owner() -> None:
    chunk_load = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    chunk_owner = torch.tensor([0, 1, 0, 1])
    rank_load = aggregate_rank_loads(chunk_load=chunk_load, chunk_owner=chunk_owner, world_size=2)
    assert rank_load.tolist() == [[4.0, 6.0], [6.0, 4.0]]


def test_summarize_rank_loads_uses_nonzero_min_and_tracks_zero_batches() -> None:
    rank_load = torch.tensor([[4.0, 2.0, 0.0], [3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
    summary = summarize_rank_loads(rank_load)
    assert summary["num_batches"] == 3
    assert summary["zero_min_load_batches"] == 1
    assert summary["all_zero_batches"] == 1
    assert math.isclose(float(summary["avg_batch_load_ratio"]), 1.5)
