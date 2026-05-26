from pathlib import Path

import torch

from atc_starrygl_lib.preprocess.partition_metrics import compute_rank_load_from_edge_owner
from atc_starrygl_lib.preprocess.speed_partition_cache import load_speed_partition_cache


def _write_lines(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def test_load_speed_partition_cache_reads_cached_assignment(tmp_path: Path) -> None:
    base = tmp_path / "WikiTalk" / "123457" / "WikiTalk_4parts_top0.1"
    base.mkdir(parents=True)
    _write_lines(base / "output0.txt", [0, 1])
    _write_lines(base / "output1.txt", [2])
    _write_lines(base / "output2.txt", [3])
    _write_lines(base / "output3.txt", [1, 4])
    _write_lines(base / "outputshared.txt", [1])
    _write_lines(base / "edge_output0.txt", [0, 2])
    _write_lines(base / "edge_output1.txt", [1])
    _write_lines(base / "edge_output2.txt", [3])
    _write_lines(base / "edge_output3.txt", [4])

    out = load_speed_partition_cache(
        root=tmp_path,
        dataset_name="WikiTalk",
        num_nodes=5,
        num_edges=5,
        world_size=4,
        topk=0.1,
    )
    assert out["node_owner"].tolist() == [0, 3, 1, 2, 3]
    assert out["edge_owner"].tolist() == [0, 1, 0, 2, 3]
    assert out["replica_mask"].tolist() == [False, True, False, False, False]


def test_compute_rank_load_from_edge_owner_counts_events_and_nodes() -> None:
    rank_load = compute_rank_load_from_edge_owner(
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        edge_owner=torch.tensor([0, 0, 1, 1]),
        time_ptr_2=torch.tensor([[0, 2], [2, 4]]),
        world_size=2,
        node_count_weight=1.0,
    )
    assert rank_load.tolist() == [[5.0, 0.0], [0.0, 5.0]]
