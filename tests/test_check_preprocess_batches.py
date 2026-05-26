from __future__ import annotations

import torch

from tools.check_preprocess_batches import check_artifact_root


def _graph() -> dict:
    return {
        "src": torch.tensor([0, 1, 2, 3]),
        "dst": torch.tensor([1, 2, 3, 0]),
        "ts": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "edge_ids": torch.arange(4, dtype=torch.long),
        "time_ptr_2": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "val": torch.tensor([[2, 3]], dtype=torch.long),
            "test": torch.tensor([[3, 4]], dtype=torch.long),
        },
    }


def _write_ctdg_artifacts(root, *, corrupt: bool = False) -> None:
    torch.save(_graph(), root / "graph.pt")
    rank0_data = torch.tensor([0] if corrupt else [0, 1], dtype=torch.long)
    rank0 = {
        "rank": 0,
        "local_edge_ids": rank0_data,
        "split_event_pos": {
            "train": {"data": rank0_data, "ptr": torch.tensor([0, 1, 1] if corrupt else [0, 1, 2])},
            "val": {"data": torch.empty(0, dtype=torch.long), "ptr": torch.tensor([0, 0])},
            "test": {"data": torch.empty(0, dtype=torch.long), "ptr": torch.tensor([0, 0])},
        },
        "split_time_ptr": {
            "train": torch.tensor([[0, 1], [1, 1]] if corrupt else [[0, 1], [1, 2]], dtype=torch.long),
            "val": torch.tensor([[0, 0]], dtype=torch.long),
            "test": torch.tensor([[0, 0]], dtype=torch.long),
        },
    }
    rank1 = {
        "rank": 1,
        "local_edge_ids": torch.tensor([2, 3], dtype=torch.long),
        "split_event_pos": {
            "train": {"data": torch.empty(0, dtype=torch.long), "ptr": torch.tensor([0, 0, 0])},
            "val": {"data": torch.tensor([2], dtype=torch.long), "ptr": torch.tensor([0, 1])},
            "test": {"data": torch.tensor([3], dtype=torch.long), "ptr": torch.tensor([0, 1])},
        },
        "split_time_ptr": {
            "train": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
            "val": torch.tensor([[0, 1]], dtype=torch.long),
            "test": torch.tensor([[0, 1]], dtype=torch.long),
        },
    }
    torch.save(rank0, root / "rank_000.pt")
    torch.save(rank1, root / "rank_001.pt")


def _td(items: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    ptr = [0]
    for item in items:
        ptr.append(ptr[-1] + int(item.numel()))
    data = torch.cat(items, dim=0) if items else torch.empty(0, dtype=torch.long)
    return {"ptr": torch.tensor(ptr, dtype=torch.long), "data": data.long()}


def _write_dtdg_artifacts(root) -> None:
    graph = _graph()
    graph["time_ptr_2"] = torch.tensor([[0, 2], [2, 4]], dtype=torch.long)
    torch.save(graph, root / "graph.pt")
    part0 = {
        "rank": 0,
        "edge_ids": _td([torch.tensor([0, 1]), torch.empty(0, dtype=torch.long)]),
        "dst_ids": _td([torch.tensor([1, 2]), torch.empty(0, dtype=torch.long)]),
        "src_ids": _td([torch.tensor([0]), torch.empty(0, dtype=torch.long)]),
        "edge_src": _td([torch.tensor([2, 0]), torch.empty(0, dtype=torch.long)]),
        "edge_dst": _td([torch.tensor([0, 1]), torch.empty(0, dtype=torch.long)]),
    }
    part1 = {
        "rank": 1,
        "edge_ids": _td([torch.empty(0, dtype=torch.long), torch.tensor([2, 3])]),
        "dst_ids": _td([torch.empty(0, dtype=torch.long), torch.tensor([0, 3])]),
        "src_ids": _td([torch.empty(0, dtype=torch.long), torch.tensor([2])]),
        "edge_src": _td([torch.empty(0, dtype=torch.long), torch.tensor([2, 1])]),
        "edge_dst": _td([torch.empty(0, dtype=torch.long), torch.tensor([1, 0])]),
    }
    torch.save(part0, root / "partition_data_000.pt")
    torch.save(part1, root / "partition_data_001.pt")


def test_check_ctdg_batches_reconstruct_global_windows(tmp_path) -> None:
    _write_ctdg_artifacts(tmp_path)
    report = check_artifact_root(tmp_path, check_ctdg=True, check_dtdg=False)
    assert report.errors == []
    assert report.stats["ctdg_train_windows"] == 2
    assert report.stats["ctdg_train_events"] == 2


def test_check_ctdg_batches_reports_missing_rank_event(tmp_path) -> None:
    _write_ctdg_artifacts(tmp_path, corrupt=True)
    report = check_artifact_root(tmp_path, check_ctdg=True, check_dtdg=False)
    assert any("missing=[1]" in error for error in report.errors)


def test_check_dtdg_partition_batches_reconstruct_global_windows(tmp_path) -> None:
    _write_dtdg_artifacts(tmp_path)
    report = check_artifact_root(tmp_path, check_ctdg=False, check_dtdg=True)
    assert report.errors == []
    assert report.stats["dtdg_slices"] == 2
    assert report.stats["dtdg_expected_edges"] == 4
