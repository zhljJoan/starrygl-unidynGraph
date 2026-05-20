from pathlib import Path

import torch

from atc_starrygl_lib.preprocess.dataset import build_dataset


def test_event_dict_stable_sort_and_reorder() -> None:
    data = {
        "src": torch.tensor([2, 0, 1]),
        "dst": torch.tensor([3, 1, 2]),
        "ts": torch.tensor([2.0, 1.0, 2.0]),
        "edge_feat": torch.tensor([[20.0], [10.0], [30.0]]),
        "edge_weight": torch.tensor([2.0, 1.0, 3.0]),
    }
    out = build_dataset(data=data, mode="event", batch_size=2)
    assert out["src"].tolist() == [0, 2, 1]
    assert out["dst"].tolist() == [1, 3, 2]
    assert out["edge_feat"].squeeze(-1).tolist() == [10.0, 20.0, 30.0]
    assert out["edge_weight"].tolist() == [1.0, 2.0, 3.0]
    assert out["time_ptr_2"].tolist() == [[0, 2], [2, 3]]


def test_event_split_then_batch_with_per_split_configs() -> None:
    data = {
        "src": torch.arange(10),
        "dst": torch.arange(10) + 1,
        "ts": torch.arange(10).float(),
    }
    out = build_dataset(
        data=data,
        mode="event",
        train_ratio=0.6,
        val_ratio=0.2,
        train_batch_size=2,
        val_num_windows=2,
        test_batch_size=1,
    )
    assert out["split"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    assert out["split_time_ptr"]["train"].tolist() == [[0, 2], [2, 4], [4, 6]]
    assert out["split_time_ptr"]["val"].tolist() == [[6, 7], [7, 8]]
    assert out["split_time_ptr"]["test"].tolist() == [[8, 9], [9, 10]]
    assert out["time_ptr_2"].tolist() == [[0, 2], [2, 4], [4, 6], [6, 7], [7, 8], [8, 9], [9, 10]]


def test_snapshot_list_build_overlapped_windows() -> None:
    data = {
        "snapshots": [
            {"src": [0, 1], "dst": [1, 2]},
            {"src": [2], "dst": [0]},
            {"src": [1, 0], "dst": [0, 2]},
        ]
    }
    out = build_dataset(data=data, mode="snapshot", lags=2)
    assert out["snapshot_ptr"].tolist() == [0, 2, 3, 5]
    assert out["time_ptr_2"].tolist() == [[0, 2], [0, 3], [2, 5]]


def test_read_pth_and_edges_files(tmp_path: Path) -> None:
    pth = tmp_path / "toy.pth"
    torch.save({"src": torch.tensor([0, 1]), "dst": torch.tensor([1, 2]), "ts": torch.tensor([2.0, 1.0])}, pth)
    out_pth = build_dataset(data=pth, mode="event", num_windows=2)
    assert out_pth["ts"].tolist() == [1.0, 2.0]
    assert out_pth["time_ptr_2"].tolist() == [[0, 1], [1, 2]]

    edges_csv = tmp_path / "edges.csv"
    edges_csv.write_text("src,dst,ts\n1,2,3\n0,1,1\n", encoding="utf-8")
    out_csv = build_dataset(data=edges_csv, mode="event", batch_size=10)
    assert out_csv["src"].tolist() == [0, 1]

    ds_dir = tmp_path / "toy"
    ds_dir.mkdir()
    edges_file = ds_dir / "toy.edges"
    edges_file.write_text("2 3 2\n1 2 1\n", encoding="utf-8")
    out_edges = build_dataset(data=edges_file, mode="event", batch_size=10)
    assert out_edges["dst"].tolist() == [2, 3]
