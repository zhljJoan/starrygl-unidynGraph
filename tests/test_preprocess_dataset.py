from pathlib import Path

import torch

from atc_starrygl_lib.preprocess.dataset import build_dataset
from atc_starrygl_lib.preprocess.dataset import _refine_adaptive_time_ptr


def test_event_dict_stable_sort_and_reorder() -> None:
    data = {
        "src": torch.tensor([2, 0, 1]),
        "dst": torch.tensor([3, 1, 2]),
        "ts": torch.tensor([2.0, 1.0, 2.0]),
        "edge_feat": torch.tensor([[20.0], [10.0], [30.0]]),
        "edge_weight": torch.tensor([2.0, 1.0, 3.0]),
        "node_label_nodes": torch.tensor([4, 5]),
        "node_label_ts": torch.tensor([7.0, 8.0]),
        "node_label": torch.tensor([1, 0]),
        "node_label_split": torch.tensor([0, 1], dtype=torch.uint8),
    }
    out = build_dataset(data=data, mode="event", batch_size=2)
    assert out["src"].tolist() == [0, 2, 1]
    assert out["dst"].tolist() == [1, 3, 2]
    assert out["edge_feat"].squeeze(-1).tolist() == [10.0, 20.0, 30.0]
    assert out["edge_weight"].tolist() == [1.0, 2.0, 3.0]
    assert out["node_label_nodes"].tolist() == [4, 5]
    assert out["node_label_ts"].tolist() == [7.0, 8.0]
    assert out["node_label"].tolist() == [1, 0]
    assert out["node_label_split"].tolist() == [0, 1]
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


def test_event_adaptive_split_mode_builds_split_time_ptr() -> None:
    data = {
        "src": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "dst": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        "ts": torch.arange(10).float(),
    }

    out = build_dataset(
        data=data,
        mode="event",
        train_ratio=1.0,
        val_ratio=0.0,
        batch_size=3,
        split_mode="adaptive",
        adaptive_split_fallback=False,
    )

    assert out["split_time_ptr"]["train"].tolist() == [[0, 3], [3, 6], [6, 10]]
    assert out["time_ptr_2"].tolist() == [[0, 3], [3, 6], [6, 10]]


def test_adaptive_split_refinement_bounds_large_windows() -> None:
    ptr = torch.tensor([[0, 10]], dtype=torch.long)
    src = torch.arange(10, dtype=torch.long)

    out = _refine_adaptive_time_ptr(
        ptr,
        src=src,
        min_batch_size=3,
        max_batch_size=4,
        coherence_chunks=0,
        max_chunk_entropy_ratio=None,
    )

    assert out.tolist() == [[0, 4], [4, 8], [8, 10]]


def test_adaptive_split_refinement_can_cut_on_chunk_entropy() -> None:
    ptr = torch.tensor([[0, 8]], dtype=torch.long)
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

    out = _refine_adaptive_time_ptr(
        ptr,
        src=src,
        min_batch_size=4,
        max_batch_size=None,
        coherence_chunks=4,
        max_chunk_entropy_ratio=0.75,
    )

    assert out.tolist() == [[0, 4], [4, 8]]


def test_event_can_generate_deterministic_random_node_features() -> None:
    data = {
        "src": torch.tensor([0, 1]),
        "dst": torch.tensor([1, 2]),
        "ts": torch.tensor([0.0, 1.0]),
        "num_nodes": 4,
    }

    first = build_dataset(data=data, mode="event", random_node_feat_dim=3, random_node_feat_seed=7)
    second = build_dataset(data=data, mode="event", random_node_feat_dim=3, random_node_feat_seed=7)

    assert first["node_feat"].shape == (4, 3)
    assert torch.equal(first["node_feat"], second["node_feat"])


def test_event_can_generate_deterministic_random_edge_features() -> None:
    data = {
        "src": torch.tensor([0, 1]),
        "dst": torch.tensor([1, 2]),
        "ts": torch.tensor([0.0, 1.0]),
    }

    first = build_dataset(data=data, mode="event", random_edge_feat_dim=172, random_edge_feat_seed=11)
    second = build_dataset(data=data, mode="event", random_edge_feat_dim=172, random_edge_feat_seed=11)

    assert first["edge_feat"].shape == (2, 172)
    assert torch.equal(first["edge_feat"], second["edge_feat"])


def test_event_can_generate_bts_style_sequential_random_features() -> None:
    data = {
        "src": torch.tensor([0, 1]),
        "dst": torch.tensor([1, 2]),
        "ts": torch.tensor([0.0, 1.0]),
        "num_nodes": 4,
    }
    gen = torch.Generator()
    gen.manual_seed(7)
    expected_node = torch.randn((4, 3), generator=gen, dtype=torch.float32)
    expected_edge = torch.randn((2, 5), generator=gen, dtype=torch.float32)

    out = build_dataset(
        data=data,
        mode="event",
        random_node_feat_dim=3,
        random_node_feat_seed=7,
        random_edge_feat_dim=5,
        random_edge_feat_seed=7,
        random_feature_seed_mode="bts",
    )

    assert torch.equal(out["node_feat"], expected_node)
    assert torch.equal(out["edge_feat"], expected_edge)


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


def test_flare_dataset_list_keeps_time_varying_node_tensors() -> None:
    data = {
        "num_nodes": 3,
        "node_feat_source": "degree",
        "node_label_source": "log_in_degree",
        "dataset": [
            {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "edge_weight": torch.tensor([2.0, 3.0]),
                "x": torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 1.0]]),
                "y": torch.tensor([0.0, 1.0, 2.0]),
            },
            {
                "edge_index": torch.tensor([[2], [0]]),
                "edge_weight": torch.tensor([4.0]),
                "x": torch.tensor([[4.0, 1.0], [0.0, 0.0], [0.0, 4.0]]),
                "y": torch.tensor([3.0, 0.0, 1.0]),
            },
            {
                "edge_index": torch.tensor([[0], [2]]),
                "edge_weight": torch.tensor([5.0]),
                "x": torch.ones(3, 2),
                "y": None,
            },
        ],
    }

    out = build_dataset(data=data, mode="snapshot", lags=1)

    assert out["snapshot_ptr"].tolist() == [0, 2, 3]
    assert out["time_ptr_2"].tolist() == [[0, 2], [2, 3]]
    assert out["node_feat_time_varying"] is True
    assert out["node_label_time_varying"] is True
    assert out["node_feat_source"] == "degree"
    assert out["node_label_source"] == "log_in_degree"
    assert out["node_feat"].shape == (2, 3, 2)
    assert out["node_label"].shape == (2, 3)
    assert out["edge_weight"].tolist() == [2.0, 3.0, 4.0]


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


def test_read_raw_dataset_directory_features_and_labels(tmp_path: Path) -> None:
    ds_dir = tmp_path / "WIKI"
    ds_dir.mkdir()
    (ds_dir / "edges.csv").write_text(
        "src,dst,time,int_roll,ext_roll\n1,3,2.0,0,0\n0,2,1.0,1,0\n",
        encoding="utf-8",
    )
    torch.save(torch.tensor([[10.0], [20.0]]), ds_dir / "edge_features.pt")
    (ds_dir / "labels.csv").write_text(
        "node,time,label,int_roll\n1,2.0,0,0\n0,3.0,1,2\n",
        encoding="utf-8",
    )

    out = build_dataset(data=ds_dir, mode="event", batch_size=10)

    assert out["src"].tolist() == [0, 1]
    assert out["edge_feat"].squeeze(-1).tolist() == [20.0, 10.0]
    assert out["node_label_nodes"].tolist() == [1, 0]
    assert out["node_label"].tolist() == [0, 1]
    assert out["node_label_split"].tolist() == [0, 2]


def test_edges_csv_ext_roll_drives_event_split(tmp_path: Path) -> None:
    ds_dir = tmp_path / "WIKI"
    ds_dir.mkdir()
    (ds_dir / "edges.csv").write_text(
        "src,dst,time,ext_roll\n1,3,2.0,2\n0,2,1.0,1\n",
        encoding="utf-8",
    )

    out = build_dataset(data=ds_dir, mode="event", batch_size=10)

    assert out["src"].tolist() == [0, 1]
    assert out["split"].tolist() == [1, 2]
    assert out["split_time_ptr"]["train"].tolist() == []
    assert out["split_time_ptr"]["val"].tolist() == [[0, 1]]
    assert out["split_time_ptr"]["test"].tolist() == [[1, 2]]


def test_read_weighted_edges_uses_last_column_as_time(tmp_path: Path) -> None:
    path = tmp_path / "ratings.edges"
    path.write_text("% bip weighted\n1 2 5.0 20\n1 3 1.0 10\n", encoding="utf-8")

    out = build_dataset(data=path, mode="event", batch_size=10)

    assert out["dst"].tolist() == [3, 2]
    assert out["ts"].tolist() == [10.0, 20.0]
    assert out["edge_label"].tolist() == [1.0, 5.0]
