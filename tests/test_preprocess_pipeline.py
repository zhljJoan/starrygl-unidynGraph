from pathlib import Path

import torch

from atc_starrygl_lib.preprocess.pipeline import run_preprocess_pipeline


def test_pipeline_forwards_split_time_ptr_to_rank_builder(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_build_dataset(**kwargs):
        return {
            "src": torch.tensor([0, 1, 2], dtype=torch.long),
            "dst": torch.tensor([1, 2, 3], dtype=torch.long),
            "ts": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "num_nodes": 4,
            "edge_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "time_ptr_2": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
            "split": torch.tensor([0, 1, 2], dtype=torch.uint8),
            "split_time_ptr": {
                "train": torch.tensor([[0, 1]], dtype=torch.long),
                "val": torch.tensor([[1, 2]], dtype=torch.long),
                "test": torch.tensor([[2, 3]], dtype=torch.long),
            },
        }

    def fake_build_dist_plan(**kwargs):
        return {"world_size": 1, "num_nodes": 4}

    def fake_build_all_rank_artifacts(**kwargs):
        captured["split_time_ptr"] = kwargs.get("split_time_ptr")
        return kwargs["dist_plan"], [{"rank": 0}]

    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.build_dataset", fake_build_dataset)
    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.build_dist_plan", fake_build_dist_plan)
    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.build_all_rank_artifacts", fake_build_all_rank_artifacts)
    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.build_all_feature_artifacts", lambda **kwargs: [])
    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.build_all_partition_data_artifacts", lambda **kwargs: [])

    run_preprocess_pipeline(
        data="dummy",
        out_dir=tmp_path,
        world_size=1,
        algorithm="speed_partition",
        chunks_per_rank=1,
        mode="event",
        build_feature=False,
        build_partition_data=False,
    )

    assert captured["split_time_ptr"] is not None
    assert captured["split_time_ptr"]["train"].tolist() == [[0, 1]]
    assert captured["split_time_ptr"]["val"].tolist() == [[1, 2]]
    assert captured["split_time_ptr"]["test"].tolist() == [[2, 3]]
