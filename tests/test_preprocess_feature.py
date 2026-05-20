import torch

from atc_starrygl_lib.preprocess.feature import build_all_feature_artifacts, build_feature_artifact


def _rank_artifact(rank: int) -> dict:
    return {
        "rank": rank,
        "local_node_ids": torch.tensor([0, 2, 1]) if rank == 0 else torch.tensor([0, 2, 3]),
        "local_edge_ids": torch.tensor([0, 2]) if rank == 0 else torch.tensor([1, 3]),
    }


def test_feature_artifact_static_selection() -> None:
    node_feat = torch.arange(16, dtype=torch.float32).view(4, 4)
    edge_feat = torch.arange(20, dtype=torch.float32).view(4, 5)
    node_label = torch.arange(4, dtype=torch.float32)
    edge_label = torch.arange(4, dtype=torch.float32) + 10
    artifact = build_feature_artifact(
        rank_artifact=_rank_artifact(0),
        node_feat=node_feat,
        edge_feat=edge_feat,
        node_label=node_label,
        edge_label=edge_label,
    )
    assert artifact["format"] == "atc_feature_v1"
    assert artifact["node_ids"].tolist() == [0, 2, 1]
    assert artifact["edge_ids"].tolist() == [0, 2]
    assert torch.equal(artifact["node_feat"], node_feat[[0, 2, 1]])
    assert torch.equal(artifact["edge_feat"], edge_feat[[0, 2]])
    assert torch.equal(artifact["node_label"], node_label[[0, 2, 1]])
    assert torch.equal(artifact["edge_label"], edge_label[[0, 2]])


def test_feature_artifact_time_varying_node_selection() -> None:
    node_feat = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    node_label = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
    artifact = build_feature_artifact(
        rank_artifact=_rank_artifact(1),
        node_feat=node_feat,
        node_label=node_label,
        node_feat_time_varying=True,
        node_label_time_varying=True,
    )
    assert torch.equal(artifact["node_feat"], node_feat[:, [0, 2, 3]])
    assert torch.equal(artifact["node_label"], node_label[:, [0, 2, 3]])
    assert artifact["node_feat_time_varying"]
    assert artifact["node_label_time_varying"]


def test_build_all_feature_artifacts() -> None:
    artifacts = build_all_feature_artifacts(
        rank_artifacts=[_rank_artifact(0), _rank_artifact(1)],
        node_feat=torch.arange(8).view(4, 2),
    )
    assert [item["rank"] for item in artifacts] == [0, 1]
    assert artifacts[0]["node_feat"].shape == (3, 2)
    assert artifacts[1]["node_feat"].shape == (3, 2)
