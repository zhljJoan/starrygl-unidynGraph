import torch

from atc_starrygl_lib.preprocess.partition_data import build_all_partition_data_artifacts


def _dist_plan() -> dict:
    return {
        "node_to_chunk": torch.tensor([0, 1, 2, 3]),
        "node_master": torch.tensor([0, 0, 1, 1]),
    }


def _rank_artifacts() -> list[dict]:
    return [
        {
            "rank": 0,
            "local_node_ids": torch.tensor([0, 2, 1]),
            "local_edge_ids": torch.tensor([0, 1]),
        },
        {
            "rank": 1,
            "local_node_ids": torch.tensor([0, 2, 3]),
            "local_edge_ids": torch.tensor([2, 3]),
        },
    ]


def test_partition_data_embeds_topology_features_and_route() -> None:
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])
    node_feat = torch.arange(12, dtype=torch.float32).view(4, 3)
    edge_feat = torch.arange(8, dtype=torch.float32).view(4, 2)
    node_label = torch.arange(4, dtype=torch.float32)
    edge_weight = torch.ones(4, dtype=torch.float32)
    parts = build_all_partition_data_artifacts(
        rank_artifacts=_rank_artifacts(),
        dist_plan=_dist_plan(),
        src=src,
        dst=dst,
        time_ptr_2=torch.tensor([[0, 2], [2, 4]]),
        node_feat=node_feat,
        edge_feat=edge_feat,
        node_label=node_label,
        edge_weight=edge_weight,
    )
    p0 = parts[0]
    assert p0["format"] == "atc_partition_data_v1"
    assert p0["dst_ids"]["ptr"].tolist() == [0, 2, 2]
    assert p0["dst_ids"]["data"].tolist() == [1, 2]
    assert p0["src_ids"]["data"].tolist() == [0]
    assert p0["edge_ids"]["data"].tolist() == [0, 1]
    assert p0["edge_ptr"]["data"].tolist() == [0, 1, 2, 0]
    assert p0["dst_chunk"]["data"].tolist() == [1, 2]
    assert torch.equal(p0["node_data"]["x"]["data"], node_feat[[1, 2, 0]])
    assert torch.equal(p0["node_data"]["y"]["data"], node_label[[1, 2]])
    assert torch.equal(p0["edge_data"]["feat"]["data"], edge_feat[[0, 1]])
    assert "gcn_norm" in p0["edge_data"]
    assert p0["route"]["recv_sizes"][0] == [0, 0]


def test_partition_data_uses_active_dst_per_slice() -> None:
    parts = build_all_partition_data_artifacts(
        rank_artifacts=_rank_artifacts(),
        dist_plan=_dist_plan(),
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        time_ptr_2=torch.tensor([[0, 1], [1, 2]]),
    )
    p0 = parts[0]
    assert p0["dst_ids"]["ptr"].tolist() == [0, 1, 2]
    assert p0["dst_ids"]["data"].tolist() == [1, 2]
