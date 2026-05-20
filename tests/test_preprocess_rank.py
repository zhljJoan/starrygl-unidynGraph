import torch

from atc_starrygl_lib.comm.dist_index import dist_index_is_cached, dist_index_is_shared, dist_index_loc, dist_index_part
from atc_starrygl_lib.preprocess.rank import build_all_rank_artifacts, build_local_chunk_view


def _dist_plan() -> dict:
    return {
        "world_size": 2,
        "num_nodes": 4,
        "num_edges": 4,
        "node_to_chunk": torch.tensor([0, 1, 2, 3]),
        "chunk_owner": torch.tensor([0, 0, 1, 1]),
        "node_master": torch.tensor([0, 0, 1, 1]),
        "node_owner": torch.tensor([0, 0, 1, 1]),
        "edge_owner": torch.tensor([0, 0, 1, 1]),
        "replica_mask": torch.tensor([True, False, True, False]),
        "replica_nodes": torch.tensor([0, 2]),
        "hot_node_ids": torch.tensor([0, 2]),
        "local_node_ids_by_part": [
            torch.tensor([0, 2, 1]),
            torch.tensor([0, 2, 3]),
        ],
        "replica_node_ids_by_part": [
            torch.tensor([0, 2]),
            torch.tensor([0, 2]),
        ],
        "owned_node_ids_by_part": [
            torch.tensor([1]),
            torch.tensor([3]),
        ],
        "shadow_node_ids_by_part": [
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ],
        "edge_ids_by_part": [
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        ],
    }


def test_local_chunk_view_excludes_shadow() -> None:
    local_chunk_ids, node_to_chunk, ptr, nodes = build_local_chunk_view(
        local_node_ids=torch.tensor([0, 2, 1, 3]),
        node_to_chunk=torch.tensor([0, 2, 1, 3]),
        replica_count=2,
        owned_count=1,
    )
    assert local_chunk_ids.tolist() == [0, 1, 2]
    assert node_to_chunk.tolist() == [0, 1, 2, -1]
    assert ptr.tolist() == [0, 1, 2, 3]
    assert nodes.tolist() == [0, 1, 2]


def test_rank_artifacts_encode_layout_and_update_contract() -> None:
    dist, ranks = build_all_rank_artifacts(
        dist_plan=_dist_plan(),
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        ts=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        time_ptr_2=torch.tensor([[0, 2], [2, 4]]),
    )
    r0, r1 = ranks
    assert r0["local_node_ids"].tolist() == [0, 2, 1]
    assert r0["replica_count"] == 2
    assert r0["owned_count"] == 1
    assert r0["shadow_count"] == 0
    assert r0["local_edge_ids"].tolist() == [0, 1]

    read0 = r0["read_dist_index"]
    assert dist_index_part(read0[0:1]).item() == 0
    assert dist_index_loc(read0[0:1]).item() == 0
    assert dist_index_is_shared(read0[0:1]).item()
    assert not dist_index_is_cached(read0[0:1]).item()

    assert dist["master_dist_index"].numel() == 4
    assert r0["update_node_ptr"].tolist() == [0, 3, 3]
    assert r0["update_node_ids"].tolist() == [0, 1, 2]
    assert r0["update_node_ts"].tolist() == [1.0, 2.0, 2.0]
    assert r0["update_local_row"].tolist() == [0, 2, 1]

    route0 = r0["memory_route"]
    assert route0["send_ptr"].tolist() == [0, 2, 2]
    assert route0["send_update_pos"].tolist() == [0, 2]
    assert route0["send_rank"].tolist() == [1, 1]
    assert route0["send_local_row"].tolist() == [0, 1]

    route1 = r1["memory_route"]
    assert route1["recv_ptr"].tolist() == [0, 2, 2]
    assert route1["recv_rank"].tolist() == [0, 0]
    assert route1["recv_local_row"].tolist() == [0, 1]


def test_rank_artifacts_build_local_split_time_ptr() -> None:
    _, ranks = build_all_rank_artifacts(
        dist_plan=_dist_plan(),
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        ts=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        time_ptr_2=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
        split_time_ptr={
            "train": torch.tensor([[0, 1], [1, 2]]),
            "val": torch.tensor([[2, 3]]),
            "test": torch.tensor([[3, 4]]),
        },
    )
    r0, r1 = ranks
    assert r0["local_edge_ids"].tolist() == [0, 1]
    assert r0["split_time_ptr"]["train"].tolist() == [[0, 1], [1, 2]]
    assert r0["split_time_ptr"]["val"].tolist() == [[0, 0]]
    assert r0["split_time_ptr"]["test"].tolist() == [[0, 0]]
    assert r1["local_edge_ids"].tolist() == [2, 3]
    assert r1["split_time_ptr"]["train"].tolist() == [[0, 0], [0, 0]]
    assert r1["split_time_ptr"]["val"].tolist() == [[0, 1]]
    assert r1["split_time_ptr"]["test"].tolist() == [[0, 1]]
