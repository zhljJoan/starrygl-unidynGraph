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


def test_rank_artifacts_collapse_repeated_hot_node_updates_within_slice() -> None:
    plan = _dist_plan()
    plan["num_edges"] = 3
    plan["edge_owner"] = torch.tensor([0, 0, 0])
    plan["edge_ids_by_part"] = [
        torch.tensor([0, 1, 2]),
        torch.empty(0, dtype=torch.long),
    ]
    dist, ranks = build_all_rank_artifacts(
        dist_plan=plan,
        src=torch.tensor([1, 1, 0]),
        dst=torch.tensor([0, 0, 1]),
        ts=torch.tensor([1.0, 2.0, 3.0]),
        time_ptr_2=torch.tensor([[0, 3]]),
    )
    r0, r1 = ranks

    # Rank 0 owns all three local events in this slice. Node 0 is a replicated hot
    # node and appears in every event, but the preprocess update contract keeps only
    # one update row per node per slice and records the latest timestamp.
    assert r0["local_edge_ids"].tolist() == [0, 1, 2]
    assert r0["update_node_ptr"].tolist() == [0, 2]
    assert r0["update_node_ids"].tolist() == [0, 1]
    assert r0["update_node_ts"].tolist() == [3.0, 3.0]

    route0 = r0["memory_route"]
    assert route0["send_ptr"].tolist() == [0, 1]
    assert route0["send_update_pos"].tolist() == [0]
    assert route0["send_rank"].tolist() == [1]

    route1 = r1["memory_route"]
    assert route1["recv_ptr"].tolist() == [0, 1]
    assert route1["recv_rank"].tolist() == [0]
    assert route1["recv_local_row"].tolist() == [0]

    # The shared node update arriving at rank 1 is likewise collapsed to one row,
    # so no per-event incremental state is preserved across the three hits.
    assert dist["edge_dist_index"].numel() == 3


def test_rank_artifacts_preserve_repeated_hot_node_updates_when_enabled() -> None:
    plan = _dist_plan()
    plan["num_edges"] = 3
    plan["edge_owner"] = torch.tensor([0, 0, 0])
    plan["edge_ids_by_part"] = [
        torch.tensor([0, 1, 2]),
        torch.empty(0, dtype=torch.long),
    ]
    _, ranks = build_all_rank_artifacts(
        dist_plan=plan,
        src=torch.tensor([1, 1, 0]),
        dst=torch.tensor([0, 0, 1]),
        ts=torch.tensor([1.0, 2.0, 3.0]),
        time_ptr_2=torch.tensor([[0, 3]]),
        preserve_replica_history=True,
    )
    r0, r1 = ranks

    assert r0["preserve_replica_history"] is True
    assert r0["update_node_ptr"].tolist() == [0, 6]
    assert r0["update_node_ids"].tolist() == [1, 1, 0, 0, 0, 1]
    assert r0["update_node_ts"].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    route0 = r0["memory_route"]
    assert route0["send_ptr"].tolist() == [0, 3]
    assert route0["send_update_pos"].tolist() == [2, 3, 4]
    assert route0["send_rank"].tolist() == [1, 1, 1]

    route1 = r1["memory_route"]
    assert route1["recv_ptr"].tolist() == [0, 3]
    assert route1["recv_rank"].tolist() == [0, 0, 0]
    assert route1["recv_local_row"].tolist() == [0, 0, 0]


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
    assert r0["split_event_pos"]["train"]["data"].tolist() == [0, 1]
    assert r0["split_event_pos"]["train"]["ptr"].tolist() == [0, 1, 2]
    assert r0["split_event_pos"]["val"]["data"].tolist() == []
    assert r0["split_event_pos"]["val"]["ptr"].tolist() == [0, 0]
    assert r0["split_event_pos"]["test"]["data"].tolist() == []
    assert r0["split_event_pos"]["test"]["ptr"].tolist() == [0, 0]
    assert r0["split_time_ptr"]["train"].tolist() == [[0, 1], [1, 2]]
    assert r0["split_time_ptr"]["val"].tolist() == [[0, 0]]
    assert r0["split_time_ptr"]["test"].tolist() == [[0, 0]]
    assert r1["local_edge_ids"].tolist() == [2, 3]
    assert r1["split_event_pos"]["train"]["data"].tolist() == []
    assert r1["split_event_pos"]["train"]["ptr"].tolist() == [0, 0, 0]
    assert r1["split_event_pos"]["val"]["data"].tolist() == [2]
    assert r1["split_event_pos"]["val"]["ptr"].tolist() == [0, 1]
    assert r1["split_event_pos"]["test"]["data"].tolist() == [3]
    assert r1["split_event_pos"]["test"]["ptr"].tolist() == [0, 1]
    assert r1["split_time_ptr"]["train"].tolist() == [[0, 0], [0, 0]]
    assert r1["split_time_ptr"]["val"].tolist() == [[0, 1]]
    assert r1["split_time_ptr"]["test"].tolist() == [[0, 1]]


def test_rank_artifacts_include_native_layout_missing_isolated_master_nodes() -> None:
    plan = _dist_plan()
    plan["num_nodes"] = 5
    plan["node_to_chunk"] = torch.tensor([0, 1, 2, 3, 0])
    plan["node_master"] = torch.tensor([0, 0, 1, 1, 0])
    plan["node_owner"] = torch.tensor([0, 0, 1, 1, 0])
    plan["replica_mask"] = torch.tensor([True, False, True, False, False])
    plan["local_node_ids_by_part"] = [
        torch.tensor([0, 2, 1]),
        torch.tensor([0, 2, 3]),
    ]
    plan["replica_node_ids_by_part"] = [
        torch.tensor([0, 2]),
        torch.tensor([0, 2]),
    ]
    plan["owned_node_ids_by_part"] = [
        torch.tensor([1]),
        torch.tensor([3]),
    ]

    dist, ranks = build_all_rank_artifacts(
        dist_plan=plan,
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        ts=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        time_ptr_2=torch.tensor([[0, 2], [2, 4]]),
    )

    r0 = ranks[0]
    assert r0["local_node_ids"].tolist() == [0, 2, 1, 4]
    assert r0["owned_count"] == 2
    assert dist_index_part(dist["master_dist_index"][4:5]).item() == 0
    assert dist_index_loc(dist["master_dist_index"][4:5]).item() == 3
    assert dist_index_loc(r0["read_dist_index"][4:5]).item() == 3
