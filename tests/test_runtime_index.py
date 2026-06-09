import torch

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part, encode_dist_index
from atc_starrygl_lib.comm.dynamic import DynamicFetchComm
from atc_starrygl_lib.features.runtime import CTDGFeatureRuntime
from atc_starrygl_lib.features.store import FeatureStore
from atc_starrygl_lib.runtime.index import DistIndexTables
from atc_starrygl_lib.runtime.index import build_feature_read_layout_from_comm
from atc_starrygl_lib.sampling.temporal import EdgeCommLayout, SamplingOutput


def test_feature_read_layout_orders_by_rank_then_local_row() -> None:
    read_dist_index = encode_dist_index(
        torch.tensor([5, 3, 1, 2], dtype=torch.long),
        torch.tensor([1, 0, 0, 1], dtype=torch.long),
    )
    layout = build_feature_read_layout_from_comm(
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.arange(4, dtype=torch.long),
        read_dist_index,
        world_size=2,
        time_slices=torch.tensor([50, 30, 10, 20], dtype=torch.long),
    )

    assert dist_index_part(layout.read_index).tolist() == [0, 0, 1, 1]
    assert dist_index_loc(layout.read_index).tolist() == [1, 3, 2, 5]
    assert layout.read_ptr.tolist() == [0, 2, 4]
    assert layout.compute_to_feature.tolist() == [3, 1, 0, 2]
    assert layout.time_slices.tolist() == [10, 30, 20, 50]


def test_ctdg_edge_feature_runtime_uses_native_direct_layout() -> None:
    read_index = encode_dist_index(
        torch.tensor([1, 3, 2], dtype=torch.long),
        torch.tensor([0, 0, 1], dtype=torch.long),
    )
    output = SamplingOutput(
        mfgs=[],
        node_compute=None,  # type: ignore[arg-type]
        edge_compute=None,
        node_comm=None,  # type: ignore[arg-type]
        edge_comm=EdgeCommLayout(
            edge_gids=torch.tensor([10, 11, 12], dtype=torch.long),
            time_slices=None,
            owner=None,
            provider=None,
            provider_ptr=None,
            compute_to_comm=torch.arange(3, dtype=torch.long),
            read_index=read_index,
            read_ptr=torch.tensor([0, 2, 3], dtype=torch.long),
            compute_to_feature=torch.tensor([0, 1, 2], dtype=torch.long),
        ),
    )
    runtime = CTDGFeatureRuntime(
        index=DistIndexTables(master_dist_index=torch.empty(0, dtype=torch.long), read_dist_index=torch.empty(0, dtype=torch.long)),
        feature_store=FeatureStore(edge_features=torch.empty((0, 1))),
        comm=DynamicFetchComm("cpu"),
        world_size=2,
        edge_dist_index=None,
    )

    layout = runtime.build_edge_layout_from_sampling(output)

    assert layout is not None
    assert torch.equal(layout.read_index, read_index)
    assert layout.read_ptr.tolist() == [0, 2, 3]
    assert layout.compute_to_feature.tolist() == [0, 1, 2]


def test_feature_store_row_map_preserves_logical_rows() -> None:
    features = torch.tensor([[20.0], [10.0], [30.0]])
    row_map = torch.tensor([1, 0, 2], dtype=torch.long)
    store = FeatureStore(node_features=features, node_row_map=row_map)

    out = store.gather_node_rows(torch.tensor([0, 1, 2], dtype=torch.long))

    assert out.tolist() == [[10.0], [20.0], [30.0]]


def test_feature_store_sorted_mapped_gather_restores_request_order() -> None:
    features = torch.tensor([[20.0], [10.0], [30.0]])
    row_map = torch.tensor([1, 0, 2], dtype=torch.long)
    store = FeatureStore(node_features=features, node_row_map=row_map, sort_mapped_gather=True)

    out = store.gather_node_rows(torch.tensor([2, 0, 1], dtype=torch.long))

    assert out.tolist() == [[30.0], [10.0], [20.0]]


def test_feature_store_pinned_transfer_keeps_cpu_requests_on_cpu() -> None:
    features = torch.tensor([[1.0], [2.0]])
    store = FeatureStore(node_features=features, pin_memory_transfer=True)

    out = store.gather_node_rows(torch.tensor([1, 0], dtype=torch.long))

    assert out.device.type == "cpu"
    assert out.tolist() == [[2.0], [1.0]]
