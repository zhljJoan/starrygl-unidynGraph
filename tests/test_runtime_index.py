import torch

from atc_starrygl_lib.comm.dist_index import dist_index_loc, dist_index_part, encode_dist_index
from atc_starrygl_lib.runtime.index import build_feature_read_layout_from_comm


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
