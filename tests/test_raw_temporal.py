import torch

from starry_unigraph.data.raw_temporal import RawTemporalEvents, build_snapshot_dataset_from_events


def test_snapshot_degree_labels_use_next_snapshot():
    events = RawTemporalEvents(
        src=torch.tensor([0, 2, 1], dtype=torch.long),
        dst=torch.tensor([1, 1, 2], dtype=torch.long),
        ts=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        weight=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32),
        edge_feat=torch.ones(3, 1, dtype=torch.float32),
        num_nodes=3,
        num_edges=3,
        source="unit",
    )

    data = build_snapshot_dataset_from_events(events, snaps=3)
    snapshots = data["dataset"]

    assert torch.allclose(snapshots[0]["x"], torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))
    assert torch.allclose(snapshots[0]["y"], torch.log(torch.tensor([1.0, 3.0, 1.0])))
    assert torch.allclose(snapshots[1]["y"], torch.log(torch.tensor([1.0, 1.0, 2.0])))
    assert snapshots[2]["y"] is None
