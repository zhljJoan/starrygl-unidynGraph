from __future__ import annotations

from pathlib import Path

import pytest
import torch

from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.ctdg.runtime.backend import _CTDGArtifactRuntime


def _write_minimal_ctdg_artifacts(root: Path) -> ArtifactBundle:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([1], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([0], dtype=torch.long),
        "num_nodes": 2,
        "split": torch.tensor([0], dtype=torch.uint8),
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
    }
    dist = {
        "master_dist_index": torch.tensor([0, 1], dtype=torch.long),
    }
    rank = {
        "local_node_ids": torch.tensor([0, 1], dtype=torch.long),
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "read_dist_index": torch.tensor([0, 1], dtype=torch.long),
        "split_event_pos": {
            "train": {
                "data": torch.tensor([0], dtype=torch.long),
                "ptr": torch.tensor([0, 1], dtype=torch.long),
            },
            "val": {
                "data": torch.empty(0, dtype=torch.long),
                "ptr": torch.tensor([0], dtype=torch.long),
            },
            "test": {
                "data": torch.empty(0, dtype=torch.long),
                "ptr": torch.tensor([0], dtype=torch.long),
            },
        },
        "split_time_ptr": {
            "train": torch.tensor([[0, 1]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
    }
    feature = {
        "edge_feat": torch.zeros((1, 172), dtype=torch.float32),
    }
    torch.save(graph, root / "graph.pt")
    torch.save(dist, root / "dist.pt")
    torch.save(rank, root / "rank_000.pt")
    torch.save(feature, root / "feature_000.pt")
    return ArtifactBundle(
        root=root,
        graph_mode="ctdg",
        files={
            "graph": root / "graph.pt",
            "dist": root / "dist.pt",
            "rank_000": root / "rank_000.pt",
            "feature_000": root / "feature_000.pt",
        },
    )


def test_ctdg_runtime_inferrs_mailbox_msg_dim_from_edge_features(tmp_path: Path) -> None:
    artifacts = _write_minimal_ctdg_artifacts(tmp_path)
    ctx = RuntimeContext(
        config={
            "task": {"name": "edge_prediction", "batch_size": 1},
            "model": {"name": "general"},
            "preprocess": {"batch_size": 1},
            "runtime": {
                "build_mailbox_runtime": True,
                "mailbox_size": 1,
                "memory_dim": 100,
            },
        },
        artifact_root=tmp_path,
        device="cpu",
        world_size=1,
        rank=0,
    )

    runtime = _CTDGArtifactRuntime.from_artifacts(ctx, artifacts)

    assert runtime.mailbox_runtime is not None
    assert tuple(runtime.mailbox_runtime.store.mailbox.shape) == (2, 1, 372)


def test_ctdg_runtime_rejects_stale_mailbox_msg_dim(tmp_path: Path) -> None:
    artifacts = _write_minimal_ctdg_artifacts(tmp_path)
    ctx = RuntimeContext(
        config={
            "task": {"name": "edge_prediction", "batch_size": 1},
            "model": {"name": "general"},
            "preprocess": {"batch_size": 1},
            "runtime": {
                "build_mailbox_runtime": True,
                "mailbox_size": 1,
                "memory_dim": 100,
                "mailbox_msg_dim": 200,
            },
        },
        artifact_root=tmp_path,
        device="cpu",
        world_size=1,
        rank=0,
    )

    with pytest.raises(ValueError, match="mailbox_msg_dim"):
        _CTDGArtifactRuntime.from_artifacts(ctx, artifacts)
