from __future__ import annotations

import torch

from atc_starrygl_lib.core.registry import TaskRegistry
from atc_starrygl_lib.core.types import ArtifactBundle, RegressionOutput, RuntimeContext
from atc_starrygl_lib.dtdg.runtime import FlareDTDGBackend, STGraphLoader, STGraphWindow
from atc_starrygl_lib.dtdg.train_loop import evaluate, evaluate_edge_prediction, prepare_edge_prediction_embeddings, train_epoch
from atc_starrygl_lib.models.dtdg import TGCN
from atc_starrygl_lib.models.shared import EdgePredictHead
from atc_starrygl_lib.preprocess.partition_data import build_all_partition_data_artifacts
from atc_starrygl_lib.tasks import EdgePredictionTask, NodeRegressionTask, register_builtin_tasks


class _SrcFeatureEncoder(torch.nn.Module):
    def forward(self, graph):
        return graph.srcdata["x"]


class _DstFeatureEncoder(torch.nn.Module):
    def forward(self, graph):
        return graph.dstdata["x"] if "x" in graph.dstdata else graph.srcdata["x"][: graph.num_dst_nodes()]


def _partition_data(num_slices: int = 4, *, self_loop: bool = False) -> dict:
    src = torch.arange(num_slices, dtype=torch.long) % 4
    dst = src if self_loop else (src + 1) % 4
    node_feat = torch.arange(12, dtype=torch.float32).view(4, 3)
    node_label = torch.arange(4, dtype=torch.float32).view(4, 1)
    parts = build_all_partition_data_artifacts(
        rank_artifacts=[{
            "rank": 0,
            "local_node_ids": torch.arange(4, dtype=torch.long),
            "local_edge_ids": torch.arange(num_slices, dtype=torch.long),
        }],
        dist_plan={
            "node_to_chunk": torch.tensor([0, 1, 2, 3]),
            "node_master": torch.zeros(4, dtype=torch.long),
        },
        src=src,
        dst=dst,
        time_ptr_2=torch.tensor([[i, i + 1] for i in range(num_slices)], dtype=torch.long),
        node_feat=node_feat,
        node_label=node_label,
    )
    return parts[0]


def test_stgraph_loader_fetches_snapshot_block() -> None:
    loader = STGraphLoader(
        partition_data=_partition_data(),
        device="cpu",
        rank=0,
        world_size=1,
    )

    snapshot = loader.fetch_snapshot(0)

    assert snapshot.snapshot_id == 0
    assert snapshot.graph.num_src_nodes() == 2
    assert snapshot.graph.num_dst_nodes() == 1
    assert snapshot.src_ids.tolist() == [1, 0]
    assert snapshot.dst_ids.tolist() == [1]
    assert snapshot.edge_src.tolist() == [1]
    assert snapshot.edge_dst.tolist() == [0]
    assert snapshot.graph.dstdata["y"].tolist() == [[1.0]]
    assert hasattr(snapshot.graph, "flare_fetch_state")
    assert hasattr(snapshot.graph, "flare_apply_route")
    assert snapshot.graph.route is None


def test_stgraph_loader_yields_sliding_window_items() -> None:
    loader = STGraphLoader(
        partition_data=_partition_data(),
        device="cpu",
        rank=0,
        world_size=1,
    )

    lengths = []
    latest_full = []
    for window in loader.iter_sliding_windows(
        snapshot_ids=range(2),
        chunk_order=torch.arange(loader.chunk_count),
        chunk_decay=[2],
        num_full_snapshots=1,
    ):
        assert isinstance(window, STGraphWindow)
        lengths.append(len(window))
        latest_full.append(window.latest_graph.flare_is_full_snapshot)

    assert lengths == [1, 2]
    assert latest_full == [True, True]


def test_dtdg_backend_defaults_to_node_regression_test_batches(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.5, "val_ratio": 0.25}, graph_path)
    torch.save(_partition_data(self_loop=True), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(config={}, artifact_root=tmp_path, rank=0, world_size=1, device="cpu"),
        artifacts,
    )

    batch = next(backend.iter_batches("test"))

    assert batch.split == "test"
    assert batch.graph is not None
    assert batch.labels is not None
    assert batch.node_ids is not None
    assert batch.node_ids.tolist() == [3]
    assert batch.labels.tolist() == [[3.0]]


def test_dtdg_node_regression_train_step_uses_sliding_window(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.75, "val_ratio": 0.0}, graph_path)
    torch.save(_partition_data(self_loop=True), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "chunk_order": "identity",
                    "chunk_decay": [2],
                    "num_full_snapshots": 1,
                },
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        artifacts,
    )
    model = TGCN(input_size=3, hidden_size=4, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    task = NodeRegressionTask()

    batch = next(backend.iter_batches("train"))
    assert isinstance(batch.graph, STGraphWindow)

    preds, _ = model(batch.graph)
    output = RegressionOutput(pred=preds[-1])
    loss = task.compute_loss(output, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert torch.isfinite(loss)


def test_dtdg_train_and_eval_loop_run_node_regression(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.5, "val_ratio": 0.25}, graph_path)
    torch.save(_partition_data(self_loop=True), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "chunk_order": "identity",
                    "chunk_decay": [2],
                    "num_full_snapshots": 1,
                },
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        artifacts,
    )
    model = TGCN(input_size=3, hidden_size=4, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    task = NodeRegressionTask()

    train_metrics = train_epoch(backend, model, task, optimizer)
    test_metrics = evaluate(backend, model, task, split="test")

    assert train_metrics["loss"] >= 0.0
    assert test_metrics["loss"] >= 0.0
    assert "mse" in test_metrics


def test_dtdg_backend_yields_edge_predict_test_batches(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.5, "val_ratio": 0.25}, graph_path)
    torch.save(_partition_data(), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(
            config={"task": {"name": "edge_predict"}, "runtime": {"negative_ratio": 1}},
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        artifacts,
    )

    batch = next(backend.iter_batches("test"))

    assert batch.split == "test"
    assert batch.graph is not None
    assert batch.eids is not None
    assert batch.pos_src is not None
    assert batch.pos_dst is not None
    assert batch.neg_dst is not None
    assert batch.src is not None and batch.dst is not None
    assert batch.roots.tolist() == batch.graph.srcdata["ID"].tolist()


def test_dtdg_edge_predict_test_loop_runs(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.5, "val_ratio": 0.25}, graph_path)
    torch.save(_partition_data(), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(
            config={"task": {"name": "edge_predict"}, "runtime": {"negative_ratio": 1}},
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        artifacts,
    )

    metrics = evaluate_edge_prediction(
        backend,
        _SrcFeatureEncoder(),
        EdgePredictHead(dim=3),
        EdgePredictionTask(),
        split="test",
    )

    assert metrics["loss"] >= 0.0
    assert "ap" in metrics
    assert "auc" in metrics


def test_dtdg_edge_predict_expands_dst_only_embeddings(tmp_path) -> None:
    graph_path = tmp_path / "graph.pt"
    part_path = tmp_path / "partition_data_000.pt"
    torch.save({"train_ratio": 0.5, "val_ratio": 0.25}, graph_path)
    torch.save(_partition_data(), part_path)
    artifacts = ArtifactBundle(
        root=tmp_path,
        graph_mode="dtdg",
        files={"graph": graph_path, "partition_data_000": part_path},
    )
    backend = FlareDTDGBackend()
    backend.build_runtime(
        RuntimeContext(
            config={"task": {"name": "edge_predict"}, "runtime": {"negative_ratio": 1}},
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        artifacts,
    )
    batch = next(backend.iter_batches("test"))
    dst_embeddings = batch.graph.srcdata["x"][: batch.graph.num_dst_nodes()]

    expanded = prepare_edge_prediction_embeddings(dst_embeddings, batch)

    assert expanded.size(0) == batch.graph.num_src_nodes()
    assert torch.equal(expanded[batch.pos_dst], dst_embeddings[batch.pos_dst])
    assert batch.pos_src.max().item() < expanded.size(0)

    metrics = evaluate_edge_prediction(
        backend,
        _DstFeatureEncoder(),
        EdgePredictHead(dim=3),
        EdgePredictionTask(),
        split="test",
    )
    assert metrics["loss"] >= 0.0


def test_edge_predict_task_alias_is_registered() -> None:
    register_builtin_tasks()
    assert TaskRegistry.get("edge_predict") is EdgePredictionTask
    assert TaskRegistry.get("link_prediction") is EdgePredictionTask
