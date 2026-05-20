from __future__ import annotations

import torch

from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook, evaluate, train_epoch
from atc_starrygl_lib.models.shared import EdgePredictHead
from atc_starrygl_lib.tasks import EdgePredictionTask


def test_ctdg_train_epoch_uses_encode_head_and_task_loss() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph="mfg",
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    session = _FakeSession([batch])
    encoder = _FakeEncoder(num_rows=3, dim=4)
    head = EdgePredictHead(dim=4)
    task = EdgePredictionTask()
    opt = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.01)

    result = train_epoch(session, encoder, head, task, opt)

    assert encoder.seen_graphs == ["mfg"]
    assert result["loss"] > 0
    assert "auc" in result
    assert "ap" in result


def test_ctdg_evaluate_does_not_require_task_adapter_wrapper() -> None:
    batch = Batch(
        split="val",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph="eval_mfg",
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    session = _FakeSession([batch])
    encoder = _FakeEncoder(num_rows=3, dim=4)
    head = EdgePredictHead(dim=4)

    result = evaluate(session, encoder, head, EdgePredictionTask(), split="val")

    assert encoder.seen_graphs == ["eval_mfg"]
    assert result["loss"] > 0


def test_ctdg_train_epoch_can_commit_memory_after_step() -> None:
    graph = _FakeBlock(edge_feat=torch.tensor([[5.0]], dtype=torch.float32))
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph=graph,
        src=torch.tensor([10], dtype=torch.long),
        dst=torch.tensor([11], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    session = _FakeSession([batch])
    encoder = _FakeEncoder(num_rows=3, dim=4)
    updater = _FakeMemoryUpdater()
    encoder.memory_updater = updater
    head = EdgePredictHead(dim=4)
    opt = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.01)

    train_epoch(session, encoder, head, EdgePredictionTask(), opt, memory_commit=CTDGMemoryCommitHook())

    assert len(updater.specs) == 1
    spec = updater.specs[0]
    assert spec.src.tolist() == [10]
    assert spec.dst.tolist() == [11]
    assert spec.ts.tolist() == [3.0]
    assert spec.edge_feat.tolist() == [[5.0]]
    assert updater.handles[0].applied


class _FakeSession:
    def __init__(self, batches: list[Batch]) -> None:
        self.batches = batches

    def iter_batches(self, split: str):
        for batch in self.batches:
            assert batch.split == split
            yield batch


class _FakeEncoder(torch.nn.Module):
    def __init__(self, *, num_rows: int, dim: int) -> None:
        super().__init__()
        self.emb = torch.nn.Parameter(torch.randn(num_rows, dim))
        self.seen_graphs: list[object] = []

    def encode(self, mfgs):
        self.seen_graphs.append(mfgs)
        return self.emb


class _FakeBlock:
    def __init__(self, edge_feat: torch.Tensor) -> None:
        self.edata = {"f": edge_feat}


class _FakeCommitHandle:
    def __init__(self) -> None:
        self.applied = False

    def wait_apply(self) -> None:
        self.applied = True


class _FakeMemoryUpdater:
    def __init__(self) -> None:
        self.specs = []
        self.handles = []

    def submit_commit(self, spec):
        handle = _FakeCommitHandle()
        self.specs.append(spec)
        self.handles.append(handle)
        return handle
