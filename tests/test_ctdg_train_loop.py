from __future__ import annotations

import torch

from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.ctdg.train_loop import evaluate, train_epoch
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
