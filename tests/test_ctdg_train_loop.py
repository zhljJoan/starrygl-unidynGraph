from __future__ import annotations

import torch

from atc_starrygl_lib.core.types import Batch, EdgePredOutput
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook, evaluate, train_epoch
from atc_starrygl_lib.memory import AsyncMemoryUpdateSpec, ReplicaPushIndex, RuntimeAsyncMemoryUpdater
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


def test_edge_prediction_task_uses_negative_weights() -> None:
    task = EdgePredictionTask()
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        neg_weight=torch.tensor([2.0, 0.5], dtype=torch.float32),
    )
    output = EdgePredOutput(
        pos_score=torch.tensor([0.3], dtype=torch.float32),
        neg_score=torch.tensor([-0.2, 0.7], dtype=torch.float32),
    )

    loss = task.compute_loss(output, batch)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.3], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-0.2, 0.7], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        weight=torch.tensor([2.0, 0.5], dtype=torch.float32),
    )
    assert torch.allclose(loss, expected)


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
    batch_edge_feat = torch.tensor([[7.0]], dtype=torch.float32)
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph=graph,
        src=torch.tensor([10], dtype=torch.long),
        dst=torch.tensor([11], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        edge_feat=batch_edge_feat,
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
    replica_index = ReplicaPushIndex(
        replica_ptr=torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
        replica_target_index=torch.tensor([0], dtype=torch.long),
    )

    train_epoch(
        session,
        encoder,
        head,
        EdgePredictionTask(),
        opt,
        memory_commit=CTDGMemoryCommitHook(memory_replica_index=replica_index),
    )

    assert len(updater.specs) == 1
    spec = updater.specs[0]
    assert spec.src.tolist() == [10]
    assert spec.dst.tolist() == [11]
    assert spec.src_rows.tolist() == [0]
    assert spec.dst_rows.tolist() == [1]
    assert spec.ts.tolist() == [3.0]
    assert spec.edge_feat.tolist() == [[7.0]]
    assert spec.memory_replica_index is replica_index
    assert updater.handles[0].applied


def test_ctdg_memory_commit_hook_uses_memshare_ordering() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph="mfg",
        src=torch.tensor([10], dtype=torch.long),
        dst=torch.tensor([11], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    session = _FakeSession([batch])
    encoder = _FakeEncoder(num_rows=3, dim=4)
    updater = _OrderedMemoryUpdater()
    encoder.memory_updater = updater
    head = EdgePredictHead(dim=4)
    opt = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.01)

    train_epoch(
        session,
        encoder,
        head,
        EdgePredictionTask(),
        opt,
        memory_commit=CTDGMemoryCommitHook(),
    )

    assert updater.events[0] == "submit_commit"
    assert updater.events[-2:] == ["synchronize_shared", "handle_last_async"]


def test_ctdg_memory_commit_hook_memshare_mode_waits_in_call() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph="mfg",
        src=torch.tensor([10], dtype=torch.long),
        dst=torch.tensor([11], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    session = _FakeSession([batch])
    encoder = _FakeEncoder(num_rows=3, dim=4)
    updater = _OrderedMemoryUpdater()
    encoder.memory_updater = updater
    head = EdgePredictHead(dim=4)
    opt = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.01)

    train_epoch(
        session,
        encoder,
        head,
        EdgePredictionTask(),
        opt,
        memory_commit=CTDGMemoryCommitHook(wait_mode="memshare"),
    )

    assert updater.events[:3] == ["submit_commit", "synchronize_shared", "handle_last_async"]


def test_ctdg_memory_commit_hook_populates_mailbox_replica_spec() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 1, 2], dtype=torch.long),
        graph="mfg",
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([1], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        edge_feat=torch.tensor([[5.0]], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    updater = _MailboxReplicaUpdater()
    hook = CTDGMemoryCommitHook(
        memory_replica_index=ReplicaPushIndex(
            replica_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
            replica_target_index=torch.tensor([0, 1], dtype=torch.long),
        ),
        mailbox_replica_index=ReplicaPushIndex(
            replica_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
            replica_target_index=torch.tensor([0, 1], dtype=torch.long),
        ),
        mailbox_runtime=_FakeMailboxRuntime(),
    )
    encoder = _FakeEncoder(num_rows=3, dim=4)
    encoder.memory_updater = updater

    hook(encoder, batch)

    spec = updater.specs[0]
    assert spec.mailbox_replica_index is not None
    assert spec.mailbox_snapshot is not None
    assert spec.mailbox_snapshot_ts is not None
    assert tuple(spec.mailbox_snapshot.shape) == (2, 1, 5)
    assert tuple(spec.mailbox_snapshot_ts.shape) == (2, 1)


def test_ctdg_memory_commit_hook_compacts_duplicate_mailbox_replica_rows() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0], dtype=torch.long),
        graph="mfg",
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([0], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        edge_feat=torch.tensor([[5.0]], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([0], dtype=torch.long),
        neg_dst=torch.tensor([0], dtype=torch.long),
    )
    updater = _MailboxReplicaUpdater()
    hook = CTDGMemoryCommitHook(
        mailbox_replica_index=ReplicaPushIndex(
            replica_ptr=torch.tensor([0, 1], dtype=torch.long),
            replica_target_index=torch.tensor([0], dtype=torch.long),
        ),
        mailbox_runtime=_FakeMailboxRuntime(),
    )
    encoder = _FakeEncoder(num_rows=1, dim=4)
    encoder.memory_updater = updater

    hook(encoder, batch)

    spec = updater.specs[0]
    assert spec.mailbox_nodes is not None
    assert spec.mailbox_nodes.tolist() == [0]
    assert spec.mailbox_snapshot is not None
    assert tuple(spec.mailbox_snapshot.shape) == (1, 1, 5)


def test_runtime_memory_commit_uses_rows_to_disambiguate_duplicate_node_ids() -> None:
    updater = RuntimeAsyncMemoryUpdater(torch.nn.Identity(), committer=None)
    updater.last_updated_nid = torch.tensor([0, 0, 1], dtype=torch.long)
    updater.last_updated_memory = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)
    updater.last_updated_ts = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)
    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
        edge_feat=torch.tensor([[9.0]], dtype=torch.float32),
        src_rows=torch.tensor([1], dtype=torch.long),
        dst_rows=torch.tensor([2], dtype=torch.long),
    )

    prepared = updater._prepare_commit_inputs(spec)

    assert prepared.memory_nodes.tolist() == [0, 1]
    assert prepared.memory_values.tolist() == [[20.0], [30.0]]
    assert prepared.memory_ts.tolist() == [2.0, 2.0]
    assert prepared.mailbox_msg.tolist() == [[20.0, 30.0, 9.0], [30.0, 20.0, 9.0]]


def test_mailbox_replica_snapshot_uses_row_ids_for_duplicate_node_ids() -> None:
    batch = Batch(
        split="train",
        roots=torch.tensor([0, 0], dtype=torch.long),
        graph="mfg",
        src=torch.tensor([0], dtype=torch.long),
        dst=torch.tensor([0], dtype=torch.long),
        ts=torch.tensor([3.0], dtype=torch.float32),
        edge_feat=torch.tensor([[5.0]], dtype=torch.float32),
        pos_src=torch.tensor([1], dtype=torch.long),
        pos_dst=torch.tensor([0], dtype=torch.long),
        neg_dst=torch.tensor([0], dtype=torch.long),
    )
    updater = _MailboxReplicaUpdater()
    updater.last_updated_nid = torch.tensor([0, 0], dtype=torch.long)
    updater.last_updated_memory = torch.tensor([[1.0, 2.0], [9.0, 10.0]], dtype=torch.float32)
    hook = CTDGMemoryCommitHook(
        mailbox_replica_index=ReplicaPushIndex(
            replica_ptr=torch.tensor([0, 1], dtype=torch.long),
            replica_target_index=torch.tensor([0], dtype=torch.long),
        ),
        mailbox_runtime=_FakeMailboxRuntime(),
    )
    encoder = _FakeEncoder(num_rows=2, dim=4)
    encoder.memory_updater = updater

    hook(encoder, batch)

    spec = updater.specs[0]
    assert spec.mailbox_snapshot is not None
    assert spec.mailbox_snapshot[0, 0].tolist() == [9.0, 10.0, 1.0, 2.0, 5.0]


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
        self.synchronize_shared_calls = 0
        self.handle_last_async_calls = 0

    def submit_commit(self, spec):
        handle = _FakeCommitHandle()
        self.specs.append(spec)
        self.handles.append(handle)
        return handle

    def synchronize_shared(self) -> None:
        self.synchronize_shared_calls += 1

    def handle_last_async(self) -> None:
        self.handle_last_async_calls += 1
        if self.handles:
            self.handles[-1].wait_apply()


class _MailboxReplicaUpdater(_FakeMemoryUpdater):
    def __init__(self) -> None:
        super().__init__()
        self.last_updated_nid = torch.tensor([0, 1], dtype=torch.long)
        self.last_updated_memory = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)


class _OrderedMemoryUpdater:
    def __init__(self) -> None:
        self.events: list[str] = []

    def synchronize_shared(self) -> None:
        self.events.append("synchronize_shared")

    def handle_last_async(self) -> None:
        self.events.append("handle_last_async")

    def submit_commit(self, spec):
        self.events.append("submit_commit")
        return _FakeCommitHandle()


class _FakeMailboxIndex:
    def master_for(self, node_ids):
        return node_ids


class _FakeMailboxStore:
    def __init__(self) -> None:
        self.mailbox_ts = torch.zeros((8, 1), dtype=torch.float32)

    def project_append_rows(self, rows, msg, ts, *, reduce="max_ts"):
        snap = msg.reshape(msg.size(0), 1, msg.size(1)).clone()
        snap_ts = ts.reshape(ts.size(0), 1).clone()
        return snap, snap_ts


class _FakeMailboxRuntime:
    def __init__(self) -> None:
        self.index = _FakeMailboxIndex()
        self.store = _FakeMailboxStore()
