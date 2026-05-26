from __future__ import annotations

import torch

from atc_starrygl_lib.memory import AsyncCommitHandle, AsyncMemoryUpdateSpec, RuntimeAsyncMemoryUpdater, SharedHistoricalCache
from atc_starrygl_lib.memory.async_updater import _build_mailbox_messages
from atc_starrygl_lib.memory.mailbox import MailboxStore
from atc_starrygl_lib.memory.shared_sync import ReplicaPushIndex


def test_build_mailbox_messages_updates_both_edge_endpoints() -> None:
    nid = torch.tensor([1, 2], dtype=torch.long)
    memory = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    edge_feat = torch.tensor([[5.0]])

    msg = _build_mailbox_messages(
        nid,
        memory,
        src=torch.tensor([1], dtype=torch.long),
        dst=torch.tensor([2], dtype=torch.long),
        edge_feat=edge_feat,
    )

    assert msg.tolist() == [[10.0, 11.0, 20.0, 21.0, 5.0], [20.0, 21.0, 10.0, 11.0, 5.0]]


def test_runtime_async_memory_updater_filters_small_shared_updates() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(
        base,
        committer=committer,
        historical_cache=SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.1, times_threshold=10),
        use_staged_commit=True,
    )

    spec = AsyncMemoryUpdateSpec(memory_replica_index=_replica_index_for_nodes([0]))
    updater.forward("mfg", spec)
    updater.synchronize_shared()
    updater.handle_last_async()

    base.last_updated_memory = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    base.last_updated_ts = torch.tensor([1.0, 1.0], dtype=torch.float32)
    base.last_updated_nid = torch.tensor([0, 1], dtype=torch.long)
    updater.submit_commit(spec)

    assert len(committer.shared_submissions) == 1
    shared_nodes, shared_memory, shared_ts, _ = committer.shared_submissions[0]
    assert shared_nodes.tolist() == [0]
    assert shared_memory.tolist() == [[1.0, 0.0]]
    assert shared_ts.tolist() == [1.0]


def test_runtime_async_memory_updater_can_disable_shared_filter() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(
        base,
        committer=committer,
        historical_cache=SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.99, times_threshold=10),
        use_staged_commit=True,
        use_shared_filter=False,
    )

    spec = AsyncMemoryUpdateSpec(memory_replica_index=_replica_index_for_nodes([0]))
    updater.forward("mfg", spec)
    updater.synchronize_shared()
    updater.handle_last_async()

    assert len(committer.shared_submissions) == 1
    shared_nodes, _, _, _ = committer.shared_submissions[0]
    assert shared_nodes.tolist() == [0]


def test_runtime_async_memory_updater_stages_shared_before_async_drain() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(base, committer=committer, use_staged_commit=True)

    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([3.0], dtype=torch.float32),
    )
    spec.memory_replica_index = _replica_index_for_nodes([0, 1])
    updater.forward("mfg", spec)

    assert committer.event_log == ["submit_shared", "submit_p2p_memory", "submit_p2p_mailbox"]
    assert committer.applied == []

    updater.synchronize_shared()
    assert committer.applied == ["shared"]

    updater.handle_last_async()
    assert committer.applied == ["shared", "memory", "mailbox"]


def test_runtime_async_memory_updater_submits_mailbox_only_shared_payload() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(
        base,
        committer=committer,
        historical_cache=SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.99, times_threshold=10),
        use_staged_commit=True,
        use_shared_filter=True,
    )

    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([3.0], dtype=torch.float32),
    )
    spec.memory_replica_index = _replica_index_for_nodes([0, 1])
    spec.mailbox_replica_index = _replica_index_for_nodes([0, 1])
    spec.mailbox_snapshot = torch.ones((2, 1, 5), dtype=torch.float32)
    spec.mailbox_snapshot_ts = torch.ones((2, 1), dtype=torch.float32)

    updater.forward("mfg", spec)
    updater.synchronize_shared()
    updater.handle_last_async()

    base.last_updated_memory = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    base.last_updated_ts = torch.tensor([1.0, 1.0], dtype=torch.float32)
    base.last_updated_nid = torch.tensor([0, 1], dtype=torch.long)
    updater.submit_commit(spec)

    assert len(committer.shared_submissions) == 2
    shared_nodes, shared_memory, shared_ts, memory_replica_index = committer.shared_submissions[1]
    assert shared_nodes.numel() == 0
    assert shared_memory.shape == (0, 2)
    assert shared_ts.numel() == 0
    assert memory_replica_index is None
    assert committer.shared_mailbox_submissions[-1] == (
        [0, 1],
        [2, 1],
    )


def test_runtime_async_memory_updater_delta_compensation_toggle() -> None:
    base = _BaseUpdater()
    cache = SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.1, times_threshold=10)
    # Preload historical stats for node 0 so compensation has a valid increment.
    cache.historical_memory[0] = torch.tensor([0.2, 0.2], dtype=torch.float32)
    cache.increment_sum[0] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    cache.increment_count[0] = torch.tensor([1.0], dtype=torch.float32)

    updater_off = RuntimeAsyncMemoryUpdater(
        base,
        committer=_FakeCommitter(),
        historical_cache=cache,
        enable_delta_compensation=False,
        delta_compensation_gamma=0.5,
    )
    updater_off.forward("mfg", None)
    off_memory = updater_off.last_updated_memory.clone()

    base_on = _BaseUpdater()
    updater_on = RuntimeAsyncMemoryUpdater(
        base_on,
        committer=_FakeCommitter(),
        historical_cache=cache,
        enable_delta_compensation=True,
        delta_compensation_gamma=0.5,
    )
    updater_on.forward("mfg", None)
    on_memory = updater_on.last_updated_memory.clone()

    # Node 0 changes due to compensation; node 1 remains unchanged (no increment history).
    assert not torch.allclose(on_memory[0], off_memory[0])
    assert torch.allclose(on_memory[1], off_memory[1])


def test_mailbox_store_rejects_message_width_mismatch() -> None:
    store = MailboxStore(
        torch.zeros((2, 1, 5), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros((2,), dtype=torch.long),
    )

    try:
        store.append_rows(
            torch.tensor([0], dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
    except ValueError as exc:
        assert "msg_dim mismatch" in str(exc)
    else:
        raise AssertionError("expected mailbox width mismatch to raise")


class _BaseUpdater(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_updated_memory = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
        self.last_updated_ts = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.last_updated_nid = torch.tensor([0, 1], dtype=torch.long)

    def forward(self, mfg):
        return self.last_updated_memory


class _RecordingHandle:
    def __init__(self, label: str, sink: list[str]) -> None:
        self.label = label
        self.sink = sink

    def wait_apply(self) -> None:
        self.sink.append(self.label)


class _FakeCommitter:
    def __init__(self) -> None:
        self.event_log: list[str] = []
        self.applied: list[str] = []
        self.shared_submissions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, ReplicaPushIndex | None]] = []
        self.shared_mailbox_submissions: list[tuple[list[int], list[int]]] = []

    def submit_shared(
        self,
        updated_nodes,
        updated_memory,
        updated_ts,
        *,
        memory_replica_index=None,
        mailbox_nodes=None,
        mailbox_replica_index=None,
        mailbox_snapshot=None,
        mailbox_snapshot_ts=None,
    ):
        self.event_log.append("submit_shared")
        self.shared_submissions.append(
            (
                updated_nodes.clone(),
                updated_memory.clone(),
                updated_ts.clone(),
                memory_replica_index,
            )
        )
        if (
            mailbox_replica_index is not None
            and mailbox_nodes is not None
            and mailbox_snapshot is not None
            and mailbox_snapshot_ts is not None
        ):
            self.shared_mailbox_submissions.append(
                (
                    mailbox_nodes.clone().tolist(),
                    list(mailbox_snapshot_ts.shape),
                )
            )
        return AsyncCommitHandle(memory_replica_handle=_RecordingHandle("shared", self.applied))

    def submit_p2p_memory(self, updated_nodes, updated_memory, updated_ts):
        self.event_log.append("submit_p2p_memory")
        return AsyncCommitHandle(memory_handle=_RecordingHandle("memory", self.applied))

    def submit_p2p_mailbox(self, mailbox_nodes, mailbox_msg, mailbox_ts):
        self.event_log.append("submit_p2p_mailbox")
        return AsyncCommitHandle(mailbox_handle=_RecordingHandle("mailbox", self.applied))


def _replica_index_for_nodes(nodes: list[int]) -> ReplicaPushIndex:
    ptr = torch.zeros(5, dtype=torch.long)
    target = []
    offset = 0
    for node in range(4):
        ptr[node] = offset
        if node in nodes:
            target.append(node)
            offset += 1
    ptr[4] = offset
    return ReplicaPushIndex(
        replica_ptr=ptr,
        replica_target_index=torch.tensor(target, dtype=torch.long),
    )
