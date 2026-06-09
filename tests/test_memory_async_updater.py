from __future__ import annotations

import torch

from atc_starrygl_lib.comm.layouts import MemoryWriteLayout
from atc_starrygl_lib.memory import (
    AsyncCommitHandle,
    AsyncMemoryCommitter,
    AsyncMemoryUpdateSpec,
    HistoricalBlend,
    HistoricalDeltaFilter,
    RuntimeAsyncMemoryUpdater,
    SharedHistoricalCache,
)
from atc_starrygl_lib.memory.async_updater import _build_mailbox_messages
from atc_starrygl_lib.memory.mailbox import MailboxStore
from atc_starrygl_lib.memory.shared_sync import ReplicaPushIndex
from atc_starrygl_lib.models.ctdg.memory_updater import TransformerMemoryUpdater


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

    assert len(committer.shared_submissions) == 2
    shared_nodes, shared_memory, shared_ts, _ = committer.shared_submissions[1]
    assert shared_nodes.numel() == 0
    assert shared_memory.shape == (0, 2)
    assert shared_ts.numel() == 0


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


def test_shared_historical_cache_only_tracks_filter_state() -> None:
    cache = SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.1, times_threshold=10)

    update_mask = cache.historical_check(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )

    assert update_mask.tolist() == [True]
    assert torch.allclose(cache.historical_memory[0], torch.tensor([1.0, 0.0], dtype=torch.float32))
    assert float(cache.historical_ts[0]) == 1.0


def test_historical_delta_filter_can_preload_candidate_delta_into_increment_history() -> None:
    filt = HistoricalDeltaFilter(
        memory_dim=2,
        num_nodes=4,
        preload_candidate_delta=True,
    )

    filt.preload_candidate(
        torch.tensor([0], dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )

    assert torch.allclose(filt.increment_sum[0], torch.tensor([1.0, 0.0], dtype=torch.float32))
    assert torch.allclose(filt.increment_count[0], torch.ones(1, dtype=torch.float32))
    assert torch.allclose(filt.historical_memory[0], torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_historical_delta_filter_accumulates_increment_average() -> None:
    filt = HistoricalDeltaFilter(memory_dim=2, num_nodes=4)

    filt.update(torch.tensor([0], dtype=torch.long), torch.tensor([[2.0, 0.0]], dtype=torch.float32))
    filt.update(torch.tensor([0], dtype=torch.long), torch.tensor([[4.0, 2.0]], dtype=torch.float32))

    assert torch.allclose(filt.increment_sum[0], torch.tensor([6.0, 2.0], dtype=torch.float32))
    assert torch.allclose(filt.increment_count[0], torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(filt.get_increment(torch.tensor([0], dtype=torch.long))[0], torch.tensor([3.0, 1.0], dtype=torch.float32))


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


def test_runtime_async_memory_updater_drains_previous_staged_commit_before_resubmit() -> None:
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
    updater.submit_commit(spec)

    assert committer.applied == ["shared", "memory", "mailbox"]
    assert committer.event_log == [
        "submit_shared",
        "submit_p2p_memory",
        "submit_p2p_mailbox",
        "submit_shared",
        "submit_p2p_memory",
        "submit_p2p_mailbox",
    ]


def test_runtime_async_memory_updater_refreshes_shared_cache_after_shared_apply() -> None:
    base = _BaseUpdater()
    base.last_updated_memory[0] = torch.tensor([9.0, 8.0], dtype=torch.float32)
    base.last_updated_ts[0] = torch.tensor(7.0, dtype=torch.float32)
    committer = _FakeCommitter()
    historical_cache = SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.1, times_threshold=10)
    historical_filter = HistoricalDeltaFilter(memory_dim=2, num_nodes=4)
    updater = RuntimeAsyncMemoryUpdater(
        base,
        committer=committer,
        historical_cache=historical_cache,
        historical_filter=historical_filter,
        use_staged_commit=True,
    )

    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([3.0], dtype=torch.float32),
    )
    spec.memory_replica_index = _replica_index_for_nodes([0])
    updater.forward("mfg", spec)

    updater.synchronize_shared()

    assert torch.allclose(historical_cache.historical_memory[0], torch.tensor([9.0, 8.0], dtype=torch.float32))
    assert float(historical_cache.historical_ts[0]) == 7.0


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
    assert memory_replica_index is spec.memory_replica_index
    assert committer.shared_mailbox_submissions[-1] == (
        [0, 1],
        [2, 1],
    )


def test_runtime_async_memory_updater_compacts_latest_node_payloads_before_p2p() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(base, committer=committer, use_staged_commit=True)

    base.last_updated_nid = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    base.last_updated_ts = torch.tensor([1.0, 1.0, 3.0, 2.0], dtype=torch.float32)
    base.last_updated_memory = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )
    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
        torch.tensor([1.0, 3.0], dtype=torch.float32),
        src_rows=torch.tensor([0, 3], dtype=torch.long),
        dst_rows=torch.tensor([1, 2], dtype=torch.long),
    )

    updater.forward("mfg", None)
    updater.submit_commit(spec)

    memory_nodes, memory_values, memory_ts = committer.p2p_memory_submissions[-1]
    assert memory_nodes.tolist() == [0, 1]
    assert memory_values.tolist() == [[30.0, 0.0], [40.0, 0.0]]
    assert memory_ts.tolist() == [3.0, 2.0]
    mailbox_nodes, mailbox_msg, mailbox_ts = committer.p2p_mailbox_submissions[-1]
    assert mailbox_nodes.tolist() == [0, 1]
    assert mailbox_ts.tolist() == [3.0, 3.0]
    assert mailbox_msg.tolist() == [[30.0, 0.0, 40.0, 0.0], [40.0, 0.0, 30.0, 0.0]]


def test_runtime_async_memory_updater_uses_precomputed_commit_rows_without_compaction() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(base, committer=committer, use_staged_commit=True)

    base.last_updated_nid = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    base.last_updated_ts = torch.tensor([1.0, 1.0, 3.0, 2.0], dtype=torch.float32)
    base.last_updated_memory = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )
    spec = AsyncMemoryUpdateSpec(
        memory_nodes=torch.tensor([0, 1], dtype=torch.long),
        memory_rows=torch.tensor([2, 3], dtype=torch.long),
        mailbox_nodes=torch.tensor([0, 1], dtype=torch.long),
        mailbox_self_rows=torch.tensor([2, 3], dtype=torch.long),
        mailbox_peer_rows=torch.tensor([3, 2], dtype=torch.long),
        mailbox_ts=torch.tensor([3.0, 2.0], dtype=torch.float32),
        precomputed_commit=True,
    )

    updater.forward("mfg", None)
    updater.submit_commit(spec)

    memory_nodes, memory_values, memory_ts = committer.p2p_memory_submissions[-1]
    assert memory_nodes.tolist() == [0, 1]
    assert memory_values.tolist() == [[30.0, 0.0], [40.0, 0.0]]
    assert memory_ts.tolist() == [3.0, 2.0]
    mailbox_nodes, mailbox_msg, mailbox_ts = committer.p2p_mailbox_submissions[-1]
    assert mailbox_nodes.tolist() == [0, 1]
    assert mailbox_msg.tolist() == [[30.0, 0.0, 40.0, 0.0], [40.0, 0.0, 30.0, 0.0]]
    assert mailbox_ts.tolist() == [3.0, 2.0]



def test_async_memory_committer_uses_precomputed_write_layout() -> None:
    runtime = _FakeMemoryRuntime()
    committer = AsyncMemoryCommitter(runtime)
    layout = MemoryWriteLayout(
        target_index=torch.tensor([20, 10], dtype=torch.long),
        target_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
        source_pos=torch.tensor([1, 0], dtype=torch.long),
    )

    committer.submit_p2p_memory(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        write_layout=layout,
    )

    assert runtime.build_write_layout_called is False
    assert runtime.last_layout is layout
    assert runtime.last_memory.tolist() == [[1.0], [2.0]]
    assert runtime.last_ts.tolist() == [3.0, 4.0]


def test_runtime_async_memory_updater_delta_compensation_toggle() -> None:
    base = _BaseUpdater()
    cache = SharedHistoricalCache(memory_dim=2, num_nodes=4, alpha=0.1, times_threshold=10)
    filt = HistoricalDeltaFilter(memory_dim=2, num_nodes=4)
    # Preload historical stats for node 0 so compensation has a valid increment.
    filt.historical_memory[0] = torch.tensor([0.2, 0.2], dtype=torch.float32)
    filt.increment_sum[0] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    filt.increment_count[0] = torch.tensor([1.0], dtype=torch.float32)

    updater_off = RuntimeAsyncMemoryUpdater(
        base,
        committer=_FakeCommitter(),
        historical_cache=cache,
        historical_filter=filt,
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
        historical_filter=filt,
        enable_delta_compensation=True,
        delta_compensation_gamma=0.5,
    )
    updater_on.forward("mfg", None)
    on_memory = updater_on.last_updated_memory.clone()

    assert not torch.allclose(on_memory[0], off_memory[0])
    assert torch.allclose(on_memory[1], off_memory[1])


def test_runtime_async_memory_updater_historical_blend_uses_filter_increment_and_updates_change() -> None:
    base = _BaseUpdater()
    committer = _FakeCommitter()
    filt = HistoricalDeltaFilter(memory_dim=2, num_nodes=4)
    filt.historical_memory[0] = torch.tensor([2.0, 3.0], dtype=torch.float32)
    filt.increment_sum[0] = torch.tensor([4.0, 0.0], dtype=torch.float32)
    filt.increment_count[0] = torch.tensor([1.0], dtype=torch.float32)
    updater = RuntimeAsyncMemoryUpdater(
        base,
        committer=committer,
        historical_blend=HistoricalBlend(memory_dim=2, learnable_gamma=False),
        historical_filter=filt,
    )

    block = _FakeGraphBlock(
        {
            "shared_mask": torch.tensor([True, False], dtype=torch.bool),
            "his_mem": torch.tensor([[2.0, 3.0], [0.0, 0.0]], dtype=torch.float32),
        }
    )
    updater.forward([block], None)

    gamma = torch.sigmoid(torch.tensor(0.9))
    pred = torch.tensor([1.0, -1.0], dtype=torch.float32)
    expected_shared = gamma * torch.tensor([1.0, 0.0], dtype=torch.float32) + (1.0 - gamma) * pred
    assert torch.allclose(updater.last_updated_memory[0], expected_shared)
    expected_change = expected_shared - torch.tensor([2.0, 3.0], dtype=torch.float32)
    assert torch.allclose(
        filt.get_increment(torch.tensor([0], dtype=torch.long))[0],
        (torch.tensor([4.0, 0.0], dtype=torch.float32) + expected_change) / 2.0,
    )


def test_historical_blend_uses_raw_increment_magnitude() -> None:
    blend = HistoricalBlend(memory_dim=2, learnable_gamma=False)
    updated = torch.tensor([[10.0, 0.0], [5.0, 5.0]], dtype=torch.float32)
    shared_mask = torch.tensor([True, False], dtype=torch.bool)
    historical = torch.tensor([[2.0, 3.0], [0.0, 0.0]], dtype=torch.float32)
    increment = torch.tensor([[4.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

    out = blend(updated, shared_mask, historical, increment)

    gamma = torch.sigmoid(torch.tensor(0.9))
    expected_shared = gamma * updated[0] + (1.0 - gamma) * (historical[0] + increment[0])
    assert torch.allclose(out[0], expected_shared)
    assert torch.allclose(out[1], updated[1])


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


def test_runtime_async_memory_updater_runs_transformer_memory_update_like_memshare() -> None:
    base = TransformerMemoryUpdater(
        memory_param={
            "mailbox_size": 2,
            "attention_head": 2,
            "combine_node_feature": False,
        },
        dim_in=4,
        dim_out=4,
        dim_time=0,
        train_param={"dropout": 0.0, "att_dropout": 0.0},
        dim_node_feat=4,
    )
    committer = _FakeCommitter()
    updater = RuntimeAsyncMemoryUpdater(base, committer=committer, use_staged_commit=True)
    block = _FakeGraphBlock(
        srcdata={
            "ID": torch.tensor([0, 1], dtype=torch.long),
            "ts": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "mem": torch.tensor([[0.2, 0.1, 0.0, 0.3], [0.0, 0.4, 0.2, 0.1]], dtype=torch.float32),
            "mem_input": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                    [0.3, 0.1, 0.7, 0.2, 0.2, 0.2, 0.8, 0.8],
                ],
                dtype=torch.float32,
            ),
            "mail_ts": torch.tensor([[0.2, 0.8], [0.5, 1.5]], dtype=torch.float32),
            "h": torch.zeros((2, 4), dtype=torch.float32),
        }
    )
    spec = AsyncMemoryUpdateSpec.from_edges(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )

    updated = updater.forward([block], spec)

    assert updated is not None
    assert base.last_updated_memory is not None
    assert torch.allclose(updater.last_updated_memory, base.last_updated_memory)
    assert torch.equal(updater.last_updated_nid, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(updater.last_updated_ts, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.allclose(block.srcdata["h"], updated)
    updater.synchronize_shared()
    updater.handle_last_async()
    assert len(committer.p2p_memory_submissions) == 1
    assert len(committer.p2p_mailbox_submissions) == 1


class _BaseUpdater(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_updated_memory = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
        self.last_updated_ts = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.last_updated_nid = torch.tensor([0, 1], dtype=torch.long)

    def forward(self, mfg):
        return self.last_updated_memory


class _FakeGraphBlock:
    def __init__(self, srcdata: dict[str, torch.Tensor]) -> None:
        self.srcdata = srcdata

    def num_src_nodes(self) -> int:
        return int(self.srcdata["ID"].numel())


class _RecordingHandle:
    def __init__(self, label: str, sink: list[str]) -> None:
        self.label = label
        self.sink = sink

    def wait_apply(self) -> None:
        self.sink.append(self.label)


class _FakeCommitter:
    def __init__(self) -> None:
        self.memory_runtime = _FakeCommitMemoryRuntime()
        self.mailbox_runtime = _FakeMailboxRuntime()
        self.event_log: list[str] = []
        self.applied: list[str] = []
        self.shared_submissions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, ReplicaPushIndex | None]] = []
        self.shared_mailbox_submissions: list[tuple[list[int], list[int]]] = []
        self.p2p_memory_submissions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.p2p_mailbox_submissions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

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
        if updated_nodes.numel() > 0:
            self.memory_runtime.store.update_rows(
                updated_nodes.long(),
                updated_memory,
                updated_ts,
                reduce="overwrite",
            )
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

    def submit_p2p_memory(self, updated_nodes, updated_memory, updated_ts, *, write_layout=None):
        self.event_log.append("submit_p2p_memory")
        self.p2p_memory_submissions.append((updated_nodes.clone(), updated_memory.clone(), updated_ts.clone()))
        return AsyncCommitHandle(memory_handle=_RecordingHandle("memory", self.applied))

    def submit_p2p_mailbox(self, mailbox_nodes, mailbox_msg, mailbox_ts, *, write_layout=None):
        self.event_log.append("submit_p2p_mailbox")
        self.p2p_mailbox_submissions.append((mailbox_nodes.clone(), mailbox_msg.clone(), mailbox_ts.clone()))
        return AsyncCommitHandle(mailbox_handle=_RecordingHandle("mailbox", self.applied))


class _FakeMailboxRuntime:
    def __init__(self) -> None:
        self.store = _FakeMailboxStore()


class _FakeMailboxStore:
    def __init__(self) -> None:
        self.mailbox = torch.zeros((4, 1, 5), dtype=torch.float32)
        self.mailbox_ts = torch.zeros((4, 1), dtype=torch.float32)


class _FakeCommitMemoryRuntime:
    def __init__(self) -> None:
        from atc_starrygl_lib.memory.store import MemoryStore

        self.store = MemoryStore(
            torch.zeros((4, 2), dtype=torch.float32),
            torch.zeros((4,), dtype=torch.float32),
        )
        self.index = _FakeCommitMemoryIndex()


class _FakeCommitMemoryIndex:
    def __init__(self) -> None:
        self.master_dist_index = torch.arange(4, dtype=torch.long)

    def master_for(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.master_dist_index.index_select(0, node_ids.long())


class _FakeMemoryRuntime:
    def __init__(self) -> None:
        self.build_write_layout_called = False
        self.last_layout = None
        self.last_memory = None
        self.last_ts = None

    def build_write_layout(self, updated_nodes):
        self.build_write_layout_called = True
        raise AssertionError("build_write_layout should not be called")

    def write(self, layout, memory, ts):
        self.last_layout = layout
        self.last_memory = memory.clone()
        self.last_ts = ts.clone()
        return _RecordingHandle("memory", [])


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
