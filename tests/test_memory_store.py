from __future__ import annotations

import torch

from atc_starrygl_lib.memory.mailbox import MailboxStore
from atc_starrygl_lib.memory.store import MemoryStore


def test_memory_store_keeps_latest_duplicate_row_update() -> None:
    store = MemoryStore(torch.zeros((2, 1)), torch.zeros(2))

    store.update_rows(
        torch.tensor([1, 1, 0], dtype=torch.long),
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([1.0, 3.0, 2.0]),
    )

    assert store.memory.tolist() == [[3.0], [2.0]]
    assert store.ts.tolist() == [2.0, 3.0]


def test_memory_store_row_map_preserves_logical_rows_for_read_and_write() -> None:
    store = MemoryStore(
        torch.tensor([[20.0], [10.0], [30.0]]),
        torch.tensor([2.0, 1.0, 3.0]),
        row_map=torch.tensor([1, 0, 2], dtype=torch.long),
    )

    mem, ts = store.gather_rows(torch.tensor([0, 1, 2], dtype=torch.long))
    assert mem.tolist() == [[10.0], [20.0], [30.0]]
    assert ts.tolist() == [1.0, 2.0, 3.0]

    store.update_rows(torch.tensor([0], dtype=torch.long), torch.tensor([[40.0]]), torch.tensor([4.0]))
    assert store.memory.tolist() == [[20.0], [40.0], [30.0]]
    assert store.ts.tolist() == [2.0, 4.0, 3.0]


def test_mailbox_store_row_map_preserves_logical_rows_for_append_and_replace() -> None:
    store = MailboxStore(
        torch.zeros((3, 1, 2), dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.float32),
        torch.zeros(3, dtype=torch.long),
        row_map=torch.tensor([1, 0, 2], dtype=torch.long),
    )

    store.append_rows(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
        torch.tensor([10.0, 20.0]),
    )
    mailbox, mailbox_ts = store.gather_rows(torch.tensor([0, 1], dtype=torch.long))
    assert mailbox[:, 0].tolist() == [[1.0, 1.5], [2.0, 2.5]]
    assert mailbox_ts.tolist() == [[10.0], [20.0]]

    store.replace_rows(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[[3.0, 3.5]]]),
        torch.tensor([[30.0]]),
    )
    mailbox, mailbox_ts = store.gather_rows(torch.tensor([0], dtype=torch.long))
    assert mailbox.tolist() == [[[3.0, 3.5]]]
    assert mailbox_ts.tolist() == [[30.0]]
