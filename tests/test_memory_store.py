from __future__ import annotations

import torch

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
