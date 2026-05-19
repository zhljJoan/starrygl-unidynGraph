from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from torch import Tensor


@dataclass
class Mail:
    node_ids: Tensor
    payload: Tensor
    timestamps: Tensor | None = None


class Mailbox:
    def __init__(self) -> None:
        self._pending: dict[int, list[Mail]] = defaultdict(list)

    def push(self, owner_rank: int, mail: Mail) -> None:
        self._pending[int(owner_rank)].append(mail)

    def pop_all(self) -> dict[int, list[Mail]]:
        pending = dict(self._pending)
        self._pending.clear()
        return pending
