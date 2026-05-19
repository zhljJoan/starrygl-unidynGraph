from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class CommRequest:
    node_ids: Tensor
    owner: Tensor
    payload: Tensor | None = None


class CommHandle(Protocol):
    def wait(self) -> Tensor | None:
        ...


class CommEngine(Protocol):
    def fetch(self, request: CommRequest) -> CommHandle:
        ...

    def push(self, request: CommRequest) -> CommHandle:
        ...
