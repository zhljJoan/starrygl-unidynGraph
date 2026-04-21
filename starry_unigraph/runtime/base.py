"""Base runtime protocol.

Defines :class:`RuntimeProtocol` — minimal abstract interface for any graph
computation paradigm (Flare DTDG, CTDG online, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class RuntimeProtocol(ABC):
    """Minimal runtime interface for all graph computation paradigms."""

    @abstractmethod
    def iter_batches(self, split: str, **kwargs: Any) -> Iterable[Any]:
        ...

    @abstractmethod
    def step(self, batch: Any, training: bool) -> dict[str, Any]:
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...
