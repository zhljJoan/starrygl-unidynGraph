from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class CacheLookup:
    hit_mask: Tensor
    values: Tensor


class HotMemoryCache:
    """Interface for resident hot-node memory cache."""

    def lookup(self, node_ids: Tensor) -> CacheLookup:
        raise NotImplementedError

    def update(self, node_ids: Tensor, values: Tensor) -> None:
        raise NotImplementedError
