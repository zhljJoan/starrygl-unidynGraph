"""DTDG schema types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SnapshotRoutePlan:
    route_type: str
    cache_policy: str

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "SnapshotRoutePlan"}


@dataclass
class DTDGPartitionBook:
    num_parts: int
    partition_algo: str
    snapshot_count: int

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"graph_mode": "dtdg"}
