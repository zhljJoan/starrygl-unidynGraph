from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class FeatureReadLayout:
    read_index: Tensor
    read_ptr: Tensor
    compute_to_feature: Tensor
    time_slices: Tensor | None = None


@dataclass(slots=True)
class EdgeFeatureReadLayout:
    read_index: Tensor
    read_ptr: Tensor
    compute_to_feature: Tensor
    time_slices: Tensor | None = None


@dataclass(slots=True)
class MemoryReadLayout:
    read_index: Tensor
    read_ptr: Tensor
    compute_to_memory: Tensor


@dataclass(slots=True)
class MemoryWriteLayout:
    target_index: Tensor
    target_ptr: Tensor
    source_pos: Tensor


@dataclass(slots=True)
class MailboxReadLayout:
    read_index: Tensor
    read_ptr: Tensor
    compute_to_mailbox: Tensor


@dataclass(slots=True)
class MailboxWriteLayout:
    target_index: Tensor
    target_ptr: Tensor
    source_pos: Tensor


@dataclass(slots=True)
class ReplicaPushLayout:
    target_index: Tensor
    target_ptr: Tensor
    source_pos: Tensor
