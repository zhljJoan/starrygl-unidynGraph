"""Packed distributed index helpers for chunk artifacts.

The layout follows the MemShare convention:

    bits 0..47   local index
    bit 48       shared flag
    bit 49       cached/shadow flag
    bits 50..65  partition id
"""

from __future__ import annotations

import torch
from torch import Tensor

LOCAL_BITS = 48
PART_BITS = 16
SHARED_BIT = 1 << LOCAL_BITS
CACHED_BIT = 1 << (LOCAL_BITS + 1)
PART_SHIFT = LOCAL_BITS + 2
LOCAL_MASK = (1 << LOCAL_BITS) - 1
PART_MASK = (1 << PART_BITS) - 1


def encode_dist_index(
    local_ids: Tensor,
    part_ids: Tensor,
    *,
    shared: Tensor | bool = False,
    cached: Tensor | bool = False,
) -> Tensor:
    """Pack local ids, partition ids, and optional flags into int64 indices."""

    local = local_ids.long() & LOCAL_MASK
    part = (part_ids.long() & PART_MASK) << PART_SHIFT
    out = local | part
    if isinstance(shared, Tensor):
        out = out | (shared.bool().long() << LOCAL_BITS)
    elif shared:
        out = out | SHARED_BIT
    if isinstance(cached, Tensor):
        out = out | (cached.bool().long() << (LOCAL_BITS + 1))
    elif cached:
        out = out | CACHED_BIT
    return out.long()


def dist_index_loc(index: Tensor) -> Tensor:
    return index.long() & LOCAL_MASK


def dist_index_part(index: Tensor) -> Tensor:
    return (index.long() >> PART_SHIFT) & PART_MASK


def dist_index_is_shared(index: Tensor) -> Tensor:
    return (((index.long() >> LOCAL_BITS) & 1) != 0)


def dist_index_is_cached(index: Tensor) -> Tensor:
    return (((index.long() >> (LOCAL_BITS + 1)) & 1) != 0)
