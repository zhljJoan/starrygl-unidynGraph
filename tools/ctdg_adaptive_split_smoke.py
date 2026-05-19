#!/usr/bin/env python3
"""Distributed smoke test for adaptive_split_cpp."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from starry_unigraph.lib import adaptive_split_cpp


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    src = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    dst = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    ts = torch.arange(8, dtype=torch.double)
    result = adaptive_split_cpp.adaptive_split(
        src,
        dst,
        ts,
        3,
        1.0,
        1.0,
        0.5,
        0.0,
        False,
        0.8,
        1000,
    )
    print(
        f"rank={rank} group_index={result.group_index.tolist()} "
        f"keep={result.keep_indices.tolist()}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
