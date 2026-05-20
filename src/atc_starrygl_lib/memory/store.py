from __future__ import annotations

import torch
from torch import Tensor


class MemoryStore:
    """Row-indexed temporal memory storage."""

    def __init__(self, memory: Tensor, ts: Tensor) -> None:
        if memory.size(0) != ts.numel():
            raise ValueError("memory and ts must have the same row count")
        self.memory = memory
        self.ts = ts

    @property
    def device(self) -> torch.device:
        return self.memory.device

    def gather_rows(self, rows: Tensor, _time_slices: Tensor | None = None) -> tuple[Tensor, Tensor]:
        row = rows.long().to(self.memory.device)
        return self.memory.index_select(0, row).to(rows.device), self.ts.index_select(0, row).to(rows.device)

    def reset_zeros(self) -> None:
        self.memory.zero_()
        self.ts.zero_()

    def update_rows(self, rows: Tensor, memory: Tensor, ts: Tensor, *, reduce: str = "max_ts") -> None:
        row = rows.long().to(self.memory.device)
        mem = memory.to(device=self.memory.device, dtype=self.memory.dtype)
        ts_in = ts.to(device=self.ts.device, dtype=self.ts.dtype).reshape(-1)
        if row.numel() != mem.size(0) or row.numel() != ts_in.numel():
            raise ValueError("rows, memory, and ts must have aligned leading dimensions")
        if reduce == "max_ts":
            if row.numel() > 1:
                row, mem, ts_in = _latest_by_row(row, mem, ts_in)
            keep = ts_in > self.ts[row]
            if keep.any():
                upd = row[keep]
                self.memory[upd] = mem[keep]
                self.ts[upd] = ts_in[keep]
            return
        if reduce == "overwrite":
            self.memory[row] = mem
            self.ts[row] = ts_in
            return
        raise ValueError(f"unknown memory reduce mode: {reduce!r}")


def _latest_by_row(row: Tensor, memory: Tensor, ts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    unique, inverse = torch.unique(row, return_inverse=True)
    if unique.numel() == row.numel():
        return row, memory, ts
    latest_ts = torch.full((unique.numel(),), float("-inf"), dtype=ts.dtype, device=ts.device)
    latest_ts.scatter_reduce_(0, inverse, ts, reduce="amax", include_self=True)
    pos = torch.arange(row.numel(), dtype=torch.long, device=row.device)
    sentinel = torch.full_like(pos, row.numel())
    selected_pos = torch.where(ts == latest_ts[inverse], pos, sentinel)
    selected = torch.full((unique.numel(),), row.numel(), dtype=torch.long, device=row.device)
    selected.scatter_reduce_(0, inverse, selected_pos, reduce="amin", include_self=True)
    return unique, memory[selected], latest_ts
