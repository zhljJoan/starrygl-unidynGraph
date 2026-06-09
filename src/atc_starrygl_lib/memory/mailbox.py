from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
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


class MailboxStore:
    """Row-indexed K-slot mailbox storage."""

    def __init__(self, mailbox: Tensor, mailbox_ts: Tensor, next_pos: Tensor, *, row_map: Tensor | None = None) -> None:
        if mailbox.dim() != 3:
            raise ValueError("mailbox must be [N, K, msg_dim]")
        if mailbox_ts.shape != mailbox.shape[:2]:
            raise ValueError("mailbox_ts must be [N, K]")
        if next_pos.numel() != mailbox.size(0):
            raise ValueError("next_pos must have one value per row")
        self.mailbox = mailbox
        self.mailbox_ts = mailbox_ts
        self.next_pos = next_pos
        self.row_map = row_map

    @property
    def device(self) -> torch.device:
        return self.mailbox.device

    def gather_rows(self, rows: Tensor, _time_slices: Tensor | None = None) -> tuple[Tensor, Tensor]:
        row = self._physical_rows(rows)
        return self.mailbox.index_select(0, row).to(rows.device), self.mailbox_ts.index_select(0, row).to(rows.device)

    def reset_zeros(self) -> None:
        self.mailbox.zero_()
        self.mailbox_ts.zero_()
        self.next_pos.zero_()

    def append_rows(self, rows: Tensor, msg: Tensor, ts: Tensor, *, reduce: str = "max_ts") -> None:
        row = self._physical_rows(rows)
        message = msg.to(device=self.mailbox.device, dtype=self.mailbox.dtype)
        ts_in = ts.to(device=self.mailbox_ts.device, dtype=self.mailbox_ts.dtype).reshape(-1)
        if row.numel() != message.size(0) or row.numel() != ts_in.numel():
            raise ValueError("rows, msg, and ts must have aligned leading dimensions")
        if message.dim() != 2:
            raise ValueError("msg must be [N, msg_dim]")
        if int(message.size(1)) != int(self.mailbox.size(2)):
            raise ValueError(
                f"msg_dim mismatch: message has width {int(message.size(1))}, "
                f"mailbox expects {int(self.mailbox.size(2))}"
            )
        if reduce == "max_ts" and row.numel() > 1:
            row, message, ts_in = _latest_by_row(row, message, ts_in)
        # NOTE: MemShare-compatible behavior - do NOT check against existing timestamps
        # Original code checked: if reduce == "max_ts": oldest = self.mailbox_ts[row].min(dim=1).values; keep = ts_in > oldest
        # This prevented old messages from overwriting newer ones, but MemShare allows it (circular queue)
        # Removing this check to match MemShare behavior for accuracy reproduction
        elif reduce != "append" and reduce != "max_ts":
            raise ValueError(f"unknown mailbox reduce mode: {reduce!r}")
        k = int(self.mailbox.size(1))
        pos = self.next_pos[row] % k
        self.mailbox[row, pos] = message
        self.mailbox_ts[row, pos] = ts_in
        self.next_pos[row] = (pos + 1) % k

    def project_append_rows(self, rows: Tensor, msg: Tensor, ts: Tensor, *, reduce: str = "max_ts") -> tuple[Tensor, Tensor]:
        row = self._physical_rows(rows)
        message = msg.to(device=self.mailbox.device, dtype=self.mailbox.dtype)
        ts_in = ts.to(device=self.mailbox_ts.device, dtype=self.mailbox_ts.dtype).reshape(-1)
        if row.numel() != message.size(0) or row.numel() != ts_in.numel():
            raise ValueError("rows, msg, and ts must have aligned leading dimensions")
        if message.dim() != 2:
            raise ValueError("msg must be [N, msg_dim]")
        if int(message.size(1)) != int(self.mailbox.size(2)):
            raise ValueError(
                f"msg_dim mismatch: message has width {int(message.size(1))}, "
                f"mailbox expects {int(self.mailbox.size(2))}"
            )
        if reduce == "max_ts" and row.numel() > 1:
            row, message, ts_in = _latest_by_row(row, message, ts_in)
        # NOTE: MemShare-compatible behavior - do NOT check against existing timestamps
        # Removed timestamp check to match MemShare behavior
        elif reduce != "append" and reduce != "max_ts":
            raise ValueError(f"unknown mailbox reduce mode: {reduce!r}")
        current_mail = self.mailbox.index_select(0, row).clone()
        current_ts = self.mailbox_ts.index_select(0, row).clone()
        current_pos = self.next_pos.index_select(0, row).clone()
        k = int(self.mailbox.size(1))
        pos = current_pos % k
        current_mail[torch.arange(int(row.numel()), device=row.device), pos] = message
        current_ts[torch.arange(int(row.numel()), device=row.device), pos] = ts_in
        return current_mail, current_ts

    def replace_rows(self, rows: Tensor, mailbox: Tensor, mailbox_ts: Tensor) -> None:
        row = self._physical_rows(rows)
        self.mailbox[row] = mailbox.to(device=self.mailbox.device, dtype=self.mailbox.dtype)
        self.mailbox_ts[row] = mailbox_ts.to(device=self.mailbox_ts.device, dtype=self.mailbox_ts.dtype)

    def _physical_rows(self, rows: Tensor) -> Tensor:
        row = rows.long().to(self.mailbox.device)
        if self.row_map is None:
            return row
        return self.row_map.to(self.mailbox.device).index_select(0, row)


def _latest_by_row(row: Tensor, msg: Tensor, ts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    unique, inverse = torch.unique(row, return_inverse=True)
    if unique.numel() == row.numel():
        return row, msg, ts
    latest_ts = torch.full((unique.numel(),), float("-inf"), dtype=ts.dtype, device=ts.device)
    latest_ts.scatter_reduce_(0, inverse, ts, reduce="amax", include_self=True)
    pos = torch.arange(row.numel(), dtype=torch.long, device=row.device)
    sentinel = torch.full_like(pos, row.numel())
    selected_pos = torch.where(ts == latest_ts[inverse], pos, sentinel)
    selected = torch.full((unique.numel(),), row.numel(), dtype=torch.long, device=row.device)
    selected.scatter_reduce_(0, inverse, selected_pos, reduce="amin", include_self=True)
    return unique, msg[selected], latest_ts
