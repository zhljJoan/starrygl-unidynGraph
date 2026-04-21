"""Per-node memory bank with K-slot mailbox for CTDG online training.

:class:`CTDGMemoryBank` is the central state container for TGN-style models.
It stores per-node hidden memory and a K-slot mailbox (edge-feature messages),
handles distributed async sync via ``all_to_all``, and integrates with
:class:`CTDGHistoricalCache` for change-detection filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from starry_unigraph.types import DistributedContext

if TYPE_CHECKING:
    from .cache import CTDGHistoricalCache


@dataclass
class _PendingMemorySync:
    id_work: Any
    payload_work: Any
    recv_ids: torch.Tensor
    recv_payload: torch.Tensor
    recv_counts: list[int]
    sent_remote_ids: torch.Tensor | None = None
    sent_remote_vals: torch.Tensor | None = None
    sent_remote_ts: torch.Tensor | None = None


@dataclass
class _PendingMailSync:
    id_work: Any
    payload_work: Any
    recv_ids: torch.Tensor
    recv_payload: torch.Tensor
    recv_counts: list[int]


@dataclass
class CTDGMemoryBank:
    """Per-node memory + K-slot mailbox.

    In distributed mode each rank only allocates storage for its own partition
    (``num_local_nodes`` nodes).  Global node IDs are mapped to local IDs via
    ``global2local``.  Nodes not owned by this rank use a temporary CPU buffer
    (``remote_memory``) that is filled by async sync operations.

    In single-rank mode ``num_local_nodes == num_nodes`` and the mapping is the
    identity, so all existing call sites work unchanged.
    """

    num_nodes: int       # total nodes in the graph
    hidden_dim: int
    device: str
    mailbox_slots: int = 1
    edge_feat_dim: int = 0
    # Distributed partition info (set by provider)
    rank: int = 0
    world_size: int = 1
    async_sync: bool = True
    node_parts: torch.Tensor | None = None  # [num_nodes] mapping node_id → partition_id

    def __post_init__(self) -> None:
        D = self.hidden_dim
        K = self.mailbox_slots
        E = max(0, self.edge_feat_dim)
        self.slot_width = 2 * D + E

        # Determine local_ids from node_parts (if available) or fall back to round-robin
        if self.node_parts is not None and self.world_size > 1:
            # Use node_parts tensor: find nodes assigned to this rank
            local_ids = torch.where(self.node_parts == self.rank)[0]
        elif self.world_size > 1:
            # Fall back to round-robin: node_id % world_size
            local_ids = torch.arange(self.rank, self.num_nodes, self.world_size, dtype=torch.long)
        else:
            # Single-rank: all nodes
            local_ids = torch.arange(self.num_nodes, dtype=torch.long)

        self.num_local_nodes = local_ids.numel()

        self.global2local = torch.full((self.num_nodes,), -1, dtype=torch.long)
        self.global2local[local_ids] = torch.arange(self.num_local_nodes, dtype=torch.long)
        self.local_ids = local_ids

        # Determine storage device once: try GPU, fall back to CPU
        try:
            torch.zeros(1, device=self.device)
            self._storage_device = self.device
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self._storage_device = "cpu"

        sd = self._storage_device
        self.memory     = torch.zeros(self.num_local_nodes, D,    dtype=torch.float32, device=sd)
        self.mailbox    = torch.zeros(self.num_local_nodes, K, self.slot_width, dtype=torch.float32, device=sd)
        self.mailbox_ts = torch.zeros(self.num_local_nodes, K,    dtype=torch.float32, device=sd)
        self.memory_ts  = torch.full( (self.num_local_nodes,), -1.0, dtype=torch.float32, device=sd)
        self.next_mail_pos = torch.zeros(self.num_local_nodes, dtype=torch.long, device=sd)

        self.memory_version  = 0
        self.mailbox_version = 0

        self.shared_nodes: torch.Tensor | None = None
        self.is_shared_mask: torch.Tensor | None = None
        self.historical_cache = None
        self.last_memory_sync: tuple | None = None
        self.last_mail_sync: tuple | None = None
        self._pending_memory_syncs: list[_PendingMemorySync] = []
        self._pending_mail_syncs: list[_PendingMailSync] = []

    def _to_local(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global node IDs → local storage indices (always CPU)."""
        return self.global2local[global_ids.cpu()]

    # ------------------------------------------------------------------
    # Backward-compat property
    # ------------------------------------------------------------------

    @property
    def last_update(self) -> torch.Tensor:
        return self.memory_ts

    @last_update.setter
    def last_update(self, value: torch.Tensor) -> None:
        self.memory_ts = value

    # ------------------------------------------------------------------
    # Basic read/write  (all accept global node IDs)
    # ------------------------------------------------------------------

    def gather(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Read memory vectors for the given global node IDs.

        Args:
            node_ids: 1-D tensor of global node IDs (any device).

        Returns:
            Tensor ``[len(node_ids), hidden_dim]`` on ``self.device``.
            Nodes not owned by this rank get zero vectors.
        """
        loc = self._to_local(node_ids)        # CPU
        valid = loc >= 0
        out = torch.zeros(node_ids.numel(), self.hidden_dim,
                          dtype=torch.float32, device=self.device)
        if valid.any():
            out[valid] = self.memory[loc[valid].to(self._storage_device)].to(self.device)
        return out

    def read_mailbox(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Read K-slot mailbox for the given global node IDs.

        Args:
            node_ids: 1-D tensor of global node IDs.

        Returns:
            Tensor ``[len(node_ids), K, slot_width]`` on ``self.device``.
        """
        loc = self._to_local(node_ids)        # CPU
        valid = loc >= 0
        out = torch.zeros(node_ids.numel(), self.mailbox_slots, self.slot_width,
                          dtype=torch.float32, device=self.device)
        if valid.any():
            out[valid] = self.mailbox[loc[valid].to(self._storage_device)].to(self.device)
        return out

    def gather_mailbox(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Legacy single-slot gather (slot 0)."""
        return self.read_mailbox(node_ids)[:, 0, :]

    def read_mailbox_ts(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return mailbox timestamps [M, K] on self.device."""
        loc = self._to_local(node_ids)
        valid = loc >= 0
        out = torch.zeros(node_ids.numel(), self.mailbox_slots, dtype=torch.float32, device=self.device)
        if valid.any():
            out[valid] = self.mailbox_ts[loc[valid].to(self._storage_device)].to(self.device)
        return out

    def write_mailbox(
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        src_mem: torch.Tensor,
        dst_mem: torch.Tensor,
        edge_feat: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Write new messages into the mailbox for src and dst nodes.

        Each message slot contains ``[src_mem || dst_mem || edge_feat]``
        concatenated.  Messages are written in timestamp order using a
        circular buffer (``next_mail_pos``).

        Args:
            src_ids: Source node global IDs, shape ``[B]``.
            dst_ids: Destination node global IDs, shape ``[B]``.
            src_mem: Source memory before update, shape ``[B, D]``.
            dst_mem: Destination memory before update, shape ``[B, D]``.
            edge_feat: Edge features, shape ``[B, E]``.
            timestamps: Event timestamps, shape ``[B]``.
        """
        device = self._storage_device
        src = src_ids.long().cpu()
        dst = dst_ids.long().cpu()
        smem  = src_mem.float().to(device)
        dmem  = dst_mem.float().to(device)
        ts    = timestamps.float().to(device)
        efeat = edge_feat.float().to(device)

        E = max(0, self.edge_feat_dim)
        efeat = efeat[..., :E] if efeat.size(-1) >= E else torch.cat(
            [efeat, torch.zeros(*efeat.shape[:-1], E - efeat.size(-1), device=device)], dim=-1)

        src_slot = torch.cat([smem, dmem, efeat], dim=-1)
        dst_slot = torch.cat([dmem, smem, efeat], dim=-1)
        self._write_slots(src, src_slot, ts)
        self._write_slots(dst, dst_slot, ts)
        self.mailbox_version += 1

    def _write_slots(self, global_ids: torch.Tensor, slot_values: torch.Tensor,
                     timestamps: torch.Tensor) -> None:
        sd = self._storage_device
        loc = self._to_local(global_ids)      # CPU
        valid = loc >= 0
        if not valid.any():
            return
        K  = self.mailbox_slots
        sw = slot_values.size(-1)
        lids = loc[valid].to(sd)
        sv   = slot_values[valid].to(sd)
        ts   = timestamps[valid].to(sd)

        order = torch.argsort(ts)
        lids = lids[order]; sv = sv[order]; ts = ts[order]

        pos      = self.next_mail_pos[lids]
        flat_idx = lids * K + pos
        self.mailbox.view(-1, sw)[flat_idx]  = sv
        self.mailbox_ts.view(-1)[flat_idx]   = ts

        uniq, inv = torch.unique(lids, return_inverse=True)
        counts = torch.zeros(uniq.numel(), dtype=torch.long, device=sd)
        counts.scatter_add_(0, inv, torch.ones_like(inv))
        self.next_mail_pos[uniq] = (self.next_mail_pos[uniq] + counts) % K

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def _apply_memory_update(self, node_ids: torch.Tensor, values: torch.Tensor,
                              timestamps: torch.Tensor) -> None:
        if node_ids.numel() == 0:
            return
        sd   = self._storage_device
        loc  = self._to_local(node_ids)       # CPU
        valid = loc >= 0
        if not valid.any():
            return
        lids = loc[valid].to(sd)
        vals = values[valid].float().to(sd)
        ts   = timestamps[valid].float().to(sd)
        order = torch.argsort(ts)
        self.memory[lids[order]]    = vals[order]
        self.memory_ts[lids[order]] = ts[order]
        self.memory_version += 1

    # Backward-compat wrapper
    def _apply_updates(self, node_ids, values, timestamps):
        self._apply_memory_update(node_ids, values, timestamps)

    def _apply_mail_update(self, node_ids: torch.Tensor, mail_slots: torch.Tensor, mail_ts: torch.Tensor) -> None:
        if node_ids.numel() == 0:
            return
        loc = self._to_local(node_ids)
        valid = loc >= 0
        if not valid.any():
            return
        lids = loc[valid].to(self._storage_device)
        slots = mail_slots[valid].to(self._storage_device)
        ts = mail_ts[valid].to(self._storage_device)
        incoming_latest = ts.max(dim=1).values
        current_latest = self.mailbox_ts[lids].max(dim=1).values
        newer = incoming_latest > current_latest
        if newer.any():
            lids_n = lids[newer]
            self.mailbox[lids_n] = slots[newer]
            self.mailbox_ts[lids_n] = ts[newer]
            self.mailbox_version += 1

    def _distributed_ready(self, ctx: DistributedContext) -> bool:
        if not ctx.is_distributed:
            return False
        import torch.distributed as dist
        return dist.is_initialized()

    def _exchange_counts(self, send_counts: list[int], device: torch.device) -> list[int]:
        import torch.distributed as dist

        send = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv = torch.zeros_like(send)
        dist.all_to_all_single(recv, send)
        return recv.tolist()

    def _pack_by_owner(
        self,
        owner: torch.Tensor,
        node_ids: torch.Tensor,
        payload: torch.Tensor,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if node_ids.numel() == 0:
            return node_ids, payload, [0] * world_size
        order = torch.argsort(owner)
        owner_sorted = owner[order]
        ids_sorted = node_ids[order]
        payload_sorted = payload[order]
        send_counts = torch.bincount(owner_sorted, minlength=world_size).tolist()
        return ids_sorted, payload_sorted, send_counts

    # ------------------------------------------------------------------
    # Async sync (distributed)
    # ------------------------------------------------------------------

    def submit_async_memory_sync(
        self,
        ctx: DistributedContext,
        node_ids: torch.Tensor,
        values: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Submit an async distributed memory sync for remote nodes.

        Packs updated memory for nodes owned by other ranks and sends via
        ``all_to_all_single``.  In single-rank mode, falls back to a local
        memory update.  Uses :class:`CTDGHistoricalCache` (if available) to
        skip syncs for nodes that haven't changed significantly.

        Args:
            ctx: :class:`DistributedContext` with rank / world_size.
            node_ids: Global node IDs with updated memory, shape ``[N]``.
            values: Updated memory vectors (detached), shape ``[N, D]``.
            timestamps: Update timestamps, shape ``[N]``.
        """
        if not self._distributed_ready(ctx):
            self._apply_memory_update(node_ids, values.detach(), timestamps.detach())
            return
        if node_ids.numel() == 0:
            return
        import torch.distributed as dist

        node_ids = node_ids.detach().long().to(self.device)
        vals = values.detach().float().to(self.device)
        ts = timestamps.detach().float().to(self.device)

        # Determine owner: use node_parts if available, else fall back to modulo
        if self.node_parts is not None:
            owner = self.node_parts[node_ids].long()
        else:
            owner = torch.remainder(node_ids, ctx.world_size).long()

        remote_mask = owner != ctx.rank
        if not remote_mask.any():
            return

        node_ids = node_ids[remote_mask]
        vals = vals[remote_mask]
        ts = ts[remote_mask]
        owner = owner[remote_mask]

        if self.historical_cache is not None:
            changed = self.historical_cache.historical_check(node_ids, vals)
            if not changed.any():
                return
            node_ids = node_ids[changed]
            vals = vals[changed]
            ts = ts[changed]
            owner = owner[changed]

        send_ids, send_payload, send_counts = self._pack_by_owner(
            owner,
            node_ids,
            torch.cat([vals, ts.unsqueeze(1)], dim=1),
            ctx.world_size,
        )
        recv_counts = self._exchange_counts(send_counts, device=node_ids.device)
        total_recv = int(sum(recv_counts))
        recv_ids = torch.empty(total_recv, dtype=torch.long, device=node_ids.device)
        width = self.hidden_dim + 1
        recv_payload_flat = torch.empty(total_recv * width, dtype=torch.float32, device=node_ids.device)

        id_work = dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            async_op=self.async_sync,
        )
        payload_work = dist.all_to_all_single(
            recv_payload_flat,
            send_payload.view(-1),
            output_split_sizes=[c * width for c in recv_counts],
            input_split_sizes=[c * width for c in send_counts],
            async_op=self.async_sync,
        )

        if self.async_sync:
            self._pending_memory_syncs.append(
                _PendingMemorySync(
                    id_work=id_work,
                    payload_work=payload_work,
                    recv_ids=recv_ids,
                    recv_payload=recv_payload_flat.view(total_recv, width),
                    recv_counts=recv_counts,
                    sent_remote_ids=node_ids,
                    sent_remote_vals=vals,
                    sent_remote_ts=ts,
                )
            )
            return

        recv_payload = recv_payload_flat.view(total_recv, width)
        merged_vals = recv_payload[:, : self.hidden_dim]
        merged_ts = recv_payload[:, self.hidden_dim]
        self._apply_memory_update(recv_ids, merged_vals, merged_ts)
        if self.historical_cache is not None and node_ids.numel() > 0:
            slot_ids = self.historical_cache.slot_indices(node_ids)
            valid_slot = slot_ids >= 0
            if valid_slot.any():
                self.historical_cache.update_cache(slot_ids[valid_slot], vals[valid_slot], ts[valid_slot])

    def submit_async_mail_sync(
        self,
        ctx: DistributedContext,
        node_ids: torch.Tensor,
        mail_slots: torch.Tensor,
        mail_ts: torch.Tensor,
    ) -> None:
        """No-op for single rank; distributed path mirrors memory sync."""
        if not self._distributed_ready(ctx):
            return
        if node_ids.numel() == 0:
            return
        import torch.distributed as dist

        node_ids = node_ids.detach().long().to(self.device)
        slots = mail_slots.detach().float().to(self.device)
        ts = mail_ts.detach().float().to(self.device)
        owner = torch.remainder(node_ids, ctx.world_size).long()
        remote_mask = owner != ctx.rank
        if not remote_mask.any():
            return
        node_ids = node_ids[remote_mask]
        slots = slots[remote_mask]
        ts = ts[remote_mask]
        owner = owner[remote_mask]

        payload_width = self.mailbox_slots * self.slot_width + self.mailbox_slots
        mail_payload = torch.cat([slots.view(node_ids.numel(), -1), ts.view(node_ids.numel(), -1)], dim=1)
        send_ids, send_payload, send_counts = self._pack_by_owner(
            owner,
            node_ids,
            mail_payload,
            ctx.world_size,
        )
        recv_counts = self._exchange_counts(send_counts, device=node_ids.device)
        total_recv = int(sum(recv_counts))
        recv_ids = torch.empty(total_recv, dtype=torch.long, device=node_ids.device)
        recv_payload_flat = torch.empty(total_recv * payload_width, dtype=torch.float32, device=node_ids.device)

        id_work = dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            async_op=self.async_sync,
        )
        payload_work = dist.all_to_all_single(
            recv_payload_flat,
            send_payload.view(-1),
            output_split_sizes=[c * payload_width for c in recv_counts],
            input_split_sizes=[c * payload_width for c in send_counts],
            async_op=self.async_sync,
        )

        if self.async_sync:
            self._pending_mail_syncs.append(
                _PendingMailSync(
                    id_work=id_work,
                    payload_work=payload_work,
                    recv_ids=recv_ids,
                    recv_payload=recv_payload_flat.view(total_recv, payload_width),
                    recv_counts=recv_counts,
                )
            )
            return

        recv_payload = recv_payload_flat.view(total_recv, payload_width)
        recv_slots = recv_payload[:, : self.mailbox_slots * self.slot_width].view(total_recv, self.mailbox_slots, self.slot_width)
        recv_ts = recv_payload[:, self.mailbox_slots * self.slot_width :].view(total_recv, self.mailbox_slots)
        self._apply_mail_update(recv_ids, recv_slots, recv_ts)

    def wait_pending_syncs(self) -> None:
        """Block until all pending async memory and mailbox syncs complete.

        Drains ``_pending_memory_syncs`` and ``_pending_mail_syncs``,
        applies received updates to local storage, and updates the
        historical cache if present.
        """
        while self._pending_memory_syncs:
            sync = self._pending_memory_syncs.pop(0)
            sync.id_work.wait()
            sync.payload_work.wait()
            recv_vals = sync.recv_payload[:, : self.hidden_dim]
            recv_ts = sync.recv_payload[:, self.hidden_dim]
            self._apply_memory_update(sync.recv_ids, recv_vals, recv_ts)
            self.last_memory_sync = (
                int(sync.recv_ids.numel()),
                tuple(sync.recv_counts),
            )
            if self.historical_cache is not None and sync.sent_remote_ids is not None and sync.sent_remote_ids.numel() > 0:
                slot_ids = self.historical_cache.slot_indices(sync.sent_remote_ids)
                valid_slot = slot_ids >= 0
                if valid_slot.any():
                    assert sync.sent_remote_vals is not None and sync.sent_remote_ts is not None
                    self.historical_cache.update_cache(
                        slot_ids[valid_slot],
                        sync.sent_remote_vals[valid_slot],
                        sync.sent_remote_ts[valid_slot],
                    )

        while self._pending_mail_syncs:
            sync = self._pending_mail_syncs.pop(0)
            sync.id_work.wait()
            sync.payload_work.wait()
            total_recv = int(sync.recv_ids.numel())
            recv_slots = sync.recv_payload[:, : self.mailbox_slots * self.slot_width].view(
                total_recv,
                self.mailbox_slots,
                self.slot_width,
            )
            recv_ts = sync.recv_payload[:, self.mailbox_slots * self.slot_width :].view(total_recv, self.mailbox_slots)
            self._apply_mail_update(sync.recv_ids, recv_slots, recv_ts)
            self.last_mail_sync = (
                total_recv,
                tuple(sync.recv_counts),
            )

    # ------------------------------------------------------------------
    # Legacy sync
    # ------------------------------------------------------------------

    def sync_updates(self, ctx, node_ids, values, timestamps):
        self.submit_async_memory_sync(ctx, node_ids, values, timestamps)

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_local_nodes": self.num_local_nodes,
            "hidden_dim": self.hidden_dim,
            "mailbox_slots": self.mailbox_slots,
            "memory_version": self.memory_version,
            "storage_device": str(self._storage_device),
        }
