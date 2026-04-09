from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from starry_unigraph.types import DistributedContext


@dataclass
class AsyncExchangeHandle:
    """Handle for a non-blocking all_to_all_single exchange."""

    handle: Any           # dist.Work
    output_buffer: torch.Tensor  # flat packed output
    recv_counts: list[int]
    slot_width: int       # ids(1) + values(D) per node
    value_dim: int        # D
    device: str

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.handle.wait()
        ids_list = []
        vals_list = []
        offset = 0
        for rc in self.recv_counts:
            if rc == 0:
                continue
            chunk = self.output_buffer[offset * self.slot_width:(offset + rc) * self.slot_width]
            chunk = chunk.view(rc, self.slot_width)
            ids_list.append(chunk[:, 0].long())
            vals_list.append(chunk[:, 1:1 + self.value_dim])
            offset += rc
        if not ids_list:
            dev = self.device
            return (
                torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, self.value_dim, dtype=torch.float32, device=dev),
            )
        return torch.cat(ids_list, dim=0), torch.cat(vals_list, dim=0)


@dataclass
class CTDGFeatureRoute:
    route_type: str
    world_size: int
    replicated_memory: bool = True

    def describe(self) -> dict[str, Any]:
        return asdict(self) | {"type": "CTDGFeatureRoute"}

    def exchange(
        self,
        ctx: DistributedContext,
        node_ids: torch.Tensor,
        values: torch.Tensor,
        async_op: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | AsyncExchangeHandle:
        if not ctx.is_distributed:
            return node_ids, values
        import torch.distributed as dist

        if not dist.is_initialized():
            return node_ids, values

        # Use fast all_to_all_single path only for nccl + GPU tensors.
        if dist.get_backend() == "nccl" and values.is_cuda:
            return self._fast_exchange(ctx, node_ids, values, async_op=async_op)

        # Tensor-only fallback (no all_gather_object).
        return self._tensor_exchange(ctx, node_ids, values)

    def _fast_exchange(
        self,
        ctx: DistributedContext,
        node_ids: torch.Tensor,
        values: torch.Tensor,
        async_op: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | AsyncExchangeHandle:
        import torch.distributed as dist

        dev = str(values.device)
        D = values.size(-1)
        n = node_ids.numel()
        world_size = ctx.world_size

        # Step 1: exchange counts
        recv_counts = self._exchange_counts(ctx, n, device=values.device)

        # Step 2: pack flat tensor [node_id(1) | values(D)]
        slot_w = 1 + D
        send_buf = torch.zeros(n, slot_w, dtype=torch.float32, device=values.device)
        send_buf[:, 0] = node_ids.float().to(values.device)
        send_buf[:, 1:] = values.float()
        send_flat = send_buf.view(-1)

        total_recv = sum(recv_counts)
        recv_flat = torch.zeros(total_recv * slot_w, dtype=torch.float32, device=values.device)

        send_splits = [n * slot_w] * world_size
        recv_splits = [rc * slot_w for rc in recv_counts]

        handle = dist.all_to_all_single(
            recv_flat,
            send_flat,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            async_op=True,
        )

        if async_op:
            return AsyncExchangeHandle(
                handle=handle,
                output_buffer=recv_flat,
                recv_counts=recv_counts,
                slot_width=slot_w,
                value_dim=D,
                device=dev,
            )

        # Blocking: wait and unpack
        async_handle = AsyncExchangeHandle(
            handle=handle,
            output_buffer=recv_flat,
            recv_counts=recv_counts,
            slot_width=slot_w,
            value_dim=D,
            device=dev,
        )
        merged_ids, merged_values = async_handle.wait()
        return merged_ids, merged_values

    def _exchange_counts(
        self,
        ctx: DistributedContext,
        n: int,
        device: torch.device | str = "cpu",
    ) -> list[int]:
        import torch.distributed as dist

        send_count = torch.tensor([n], dtype=torch.long, device=device)
        recv_counts_t = torch.zeros(ctx.world_size, dtype=torch.long, device=device)
        dist.all_gather_into_tensor(recv_counts_t, send_count)
        return recv_counts_t.tolist()

    def _tensor_exchange(
        self,
        ctx: DistributedContext,
        node_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch.distributed as dist

        dev = values.device
        dtype = values.dtype
        width = int(values.size(-1))
        local_n = int(node_ids.numel())

        send_count = torch.tensor([local_n], dtype=torch.long, device=dev)
        recv_counts_t = torch.zeros(ctx.world_size, dtype=torch.long, device=dev)
        dist.all_gather_into_tensor(recv_counts_t, send_count)
        recv_counts = recv_counts_t.tolist()
        max_n = int(max(recv_counts) if recv_counts else 0)
        if max_n == 0:
            return (
                torch.empty(0, dtype=torch.long, device=node_ids.device),
                torch.empty(0, width, dtype=dtype, device=values.device),
            )

        pad_ids = torch.zeros(max_n, dtype=torch.long, device=dev)
        pad_vals = torch.zeros(max_n, width, dtype=dtype, device=dev)
        if local_n > 0:
            pad_ids[:local_n] = node_ids.to(dev).long()
            pad_vals[:local_n] = values.to(dev)

        ids_buf = [torch.zeros_like(pad_ids) for _ in range(ctx.world_size)]
        vals_buf = [torch.zeros_like(pad_vals) for _ in range(ctx.world_size)]
        dist.all_gather(ids_buf, pad_ids)
        dist.all_gather(vals_buf, pad_vals)

        merged_ids_chunks: list[torch.Tensor] = []
        merged_vals_chunks: list[torch.Tensor] = []
        for rank, rc in enumerate(recv_counts):
            if rc <= 0:
                continue
            merged_ids_chunks.append(ids_buf[rank][:rc])
            merged_vals_chunks.append(vals_buf[rank][:rc])

        if not merged_ids_chunks:
            return (
                torch.empty(0, dtype=torch.long, device=node_ids.device),
                torch.empty(0, width, dtype=dtype, device=values.device),
            )
        merged_ids = torch.cat(merged_ids_chunks, dim=0).to(node_ids.device)
        merged_values = torch.cat(merged_vals_chunks, dim=0).to(values.device)
        return merged_ids, merged_values
