"""Mailbox: Historical message cache for CTDG temporal graph learning.

Implements MemShare-style mailbox for storing and retrieving historical messages.
Supports distributed training with local/remote message management.

Reference: ~/MemShare-public/starrygl/module/historical_cache.py
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "Mailbox",
    "MailboxConfig",
]


class MailboxConfig:
    """Configuration for Mailbox.

    Args:
        num_nodes: Total number of nodes
        memory_dim: Dimension of memory/message vectors
        cache_size: Maximum number of messages per node
        device: Target device
        deliver_to: "self" (only to dst) or "neighbors" (to src and dst)
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        cache_size: int = 10,
        device: str = "cuda",
        deliver_to: str = "self",
    ):
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.cache_size = cache_size
        self.device = device
        self.deliver_to = deliver_to


class Mailbox:
    """Mailbox for storing historical messages.

    Stores messages for each node with timestamps. Supports:
    - Local storage (this rank's owned nodes)
    - Remote storage (other ranks' nodes, for distributed training)
    - Message retrieval with timestamp filtering
    - Message aggregation (mean, last, max)

    Args:
        config: MailboxConfig
    """

    def __init__(self, config: MailboxConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Local mailbox: node_id -> deque of (message, timestamp)
        self.local_mailbox: Dict[int, deque] = {}

        # Local memory cache: [num_nodes, memory_dim]
        self.local_memory = torch.zeros(
            config.num_nodes,
            config.memory_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # Local memory timestamps: [num_nodes]
        self.local_memory_ts = torch.zeros(
            config.num_nodes,
            dtype=torch.float32,
            device=self.device,
        )

        # Deliver mode
        self.deliver_to = config.deliver_to

    def push(
        self,
        node_ids: Tensor,
        messages: Tensor,
        timestamps: Tensor,
    ) -> None:
        """Store messages to mailbox.

        Args:
            node_ids: [num_msgs] Node IDs
            messages: [num_msgs, memory_dim] Messages
            timestamps: [num_msgs] Timestamps
        """
        node_ids = node_ids.cpu().numpy()
        messages = messages.cpu()
        timestamps = timestamps.cpu()

        for i, nid in enumerate(node_ids):
            nid = int(nid)
            if nid not in self.local_mailbox:
                self.local_mailbox[nid] = deque(maxlen=self.config.cache_size)

            self.local_mailbox[nid].append((messages[i], timestamps[i].item()))

    def get(
        self,
        node_ids: Tensor,
        before_timestamp: Optional[Tensor] = None,
        aggregation: str = "last",
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve messages from mailbox.

        Args:
            node_ids: [num_nodes] Node IDs to query
            before_timestamp: [num_nodes] Only return messages before this time
            aggregation: "last", "mean", or "max"

        Returns:
            (messages, timestamps):
                messages: [num_nodes, memory_dim] Aggregated messages
                timestamps: [num_nodes] Message timestamps
        """
        node_ids_cpu = node_ids.cpu().numpy()
        num_nodes = len(node_ids_cpu)

        messages = torch.zeros(
            num_nodes,
            self.config.memory_dim,
            dtype=torch.float32,
        )
        timestamps = torch.zeros(num_nodes, dtype=torch.float32)

        for i, nid in enumerate(node_ids_cpu):
            nid = int(nid)
            if nid not in self.local_mailbox or len(self.local_mailbox[nid]) == 0:
                # No messages - return zero
                continue

            # Filter by timestamp
            if before_timestamp is not None:
                threshold = before_timestamp[i].item()
                valid_msgs = [
                    (msg, ts)
                    for msg, ts in self.local_mailbox[nid]
                    if ts < threshold
                ]
            else:
                valid_msgs = list(self.local_mailbox[nid])

            if len(valid_msgs) == 0:
                continue

            # Aggregate
            if aggregation == "last":
                messages[i] = valid_msgs[-1][0]
                timestamps[i] = valid_msgs[-1][1]
            elif aggregation == "mean":
                msg_stack = torch.stack([msg for msg, _ in valid_msgs], dim=0)
                messages[i] = msg_stack.mean(dim=0)
                timestamps[i] = valid_msgs[-1][1]
            elif aggregation == "max":
                msg_stack = torch.stack([msg for msg, _ in valid_msgs], dim=0)
                messages[i] = msg_stack.max(dim=0)[0]
                timestamps[i] = valid_msgs[-1][1]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        return messages.to(self.device), timestamps.to(self.device)

    def set_memory_local(
        self,
        node_ids: Tensor,
        memory: Tensor,
        timestamps: Tensor,
        reduce_op: str = "max",
    ) -> None:
        """Update local memory cache.

        Args:
            node_ids: [num_nodes] Local node IDs
            memory: [num_nodes, memory_dim] New memory
            timestamps: [num_nodes] Timestamps
            reduce_op: "max" (keep latest) or "mean"
        """
        if reduce_op == "max":
            # Keep latest memory
            mask = timestamps > self.local_memory_ts[node_ids]
            update_ids = node_ids[mask]
            self.local_memory[update_ids] = memory[mask]
            self.local_memory_ts[update_ids] = timestamps[mask]
        elif reduce_op == "mean":
            # Average with existing memory
            self.local_memory[node_ids] = (
                self.local_memory[node_ids] + memory
            ) / 2.0
            self.local_memory_ts[node_ids] = torch.maximum(
                self.local_memory_ts[node_ids], timestamps
            )
        else:
            raise ValueError(f"Unknown reduce_op: {reduce_op}")

    def set_mailbox_local(
        self,
        node_ids: Tensor,
        messages: Tensor,
        timestamps: Tensor,
        reduce_op: str = "max",
    ) -> None:
        """Update local mailbox.

        Args:
            node_ids: [num_msgs] Local node IDs
            messages: [num_msgs, memory_dim] Messages
            timestamps: [num_msgs] Timestamps
            reduce_op: "max" (keep latest) or "append"
        """
        if reduce_op == "max":
            # Only keep latest message per node
            for i, nid in enumerate(node_ids.cpu().numpy()):
                nid = int(nid)
                if nid not in self.local_mailbox:
                    self.local_mailbox[nid] = deque(maxlen=self.config.cache_size)

                # Check if we should update
                if len(self.local_mailbox[nid]) > 0:
                    last_ts = self.local_mailbox[nid][-1][1]
                    if timestamps[i].item() > last_ts:
                        self.local_mailbox[nid].append(
                            (messages[i].cpu(), timestamps[i].item())
                        )
                else:
                    self.local_mailbox[nid].append(
                        (messages[i].cpu(), timestamps[i].item())
                    )
        else:
            # Append all messages
            self.push(node_ids, messages, timestamps)

    def get_memory(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Get memory from local cache.

        Args:
            node_ids: [num_nodes] Node IDs

        Returns:
            (memory, timestamps):
                memory: [num_nodes, memory_dim]
                timestamps: [num_nodes]
        """
        return self.local_memory[node_ids], self.local_memory_ts[node_ids]

    def clear(self, node_ids: Optional[Tensor] = None) -> None:
        """Clear mailbox.

        Args:
            node_ids: Optional node IDs to clear. If None, clear all.
        """
        if node_ids is None:
            self.local_mailbox.clear()
            self.local_memory.zero_()
            self.local_memory_ts.zero_()
        else:
            for nid in node_ids.cpu().numpy():
                nid = int(nid)
                if nid in self.local_mailbox:
                    self.local_mailbox[nid].clear()
                self.local_memory[nid].zero_()
                self.local_memory_ts[nid] = 0.0

    def get_update_mail(
        self,
        node_ids: Tensor,
        src: Tensor,
        dst: Tensor,
        timestamps: Tensor,
        edge_feats: Tensor,
        memory: Tensor,
        node_feats: Optional[Tensor] = None,
        compute_message: bool = True,
        aggregate: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute and retrieve messages for nodes.

        Args:
            node_ids: [num_nodes] Query node IDs
            src: [num_edges] Edge source nodes
            dst: [num_edges] Edge destination nodes
            timestamps: [num_edges] Edge timestamps
            edge_feats: [num_edges, edge_dim] Edge features
            memory: [num_nodes, memory_dim] Current memory
            node_feats: Optional node features
            compute_message: Whether to compute new messages
            aggregate: Whether to aggregate messages

        Returns:
            (update_ids, messages, message_ts):
                update_ids: Node IDs with messages
                messages: Aggregated messages
                message_ts: Message timestamps
        """
        # Simplified implementation - full version would compute messages
        # from edges and aggregate them

        # For now, just return existing mailbox messages
        messages, message_ts = self.get(node_ids, aggregation="last")

        return node_ids, messages, message_ts

    def __len__(self) -> int:
        """Return number of nodes with messages."""
        return len(self.local_mailbox)

    def __repr__(self) -> str:
        return (
            f"Mailbox(num_nodes={self.config.num_nodes}, "
            f"memory_dim={self.config.memory_dim}, "
            f"cache_size={self.config.cache_size}, "
            f"cached_nodes={len(self.local_mailbox)})"
        )
