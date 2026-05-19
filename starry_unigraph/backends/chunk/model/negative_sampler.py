"""Negative sampling for edge prediction tasks.

Provides two sampling strategies:
- train: local-biased sampling (prefer negatives from local chunk)
- test: global-average sampling (uniform across all nodes)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor


class EdgePredictNegativeSampler:
    """Negative sampler for temporal edge prediction.

    Supports two modes:
    - local_biased: Train mode, samples negatives preferentially from local chunk
    - global_average: Test mode, samples uniformly from all nodes

    Args:
        num_nodes: Total number of nodes in the graph
        node_to_chunk: [num_nodes] Chunk assignment for each node
        chunk_to_owner: [num_chunks] Owner partition for each chunk
        rank: Current partition rank
        num_negatives: Number of negative samples per positive edge
        local_bias_ratio: Fraction of negatives from local chunk (train mode)
    """

    def __init__(
        self,
        num_nodes: int,
        node_to_chunk: Tensor,
        chunk_to_owner: Tensor,
        rank: int,
        num_negatives: int = 1,
        local_bias_ratio: float = 0.7,
    ):
        self.num_nodes = num_nodes
        self.node_to_chunk = node_to_chunk
        self.chunk_to_owner = chunk_to_owner
        self.rank = rank
        self.num_negatives = num_negatives
        self.local_bias_ratio = local_bias_ratio

        # Precompute local node set
        node_owners = chunk_to_owner[node_to_chunk]
        self.local_nodes = (node_owners == rank).nonzero(as_tuple=True)[0]
        self.num_local_nodes = int(self.local_nodes.numel())

    def sample(
        self,
        pos_dst: Tensor,
        mode: Literal["local_biased", "global_average"] = "local_biased",
    ) -> Tensor:
        """Sample negative destination nodes for positive edges.

        Args:
            pos_dst: [B] Positive destination node IDs
            mode: Sampling strategy

        Returns:
            neg_dst: [B, num_negatives] Negative destination node IDs
        """
        if mode == "local_biased":
            return self._sample_local_biased(pos_dst)
        elif mode == "global_average":
            return self._sample_global_average(pos_dst)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

    def _sample_local_biased(self, pos_dst: Tensor) -> Tensor:
        """Train mode: Sample negatives with local bias.

        For each positive edge, sample:
        - local_bias_ratio * num_negatives from local chunk
        - (1 - local_bias_ratio) * num_negatives from global pool

        Args:
            pos_dst: [B] Positive destination nodes

        Returns:
            neg_dst: [B, num_negatives] Negative samples
        """
        batch_size = int(pos_dst.numel())
        device = pos_dst.device

        num_local = int(self.num_negatives * self.local_bias_ratio)
        num_global = self.num_negatives - num_local

        neg_samples = []

        if num_local > 0 and self.num_local_nodes > 0:
            # Sample from local nodes
            local_indices = torch.randint(
                0, self.num_local_nodes,
                (batch_size, num_local),
                device=device,
            )
            local_neg = self.local_nodes[local_indices]
            neg_samples.append(local_neg)

        if num_global > 0:
            # Sample from all nodes
            global_neg = torch.randint(
                0, self.num_nodes,
                (batch_size, num_global),
                device=device,
            )
            neg_samples.append(global_neg)

        if len(neg_samples) == 0:
            # Fallback: sample globally
            return torch.randint(
                0, self.num_nodes,
                (batch_size, self.num_negatives),
                device=device,
            )

        neg_dst = torch.cat(neg_samples, dim=1)

        # Shuffle negatives within each row
        perm = torch.argsort(torch.rand(batch_size, self.num_negatives, device=device), dim=1)
        neg_dst = torch.gather(neg_dst, 1, perm)

        return neg_dst

    def _sample_global_average(self, pos_dst: Tensor) -> Tensor:
        """Test mode: Sample negatives uniformly from all nodes.

        Args:
            pos_dst: [B] Positive destination nodes

        Returns:
            neg_dst: [B, num_negatives] Negative samples
        """
        batch_size = int(pos_dst.numel())
        device = pos_dst.device

        neg_dst = torch.randint(
            0, self.num_nodes,
            (batch_size, self.num_negatives),
            device=device,
        )

        return neg_dst

    def sample_with_filter(
        self,
        pos_dst: Tensor,
        filter_nodes: Optional[Tensor] = None,
        mode: Literal["local_biased", "global_average"] = "local_biased",
        max_retries: int = 10,
    ) -> Tensor:
        """Sample negatives with filtering to avoid specific nodes.

        Args:
            pos_dst: [B] Positive destination nodes
            filter_nodes: [F] Nodes to exclude from sampling (e.g., true positives)
            mode: Sampling strategy
            max_retries: Maximum rejection sampling attempts

        Returns:
            neg_dst: [B, num_negatives] Negative samples
        """
        neg_dst = self.sample(pos_dst, mode=mode)

        if filter_nodes is None or filter_nodes.numel() == 0:
            return neg_dst

        # Rejection sampling: resample if negative matches filter set
        filter_set = set(filter_nodes.cpu().tolist())
        batch_size = int(pos_dst.numel())

        for retry in range(max_retries):
            # Check which samples need resampling
            needs_resample = torch.zeros(batch_size, self.num_negatives, dtype=torch.bool, device=neg_dst.device)
            for i in range(batch_size):
                for j in range(self.num_negatives):
                    if int(neg_dst[i, j].item()) in filter_set:
                        needs_resample[i, j] = True

            if not needs_resample.any():
                break

            # Resample only the problematic entries
            num_resample = int(needs_resample.sum().item())
            if mode == "local_biased" and self.num_local_nodes > 0:
                local_indices = torch.randint(0, self.num_local_nodes, (num_resample,), device=neg_dst.device)
                resampled = self.local_nodes[local_indices]
            else:
                resampled = torch.randint(0, self.num_nodes, (num_resample,), device=neg_dst.device)

            neg_dst[needs_resample] = resampled

        return neg_dst


def create_negative_sampler(
    num_nodes: int,
    node_to_chunk: Tensor,
    chunk_to_owner: Tensor,
    rank: int,
    num_negatives: int = 1,
    local_bias_ratio: float = 0.7,
) -> EdgePredictNegativeSampler:
    """Factory function to create a negative sampler.

    Args:
        num_nodes: Total number of nodes
        node_to_chunk: [num_nodes] Chunk assignment
        chunk_to_owner: [num_chunks] Owner partition
        rank: Current partition rank
        num_negatives: Number of negatives per positive
        local_bias_ratio: Local bias ratio for train mode

    Returns:
        EdgePredictNegativeSampler instance
    """
    return EdgePredictNegativeSampler(
        num_nodes=num_nodes,
        node_to_chunk=node_to_chunk,
        chunk_to_owner=chunk_to_owner,
        rank=rank,
        num_negatives=num_negatives,
        local_bias_ratio=local_bias_ratio,
    )
