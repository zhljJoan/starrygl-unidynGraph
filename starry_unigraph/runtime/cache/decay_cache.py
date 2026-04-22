"""DecayCache: 衰减缓存，由 reuse_table 驱动。

为 CTDG 场景设计：基于 chunk 序列的重用分析（reuse_table），
预测哪些节点在接下来的 K 个 chunk 内会被再次访问，按优先级缓存。
超过重用窗口的节点自动衰减淘汰。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class DecayCache:
    """衰减缓存：基于 reuse_table 的预测性缓存。

    reuse_table[node_id] = next_access_chunk_idx：记录每个节点下次被访问的 chunk 索引。
    缓存策略：优先保留 next_access_chunk_idx 最小（即最快被访问）的节点。
    step() 时更新 reuse_table，驱逐 next_access_chunk_idx < current_idx 的过期节点。

    Attributes:
        capacity:     缓存容量（节点数上限）
        feat_dim:     特征维度
        reuse_table:  node_id → next_access_chunk_idx（由离线分析填充）
        _keys:        当前缓存节点 ID
        _values:      对应特征向量
        _next_access: 对应的 next_access_chunk_idx（用于优先级排序）
    """

    def __init__(
        self,
        capacity: int,
        feat_dim: int,
        device: torch.device,
        reuse_table: Optional[Dict[int, int]] = None,
    ) -> None:
        """初始化 DecayCache。

        Args:
            capacity:     最大缓存节点数
            feat_dim:     特征维度
            device:       缓存所在设备
            reuse_table:  预计算的重用表，node_id → next_access_chunk_idx
                          若为 None，则退化为 LRU 策略
        """
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.device = device
        self.reuse_table: Dict[int, int] = reuse_table or {}
        self._keys: Optional[Tensor] = None
        self._values: Optional[Tensor] = None
        self._next_access: Optional[Tensor] = None

    def lookup(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """查询节点是否在衰减缓存中。

        Args:
            node_ids: [N] 查询节点全局 ID

        Returns:
            (hit_mask, cached_values)
              hit_mask:      [N] bool Tensor
              cached_values: [N, d] 命中节点特征（未命中位置为 0）
        """
        raise NotImplementedError("DecayCache.lookup() 待实现")

    def update(self, node_ids: Tensor, values: Tensor) -> None:
        """更新节点的缓存值，并根据 reuse_table 设置优先级。

        Args:
            node_ids: [N] 节点全局 ID
            values:   [N, d] 新特征向量

        Notes:
            - 若 node_id 在 reuse_table 中，使用其 next_access 作为优先级
            - 否则使用 current_chunk_idx + 1（假设下一个 chunk 访问）
            - 缓存满时替换 next_access 最大（最晚被访问）的节点
        """
        raise NotImplementedError("DecayCache.update() 待实现")

    def evict(self, current_chunk_idx: int) -> int:
        """驱逐 next_access_chunk_idx < current_chunk_idx 的过期节点。

        Args:
            current_chunk_idx: 当前 chunk 在序列中的索引

        Returns:
            驱逐的节点数量
        """
        raise NotImplementedError("DecayCache.evict() 待实现")

    def step(self, current_chunk_idx: int) -> None:
        """chunk 切换时调用，更新缓存优先级并淘汰过期节点。

        Args:
            current_chunk_idx: 切换后的新 chunk 索引

        Notes:
            1. evict(current_chunk_idx) 驱逐过期节点
            2. 对仍在缓存中的节点，从 reuse_table 更新 next_access
            3. 为下一 chunk 的预期 miss 节点预留空间（可选）
        """
        raise NotImplementedError("DecayCache.step() 待实现")
