"""HotCache: 热点节点常驻缓存。

存储访问频率最高的节点 embedding/memory，常驻显存，
避免跨 rank 通信。适用于 CTDG 的 owned_memory 热点分层。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class HotCache:
    """热点节点常驻缓存。

    将访问频率超过阈值的节点的 embedding/memory 常驻显存，
    lookup 命中时直接返回，无需通信。

    Attributes:
        capacity:   缓存容量（节点数上限）
        feat_dim:   特征维度 d
        _keys:      当前缓存的节点全局 ID，shape = [capacity]
        _values:    对应特征向量，shape = [capacity, d]
        _freq:      访问频次统计，shape = [capacity]
    """

    def __init__(self, capacity: int, feat_dim: int, device: torch.device) -> None:
        """初始化 HotCache。

        Args:
            capacity:  最大缓存节点数
            feat_dim:  特征/embedding 维度
            device:    缓存所在 CUDA 设备
        """
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.device = device
        # 延迟初始化，首次 update 时分配
        self._keys: Optional[Tensor] = None
        self._values: Optional[Tensor] = None
        self._freq: Optional[Tensor] = None

    def lookup(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """查询节点是否在热点缓存中。

        Args:
            node_ids: [N] 查询节点全局 ID

        Returns:
            (hit_mask, cached_values)
              hit_mask:      [N] bool Tensor，True 表示缓存命中
              cached_values: [N, d] 命中节点的特征（未命中位置为 0）

        Raises:
            NotImplementedError: 待实现（建议用 hash table 或排序+二分查找）
        """
        raise NotImplementedError("HotCache.lookup() 待实现")

    def update(self, node_ids: Tensor, values: Tensor) -> None:
        """更新热点节点的缓存值，并增加访问频次。

        Args:
            node_ids: [N] 节点全局 ID
            values:   [N, d] 新特征向量

        Notes:
            - 若节点不在缓存中且缓存未满，直接插入
            - 若缓存已满，替换频次最低的节点（LFU 策略）
        """
        raise NotImplementedError("HotCache.update() 待实现")

    def evict(self, threshold: Optional[int] = None) -> int:
        """驱逐低频节点，释放缓存空间。

        Args:
            threshold: 频次阈值，低于此值的节点被驱逐。
                       若为 None，驱逐频次最低的 10% 节点。

        Returns:
            实际驱逐的节点数量
        """
        raise NotImplementedError("HotCache.evict() 待实现")
