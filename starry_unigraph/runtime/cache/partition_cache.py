"""PartitionCache: 邻居分区 embedding 缓存（DTDG 侧）。

DTDG 快照间拓扑固定，邻居分区的 embedding 可跨快照复用。
PartitionCache 按分区（rank）组织缓存，每个快照后更新对应分区的 embedding 块。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class PartitionCache:
    """按 rank 分区组织的 embedding 缓存（DTDG 专用）。

    DTDG 快照拓扑固定，CommPlan 静态预编译。
    每次 all_to_all 后将收到的邻居分区 embedding 写入 PartitionCache，
    下一快照的 prepare_data 直接从缓存读取（若未更新则沿用上一快照）。

    Attributes:
        world_size:   总 rank 数
        feat_dim:     embedding 维度
        _partition:   rank → (node_ids, embeddings) 的字典
        _staleness:   rank → 上次更新的快照索引（用于检测过期）
    """

    def __init__(self, world_size: int, feat_dim: int, device: torch.device) -> None:
        """初始化 PartitionCache。

        Args:
            world_size: 总 rank 数
            feat_dim:   embedding 维度
            device:     缓存所在 CUDA 设备
        """
        self.world_size = world_size
        self.feat_dim = feat_dim
        self.device = device
        self._partition: Dict[int, Tuple[Tensor, Tensor]] = {}
        self._staleness: Dict[int, int] = {}

    def lookup(self, rank: int, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """从指定 rank 的分区缓存中查询节点 embedding。

        Args:
            rank:     来源 rank
            node_ids: [N] 需要查询的节点全局 ID

        Returns:
            (hit_mask, embeddings)
              hit_mask:   [N] bool，True 表示节点在缓存中
              embeddings: [N, d]，命中节点的 embedding（未命中为 0）
        """
        raise NotImplementedError("PartitionCache.lookup() 待实现")

    def update(self, rank: int, node_ids: Tensor, embeddings: Tensor, snapshot_idx: int) -> None:
        """更新来自指定 rank 的分区 embedding 块。

        Args:
            rank:         来源 rank
            node_ids:     [N] 节点全局 ID
            embeddings:   [N, d] 新 embedding 向量
            snapshot_idx: 当前快照索引（用于 staleness 追踪）

        Notes:
            - 通常在 all_to_all 完成后（handle.wait()）调用
            - 覆盖写入，不做合并
        """
        raise NotImplementedError("PartitionCache.update() 待实现")

    def evict(self, rank: Optional[int] = None) -> None:
        """清除指定 rank（或全部）的缓存。

        Args:
            rank: 若为 None，清除所有分区缓存

        Notes:
            - epoch 结束时调用以释放显存
            - 也可在 rank 断开连接时调用
        """
        raise NotImplementedError("PartitionCache.evict() 待实现")
