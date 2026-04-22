"""StateDeps: 跨时间状态读写依赖描述。

记录一个 ChunkAtomic 在状态层（embedding/memory）上的跨 rank 依赖，
用于 CommEngine.async_push_delta() 和 async_fetch()。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class StateDeps:
    """chunk 声明：我的计算涉及的跨 rank 状态读写依赖。

    Fields:
        owner_map:          node_id → owner_rank 的映射（稀疏，仅含跨 rank 节点）
                            实现可为 dict 或 (ids, ranks) 两个 Tensor
        delta_accumulator:  本 chunk 计算后需要推送给其他 rank 的增量 buffer
                            shape = [N_write, d]，由 StateManager.update() 填充

    Notes:
        - DTDG: owner_map 静态（节点归属分区固定），delta 为 embedding 更新
        - CTDG: owner_map 随采样动态变化，delta 为 memory 增量
        - CommEngine.async_push_delta() 使用此结构决定向哪些 rank 推送什么
    """
    # node_id → owner_rank 映射，仅含跨 rank 节点
    owner_node_ids: Optional[Tensor] = None    # [N_cross] 跨 rank 节点全局 ID
    owner_ranks: Optional[Tensor] = None       # [N_cross] 对应 owner rank

    # 本 chunk 计算后需要向外推送的增量
    delta_node_ids: Optional[Tensor] = None    # [N_write] 需要推送更新的节点 ID
    delta_values: Optional[Tensor] = None      # [N_write, d] 对应增量值（由 update() 填充）

    def get_owner(self, node_id: int) -> int:
        """查询节点的 owner rank。

        Args:
            node_id: 全局节点 ID

        Returns:
            owner rank index

        Raises:
            NotImplementedError: 待实现（可用 Tensor 二分查找或 dict 映射）
        """
        raise NotImplementedError("get_owner() 待实现")
