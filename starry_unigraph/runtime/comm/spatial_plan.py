"""SpatialDeps: 空间通信依赖描述。

记录一个 ChunkAtomic 的 GNN 计算所需的跨 rank 节点 embedding 通信计划。
DTDG 侧 CommPlan 完全静态（快照拓扑固定），CTDG 侧按 miss 动态构建。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from torch import Tensor


@dataclass
class SpatialDeps:
    """chunk 声明：我的 GNN 计算需要这些远程节点的 embedding。

    用于 CommEngine.async_exchange()，驱动 all-to-all 或 isend/irecv 通信。

    Fields:
        send_rank_list:  需要向哪些 rank 发送本地 embedding，长度 = num_peers
        recv_rank_list:  需要从哪些 rank 接收远程 embedding，长度 = num_peers
        node_id_lists:   send_rank_list[i] 对应需要发送的节点全局 ID 列表
                         长度 = len(send_rank_list)，每项为 1D Tensor

    Notes:
        - DTDG: send/recv 列表在预处理阶段静态编译，每个快照复用同一 plan
        - CTDG: 由 materialize() 的 locality_mask 动态生成，每 batch 不同
        - CommEngine 根据此结构调度 async_exchange()，不感知 DTDG/CTDG 差异
    """
    send_rank_list: List[int]           # 需要发送的目标 rank 列表
    recv_rank_list: List[int]           # 需要接收的来源 rank 列表
    node_id_lists: List[Tensor]         # 每个目标 rank 对应的节点 ID
    send_sizes: Optional[List[int]] = field(default=None)  # 各 rank 发送数量（可选，用于静态 plan）
    recv_sizes: Optional[List[int]] = field(default=None)  # 各 rank 接收数量（可选）
