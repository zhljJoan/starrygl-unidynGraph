"""GlobalCSR Protocol: 全图 CSR 结构的访问接口。

GlobalCSR 存储全图（所有分区）的静态图结构（不含时间戳），
主要用于节点聚类（ChunkBuilder）和 DTDG 的全图 materialize。
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class GlobalCSR(Protocol):
    """全图压缩稀疏行（CSR）结构的统一接口。

    Notes:
        - 仅存储拓扑（rowptr + col），特征存储在 FeatureStore
        - 支持按节点子集提取子图 CSR（用于 ChunkAtomic 构建）
        - 大图场景下可分片加载（shard_load），每个 rank 仅加载本分区节点的邻居
    """

    @property
    def num_nodes(self) -> int:
        """全图节点总数。"""
        ...

    @property
    def num_edges(self) -> int:
        """全图边总数。"""
        ...

    def rowptr(self) -> Tensor:
        """返回 CSR row_ptr，shape = [num_nodes + 1]。"""
        ...

    def col(self) -> Tensor:
        """返回 CSR col 索引，shape = [num_edges]。"""
        ...

    def neighbors(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """返回指定节点的所有邻居及对应边 ID。

        Args:
            node_ids: [N] 节点全局 ID

        Returns:
            (neighbor_ids, edge_ids)，各为 1D Tensor，邻居按 node_ids 顺序拼接
        """
        ...

    def subgraph(self, node_ids: Tensor) -> "GlobalCSR":
        """提取 node_ids 诱导子图的 CSR 结构。

        Args:
            node_ids: [N] 节点全局 ID（无需排序）

        Returns:
            新的 GlobalCSR，节点 ID 重映射为 [0, N)
        """
        ...

    @classmethod
    def load(cls, data_root: str, rank: int = 0, world_size: int = 1) -> "GlobalCSR":
        """从磁盘加载 CSR 结构。

        Args:
            data_root:  数据根目录
            rank:       当前 rank（用于分片加载）
            world_size: 总 rank 数

        Returns:
            GlobalCSR 实例
        """
        ...
