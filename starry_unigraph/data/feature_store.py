"""FeatureStore Protocol: 节点/边特征的分片存储接口。

每个 rank 仅持有本分区节点的特征（owned features）；
跨分区特征通过 CommEngine 按需拉取。
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class FeatureStore(Protocol):
    """节点/边特征分片存储的统一接口。

    DTDG 和 CTDG 使用相同的 FeatureStore，特征按节点分区存储。

    Notes:
        - 本地读取 (get_node_feat) 从本 rank 内存/显存直接返回
        - 远程读取由 CommEngine 驱动，FeatureStore 不感知通信细节
        - 支持 pin_memory + 异步 H2D，与 PipelineEngine 的 cache_stream 配合
    """

    def get_node_feat(self, node_ids: Tensor, feat_name: str = "x") -> Tensor:
        """获取本地节点的特征向量。

        Args:
            node_ids:  [N] 本分区内的节点全局 ID
            feat_name: 特征名称，默认 'x'

        Returns:
            [N, d_feat] 特征 Tensor

        Raises:
            KeyError: feat_name 不存在
            IndexError: node_ids 含有不属于本分区的节点
        """
        ...

    def get_edge_feat(self, edge_ids: Tensor, feat_name: str = "edge_attr") -> Tensor:
        """获取边特征向量。

        Args:
            edge_ids:  [E] 边全局 ID
            feat_name: 特征名称，默认 'edge_attr'

        Returns:
            [E, d_edge] 特征 Tensor
        """
        ...

    def num_nodes(self) -> int:
        """本分区的节点总数（owned nodes）。"""
        ...

    def feat_dim(self, feat_name: str = "x") -> int:
        """返回指定特征的维度 d_feat。"""
        ...

    def pin_memory(self) -> "FeatureStore":
        """将特征数据 pin 到页锁定内存，加速异步 H2D 传输。

        Returns:
            self（支持链式调用）
        """
        ...
