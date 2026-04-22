"""ChunkAtomic: 统一底层数据单元，用于 DTDG/CTDG 双模共享 PipelineEngine。

每个 ChunkAtomic 代表一个时间切片 × 节点簇的计算单元。
Python 层仅作薄封装，所有密集计算委托给 C++/CUDA 扩展。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor

from starry_unigraph.runtime.comm.spatial_plan import SpatialDeps  # noqa: F401
from starry_unigraph.runtime.comm.state_plan import StateDeps  # noqa: F401


@dataclass
class RemoteManifest:
    """本地采样阶段产出的远程数据请求清单。

    记录哪些远程节点的 embedding/memory 需要从其他 rank 拉取，
    以便 CommEngine 统一调度。
    """
    remote_node_ids: Tensor     # [N_remote] 需要拉取的远程节点全局 ID
    remote_owners: Tensor       # [N_remote] 对应的 owner rank
    local_slots: Tensor         # [N_remote] 合并后填入 MFG 的槽位索引


@dataclass
class ChunkAtomic:
    """时间切片 × 节点簇的原子计算单元。

    chunk_id = (time_slice_id, node_cluster_id)

    数据分三层：
      L1 — 图结构 (Temporal-CSR)
      L2 — 跨界邻居信息
      L3 — 调度元数据

    Python 层不做任何密集计算；materialize() 调用 C++/CUDA 扩展。
    """

    # --- 标识 ---
    chunk_id: Tuple[int, int]       # (time_slice_id, node_cluster_id)
    time_range: Tuple[float, float] # (t_start, t_end)，浮点时间戳

    # --- L1: 本地图结构 (Temporal-CSR) ---
    node_set: Tensor        # [N_local]  本 cluster 的 master 节点全局 ID
    tcsr_rowptr: Tensor     # [N_local+1] Temporal-CSR row pointers
    tcsr_col: Tensor        # [E]        邻居列索引
    tcsr_ts: Tensor         # [E]        对应边时间戳
    tcsr_edge_id: Tensor    # [E]        全局边 ID

    # --- L2: 跨界邻居 ---
    cross_node_ids: Tensor    # [N_cross]   跨界邻居节点全局 ID (sorted)
    cross_node_home: Tensor   # [N_cross]   对应的 home cluster_id
    cross_edge_count: Tensor  # [N_cluster] 到每个 cluster 的跨界边数

    # --- L3: 调度元数据 ---
    load_estimate: float = 0.0          # 计算负载估计（用于设备分配均衡）
    spatial_deps: Optional[SpatialDeps] = None  # 同时段 cluster 间通信依赖
    state_deps: Optional[StateDeps] = None      # 跨时间 state 读写依赖

    # --- 运行时钩子（由 Runner 注册，不参与序列化）---
    _neg_hook: Optional[Callable] = field(default=None, repr=False, compare=False)
    _neighbor_hook: Optional[Callable] = field(default=None, repr=False, compare=False)

    def register_negative_hook(self, functional: Callable) -> None:
        """注册负采样函数，供 C++ 扩展回调。"""
        self._neg_hook = functional

    def register_neighbor_hook(self, functional: Callable) -> None:
        """注册邻居采样函数，供 C++ 扩展回调。"""
        self._neighbor_hook = functional

    def materialize(self, sampler_cfg: Any) -> Tuple[Any, RemoteManifest]:
        """Phase 1: 本地时序邻居采样 + 产出远程请求清单。

        实现委托给 C++/CUDA 扩展：
          1. 在 Temporal-CSR 上对每个事件 (src, dst, t) 做二分查找，
             采样 t 之前最近的 k 个邻居。
          2. locality_mask 标记每个采样邻居是本地还是远程。
          3. 构建 local_mfg（本地可直接聚合的子图）。
          4. 提取 RemoteManifest（需要跨 rank 拉取的远程节点清单）。

        Args:
            sampler_cfg: 采样配置，含 num_neighbors / num_layers / sample_type。

        Returns:
            (local_mfg, remote_manifest)
              local_mfg      — 本地 MFG，可直接用于前向传播（远程槽位待填充）
              remote_manifest — 需要 CommEngine 异步拉取的远程数据描述
        """
        raise NotImplementedError(
            "materialize() 依赖 C++/CUDA 扩展 (_C)，需在完整运行时中实现。"
        )

    def complete(self, local_mfg: Any, remote_data: Any) -> Any:
        """Phase 2: 将远程拉取的 embedding 合并入 MFG，生成最终训练 batch。

        Args:
            local_mfg:    materialize() 返回的本地 MFG
            remote_data:  CommEngine 拉取并返回的远程 embedding tensor

        Returns:
            完整的训练 batch（remote 槽位已填充）
        """
        raise NotImplementedError(
            "complete() 依赖 C++/CUDA 扩展 (_C.merge_remote_into_mfg)。"
        )
