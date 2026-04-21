"""Runner Protocol: 双模 Runner 的统一接口。

DTDG（DTDGRunner）和 CTDG（CTDGRunner）各自实现此 Protocol。
PipelineEngine.run() 只依赖 Runner Protocol，不感知 DTDG/CTDG 差异。

Runner 负责：
  1. build_sequence：从 ChunkAtomic Pool 组合出执行序列
  2. prepare_data：采样 + 状态预取
  3. sync_data：等待异步预取完成
  4. compute：model forward + backward
  5. post_compute：状态写回 + 增量推送
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from torch import Tensor

from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.runtime.state.base import StateManager


class BatchData:
    """prepare_data() 返回的批次数据。

    compute() 前由 sync_data() 保证完整（所有异步预取已完成）。
    具体字段由各 Runner 填充：
      DTDG: full_mfg, state_handle
      CTDG: sampled_mfg, state_handle
    """
    pass


@runtime_checkable
class Runner(Protocol):
    """双模 Runner 统一 Protocol。

    PipelineEngine 通过此接口驱动训练循环，不关心 DTDG/CTDG 的具体语义。
    """

    def build_sequence(
        self,
        pool: Sequence[ChunkAtomic],
        device_chunks: Sequence[ChunkAtomic],
    ) -> Sequence[ChunkAtomic]:
        """从 ChunkAtomic Pool 组合出本 rank 的执行序列。

        Args:
            pool:          全局 ChunkAtomic 列表（所有 rank 共享）
            device_chunks: 本 rank 被分配的 ChunkAtomic 子集

        Returns:
            有序的 ChunkAtomic 序列，PipelineEngine 按此顺序执行

        Implementation notes:
            DTDG: 按时间顺序排列快照，同一快照内按 cluster 顺序排列
                  显存限制下随机选取 cluster 子集（不是全部同时加载）
                  示例: [chunk(t=0,c=0), chunk(t=0,c=2), chunk(t=1,c=0), ...]

            CTDG: 按时间顺序排列事件批次，同一时间窗口内的 cluster 可并发
                  短窗口策略：连续几个时间片的 chunk 合并为一个批次
                  示例: [chunk(t=0,c=0), chunk(t=0,c=1), chunk(t=1,c=0), ...]
        """
        ...

    def prepare_data(
        self,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> BatchData:
        """采样 + 预取状态，构建 BatchData（可包含异步句柄）。

        Args:
            chunk:     当前 ChunkAtomic
            state_mgr: StateManager，用于预取节点状态

        Returns:
            BatchData（可能含未完成的异步通信句柄，sync_data 前不完整）

        Implementation notes:
            DTDG: chunk.materialize(full_graph_cfg) → full_mfg + remote_manifest
                  state_mgr.prepare(mfg.all_node_ids, chunk.spatial_deps)
            CTDG: chunk.materialize(sampler_cfg) → sampled_mfg + remote_manifest
                  state_mgr.prepare(mfg.sampled_node_ids, chunk.state_deps)
        """
        ...

    def sync_data(self, data: BatchData) -> None:
        """等待 prepare_data 中的异步预取完成，保证 data 完整可用。

        Args:
            data: prepare_data() 返回的 BatchData

        Notes:
            - 内部调用 state_mgr.patch(data.state_handle)
            - PipelineEngine 在此阶段与前一 chunk 的 compute 重叠
            DTDG: patch 跨分区 embedding（来自 all_to_all）
            CTDG: patch 远程 memory（来自 async_fetch）
        """
        ...

    def compute(self, data: BatchData, model: Any) -> Any:
        """model forward + backward，返回输出（loss / predictions）。

        Args:
            data:  sync_data 后完整的 BatchData
            model: 神经网络模型（不感知 DTDG/CTDG）

        Returns:
            output：包含 loss 和 predictions 的对象

        Notes:
            - 模型只接收标准 tensor（MFG + state），不感知 chunk 语义
            - backward 在此方法内调用（或由调用方在 post_compute 前调用）
        """
        ...

    def post_compute(
        self,
        output: Any,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> None:
        """更新节点状态并异步推送增量。

        Args:
            output:    compute() 的返回值（含更新后的 embedding/memory）
            chunk:     当前 ChunkAtomic
            state_mgr: StateManager

        Notes:
            - 调用 state_mgr.update(node_ids, new_state) 写回并推送
            DTDG: 更新 owned_embedding + 推送跨分区 embedding delta
            CTDG: 更新 owned_memory + 推送 memory delta 给 owner rank
        """
        ...
