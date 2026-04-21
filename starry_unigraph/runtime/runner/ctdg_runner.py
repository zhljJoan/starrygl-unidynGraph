"""CTDGRunner: CTDG 事件流重组策略实现。

CTDG 特点：
  - 短窗口时序依赖，事件驱动的 memory 更新
  - 显存压力相对小（仅需 batch 内事件的邻居状态）
  - CommPlan 半动态：采样随机性导致每 batch 通信模式不同

重组策略：
  每个批次由同一时间窗口内的事件构成，可跨多个 cluster。
  同一时间内不同 cluster 的 chunk 无时序依赖，可并发或直接合并。

多设备：各 rank 持有不同 cluster，数据并行。
"""
from __future__ import annotations

from typing import Any, List, Sequence

from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.runtime.runner.base import BatchData, Runner
from starry_unigraph.runtime.state.base import StateManager


class CTDGRunner:
    """CTDG 事件流重组 Runner。

    实现 Runner Protocol，面向连续时间动态图（Continuous-Time Dynamic Graph）。

    核心策略：
      build_sequence: 按时间顺序排列 chunk，同一时间窗口内 cluster 可并发
      prepare_data:   时序邻居采样（BTS）+ 预取远程 memory
      sync_data:      等待 async_fetch 完成，patch 远程 memory
      compute:        model(sampled_mfg, memory) → TGN / MemShare
      post_compute:   写回 memory + 推送 memory delta
    """

    def build_sequence(
        self,
        pool: Sequence[ChunkAtomic],
        device_chunks: Sequence[ChunkAtomic],
    ) -> List[ChunkAtomic]:
        """按时间顺序组合 CTDG 执行序列。

        Strategy:
          1. 按 chunk_id[0]（time_slice_id）对 device_chunks 排序
          2. 同一 time_slice 内的多个 cluster chunk 可视为独立批次
             （无时序依赖，可合并为单一更大批次，减少调度开销）
          3. 跨 time_slice 顺序执行（CTDG 时序依赖）

        Args:
            pool:          全局 ChunkAtomic 池
            device_chunks: 本 rank 分配的 ChunkAtomic 子集

        Returns:
            按 time_slice_id 排序的 chunk 列表（同时间步内可合并）

        Example output:
            [chunk(t=0,c=0),   # batch 0
             chunk(t=0,c=1),   # batch 1（与 batch 0 并发，可合并为单批）
             chunk(t=1,c=0),   # batch 2（依赖 batch 0，时序）
             ...]
        """
        raise NotImplementedError("CTDGRunner.build_sequence() 待实现")

    def prepare_data(
        self,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> BatchData:
        """时序邻居采样 + 负采样 + 预取远程 memory。

        Steps:
          1. chunk.materialize(sampler_cfg) → sampled_mfg, remote_manifest
             BTS 采样：在 Temporal-CSR 上对每个事件采样 t 之前最近的 k 个邻居
          2. state_mgr.prepare(sampled_mfg.all_node_ids, chunk.state_deps)
             三层查找（owned/hot/decay），miss 节点异步拉取
          3. 打包 BatchData(mfg=sampled_mfg, state_handle=handle,
                             remote_manifest=remote_manifest)
        """
        raise NotImplementedError("CTDGRunner.prepare_data() 待实现")

    def sync_data(self, data: BatchData) -> None:
        """等待 async_fetch 完成，将远程 memory 写入 MFG。

        Steps:
          1. state_mgr.patch(data.state_handle) → 填充 miss 节点的 memory
          2. chunk.complete(data.mfg, fetched_remote_data) → 完整 MFG
        """
        raise NotImplementedError("CTDGRunner.sync_data() 待实现")

    def compute(self, data: BatchData, model: Any) -> Any:
        """TGN / MemShare forward + backward。

        Steps:
          1. 从 data.state_handle 获取所有节点 memory
          2. model.forward(data.mfg, memory) → new_memory, loss, predictions
          3. loss.backward()
          4. 返回 output(new_memory=..., loss=..., node_ids=..., predictions=...)

        Notes:
            model 只接收 MFG + memory tensor，不感知 CTDG/DTDG。
        """
        raise NotImplementedError("CTDGRunner.compute() 待实现")

    def post_compute(
        self,
        output: Any,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> None:
        """写回 memory，推送 memory delta，更新 mailbox。

        Steps:
          1. state_mgr.update(output.node_ids, output.new_memory)
             写回 owned_memory + hot_cache/decay_cache + async_push_delta
          2. state_mgr.step() 更新 decay_cache 优先级，消费 mailbox
        """
        raise NotImplementedError("CTDGRunner.post_compute() 待实现")
