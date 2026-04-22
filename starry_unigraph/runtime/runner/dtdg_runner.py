"""DTDGRunner: DTDG 快照重组策略实现。

DTDG 特点：
  - 长窗口时序依赖，快照间 embedding 传递
  - 显存压力大：不能同时加载所有 cluster
  - CommPlan 静态：快照拓扑固定，send/recv 列表预编译

重组策略：
  每个时间步从 pool 中选取同一时间片的多个 cluster chunk（随机子集）。
  跨时间步顺序执行（时序依赖，不可并发）。

多设备：各 rank 持有不同 cluster 子集，数据并行。
"""
from __future__ import annotations

from typing import Any, List, Sequence

from starry_unigraph.data.chunk_atomic import ChunkAtomic
from starry_unigraph.runtime.runner.base import BatchData, Runner
from starry_unigraph.runtime.state.base import StateManager


class DTDGRunner:
    """DTDG 快照重组 Runner。

    实现 Runner Protocol，面向离散快照图（Discrete-Time Dynamic Graph）。

    核心策略：
      build_sequence: 按时间顺序排列 chunk，同一快照内随机选子集
      prepare_data:   全子图构建 MFG（无采样）+ 预取跨分区 embedding
      sync_data:      等待 all_to_all 完成，patch 邻居分区 embedding
      compute:        model(full_mfg, embedding) → GNN + LSTM
      post_compute:   写回 embedding + 推送跨分区 embedding delta
    """

    def build_sequence(
        self,
        pool: Sequence[ChunkAtomic],
        device_chunks: Sequence[ChunkAtomic],
    ) -> List[ChunkAtomic]:
        """按时间顺序组合 DTDG 执行序列。

        Strategy:
          1. 按 chunk_id[0]（time_slice_id）对 device_chunks 分组
          2. 对每个时间步，按 chunk_id[1]（cluster_id）顺序排列
          3. （可选）按显存限制对同一时间步的 chunk 数量剪裁

        Args:
            pool:          全局 ChunkAtomic 池（此实现主要使用 device_chunks）
            device_chunks: 本 rank 分配的 ChunkAtomic 子集

        Returns:
            按 (time_slice_id, cluster_id) 排序的 chunk 列表

        Example output:
            [chunk(t=0,c=0), chunk(t=0,c=2),
             chunk(t=1,c=0), chunk(t=1,c=2), chunk(t=1,c=5), ...]
        """
        raise NotImplementedError("DTDGRunner.build_sequence() 待实现")

    def prepare_data(
        self,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> BatchData:
        """全子图构建 MFG + 预取跨分区 embedding。

        Steps:
          1. chunk.materialize(full_graph_cfg) → full_mfg, remote_manifest
             DTDG 无需采样，所有节点参与 GNN
          2. state_mgr.prepare(full_mfg.all_node_ids, chunk.spatial_deps)
             预取邻居分区的 embedding（静态 CommPlan）
          3. 打包 BatchData(mfg=full_mfg, state_handle=handle)
        """
        raise NotImplementedError("DTDGRunner.prepare_data() 待实现")

    def sync_data(self, data: BatchData) -> None:
        """等待 all_to_all 完成，将邻居分区 embedding 写入 MFG。

        Steps:
          1. state_mgr.patch(data.state_handle)
          2. 将 state_handle.gathered_states 填入 data.mfg 的跨分区节点槽位
        """
        raise NotImplementedError("DTDGRunner.sync_data() 待实现")

    def compute(self, data: BatchData, model: Any) -> Any:
        """GNN + RNN forward + backward。

        Steps:
          1. 从 data.state_handle 获取完整 embedding（owned + neighbor）
          2. model.forward(data.mfg, embedding) → new_embedding, loss
          3. loss.backward()
          4. 返回 output(new_embedding=..., loss=..., node_ids=...)

        Notes:
            model 不感知 DTDG/CTDG，只接收标准 MFG + embedding tensor。
        """
        raise NotImplementedError("DTDGRunner.compute() 待实现")

    def post_compute(
        self,
        output: Any,
        chunk: ChunkAtomic,
        state_mgr: StateManager,
    ) -> None:
        """写回 embedding，推送跨分区 embedding delta。

        Steps:
          1. state_mgr.update(output.node_ids, output.new_embedding)
             写回 owned_embedding + 发起 async_push_delta
          2. state_mgr.step() 通知缓存层切换
        """
        raise NotImplementedError("DTDGRunner.post_compute() 待实现")
