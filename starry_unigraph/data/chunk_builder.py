"""ChunkBuilder: 将原始时序图切分为 ChunkAtomic 池，并分配到计算设备。

时间切分策略：
  - 将全局时间轴均匀（或按事件密度自适应）切成 T 个 time_slice。
  - 每个 time_slice 内，对活跃节点做图聚类，生成 C 个 cluster。
  - 每个 (time_slice, cluster) 对构成一个 ChunkAtomic。

设备分配策略（assign_devices）：
  - 统计每个 ChunkAtomic 的 load_estimate（μ 均值 + σ² 方差）。
  - 贪心轮循（大优先）将 chunk 分配给各 rank，使总负载均衡。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from starry_unigraph.data.chunk_atomic import ChunkAtomic


@dataclass
class ChunkBuildConfig:
    """ChunkBuilder 的配置参数。"""
    num_time_slices: int = 100          # 时间轴切分数 T
    num_clusters: int = 8               # 每个 time_slice 的节点簇数 C
    adaptive_split: bool = False        # 是否按事件密度自适应切分时间轴
    cluster_method: str = "metis"       # 节点聚类算法：'metis' | 'random' | 'degree'
    min_chunk_edges: int = 1            # 过滤空 chunk 的最小边数阈值


class ChunkBuilder:
    """将原始时序图构建为 ChunkAtomic 池。

    Usage::

        builder = ChunkBuilder()
        pool = builder.build(cfg, raw_graph)
        device_map = builder.assign_devices(pool, world_size=4)
    """

    def build(self, cfg: ChunkBuildConfig, raw_graph: Any) -> List[ChunkAtomic]:
        """执行时间切分 + 节点聚类，返回 ChunkAtomic 列表（Pool）。

        Args:
            cfg:       构建配置
            raw_graph: 原始时序图（edge list 或 CSR 结构）

        Returns:
            list[ChunkAtomic]，长度 ≤ T × C（空 chunk 已过滤）

        Implementation notes:
          Step 1. _split_time_axis(raw_graph, cfg) → List[TimeSlice]
                  均匀切分：等分时间戳范围为 cfg.num_time_slices 段
                  自适应切分：按事件密度使每段事件数近似相等

          Step 2. for each time_slice:
                  _cluster_nodes(time_slice, cfg) → List[NodeCluster]
                  'metis'  : 使用 METIS 按连接性聚类，最小化跨界边
                  'degree' : 按节点度数分桶，高度节点优先分配
                  'random' : 随机分配（用于基线测试）

          Step 3. for each (time_slice, cluster):
                  a. 提取 Temporal-CSR（本 cluster master 节点的时序邻居）
                  b. 提取 cross_node_ids / cross_node_home / cross_edge_count
                  c. load_estimate = num_edges (可替换为更精细的 FLOP 估计)
                  d. 构造 ChunkAtomic(chunk_id=(t, c), ...)

          Step 4. 过滤 edge 数 < cfg.min_chunk_edges 的空 chunk
        """
        raise NotImplementedError("ChunkBuilder.build() 待实现")

    def assign_devices(
        self,
        pool: List[ChunkAtomic],
        world_size: int,
    ) -> Dict[int, List[ChunkAtomic]]:
        """将 ChunkAtomic 池按负载均衡分配到各 rank。

        算法（大优先贪心）：
          1. 计算 pool 中所有 chunk 的 load_estimate，得到均值 μ 和方差 σ²。
          2. 按 load_estimate 降序排列 chunk（large-first）。
          3. 维护 rank_loads[r] 记录各 rank 当前总负载，初始化为 0。
          4. 逐 chunk 分配：找 rank_loads 最小的 rank，将 chunk 分给它。
          5. 返回 rank → List[ChunkAtomic] 的映射。

        Args:
            pool:       ChunkBuilder.build() 返回的 chunk 列表
            world_size: GPU rank 数量

        Returns:
            dict[rank_id, list[ChunkAtomic]]，各 rank 的负载近似均衡
        """
        raise NotImplementedError("ChunkBuilder.assign_devices() 待实现")
