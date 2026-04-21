import torch
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from starry_unigraph.cli.main import build_parser


class ModelFamily(Enum):
    DTDG = "dtdg"
    CTDG = "ctdg"




@dataclass
class ChunkConfig:
    model_family: ModelFamily
    batch_size_base: int
    window_size_base: int              # DTDG: 快照数; CTDG: 时间长度
    adaptive_adjustment: bool = True
    info_loss_threshold: float = 0.2   
    available_gpu_mem_gb: float = 8.0
    num_partitions: int = 4
    num_clusters: int = 64
    num_clusters_per_partition: int = 16
    neg_sample_ratio: int = 1
    neighbor_sample_size: Optional[List[int]] = None  # None=不支持邻居采样采样
    num_per_cluster: Optional[List[int]]= None         # 如果为None，则切分所有
    shared_ratio = 0.1                                 # 共享节点比例


# ------------------------------------------------------------------ #
#  数据结构                                                              #
# ------------------------------------------------------------------ #

@dataclass
class TimeSlice:
    """单个时间切片"""
    slice_id: int
    edge_index: torch.Tensor           # [2, E]
    edge_ts: torch.Tensor              # [E]
    t_start: float
    t_end: float


@dataclass
class ChunkData:
    """单个chunk的完整数据，可直接送入模型"""
    chunk_id: int
    slice_id: int
    cluster_id: int                    
    master_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_ts: torch.Tensor              # [E_local]

    neighbors: List[torch.Tensor] | None = None       # [2, E_neighbors] 邻居边（局部id）

    def save(self, path: str):
        """持久化单个chunk"""
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: str) -> "ChunkData":
        return ChunkData(**torch.load(path))
    
    
class ChunkAtomic:
    """Python层仅作薄封装，所有密集计算在C++/CUDA"""
    
    def materialize(self, sampler_cfg):
        """Phase 1: 本地采样 + 产出远程请求清单"""
        # ---- 全部走C++扩展，Python零逻辑 ----
        # 1. 时序邻居采样: 在Temporal-CSR上做二分查找
        #    对每个event(src, dst, t), 找t之前最近的k个邻居
        #    返回: sampled_ids, sampled_ts, locality_mask(标记本地/远程)
        sampled, locality_mask = _C.temporal_neighbor_sample(
            self.row_ptr, self.col_idx, self.timestamps,
            self.events_src, self.events_dst, self.events_ts,
            sampler_cfg.num_neighbors, sampler_cfg.num_layers
        )
        
        # 2. 负采样: 从预过滤池中随机抽取
        neg = self.neg_pool[torch.randint(0, len(self.neg_pool),
                                          (len(self.events_src),))]
        
        # 3. 去重 + 构建本地MFG (Message Flow Graph)
        #    内部完成: 节点去重、边去重、local subgraph construction
        local_mfg = _C.build_mfg(sampled, locality_mask, neg)
        
        # 4. 从locality_mask中提取远程请求清单
        #    格式: {remote_chunk_id: [needed_node_ids]}
        remote_manifest = _C.extract_remote_manifest(
            sampled, locality_mask, self.dep_manifest
        )
        
        return local_mfg, remote_manifest  # Python只传递，不处理
    
    def complete(self, local_mfg, remote_data):
        """Phase 2: 合并远程数据，生成最终训练batch"""
        return _C.merge_remote_into_mfg(local_mfg, remote_data)