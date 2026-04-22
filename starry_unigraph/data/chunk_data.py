from dataclasses import field
from typing import Tuple

import dgl
import torch
from torch import Optional, Tensor

from starry_unigraph.runtime.route.comm import SpatialDeps, StateDeps


class ChunkAtomic:
    """Python层仅作薄封装，所有密集计算在C++/CUDA"""
    
    chunk_id: Tuple[int,int] #(time_slice_id, node_cluster_id)
    time_range: Tuple[int,int] #(t_start, t_end)
    
    #是否保留，存本地特征，chunk特征，还是普通global的特征
    node_set: torch.Tensor # 本地master节点ID列表
    tcsr_rowptr: torch.Tensor # Temporal-CSR的row_ptr
    tcsr_col: torch.Tensor # Temporal-CSR的col_idx
    tcsr_ts: torch.Tensor # Temporal-CSR的timestamps
    tcsr_edge_id: torch.Tensor # Temporal-CSR的edge_id
    
    #boundary info
    cross_node_ids: Tensor    # [N_cross]   跨界邻居节点ID (sorted)
    cross_node_home: Tensor   # [N_cross]   对应的home cluster_id
    cross_edge_count: Tensor  # [N_cluster] 到每个cluster的跨界边数
    
    # === L3: Scheduling Meta ===
    load_estimate: float               # 计算负载估计
    spatial_deps: SpatialDeps    # 固定快照的通信依赖关系 
    state_deps: StateDeps        # 跨时间的state依赖关系
    
    def load_from_chunk(block) -> 'ChunkAtomic':
        src_ids: list[torch.Tensor] = []
        dst_ids: list[torch.Tensor] = []
        edge_ids: list[torch.Tensor] = []
        edge_src: list[torch.Tensor] = []
        edge_dst: list[torch.Tensor] = []
        src_node_data: dict[str, list[torch.Tensor]] = {}
        dst_node_data: dict[str, list[torch.Tensor]] = {}
        edge_data: dict[str, list[torch.Tensor]] = {}
        routes: list = []
        src, dst = block.edges()
        edge_src.append(src)
        edge_dst.append(dst)
        src_id = block.srcdata[dgl.NID] if dgl.NID in block.srcdata else torch.arange(block.num_src_nodes())
        dst_id = block.dstdata[dgl.NID] if dgl.NID in block.dstdata else torch.arange(block.num_dst_nodes())
        edge_id = block.edata[dgl.EID] if dgl.EID in block.edata else torch.arange(block.num_edges())
            if torch.any(src_id[: dst_id.numel()] != dst_id):
                raise ValueError("prefix of src_ids and dst_ids must be the same")
            src_ids.append(src_id[dst_id.numel() :])
            dst_ids.append(dst_id)
            edge_ids.append(edge_id)
            for key in block.srcdata.keys():
                if key != dgl.NID:
                    src_node_data.setdefault(key, []).append(block.srcdata[key])
            for key in block.dstdata.keys():
                if key != dgl.NID:
                    dst_node_data.setdefault(key, []).append(block.dstdata[key])
            for key in block.edata.keys():
                if key != dgl.EID:
                    edge_data.setdefault(key, []).append(block.edata[key])
            route = getattr(block, "route", None)
            if route is not None:
                routes.append(route)
        merged_node_data = {k: TensorData.from_tensors(v) for k, v in dst_node_data.items() if len(v) == len(blocks)}
        for key, values in src_node_data.items():
            if len(values) == len(blocks):
                merged_node_data[key] = TensorData.from_tensors(values)
        return cls(
            src_ids=TensorData.from_tensors(src_ids),
            dst_ids=TensorData.from_tensors(dst_ids),
            edge_ids=TensorData.from_tensors(edge_ids),
            edge_src=TensorData.from_tensors(edge_src),
            edge_dst=TensorData.from_tensors(edge_dst),
            node_data=merged_node_data,
            edge_data={k: TensorData.from_tensors(v) for k, v in edge_data.items() if len(v) == len(blocks)},
            routes=RouteData.from_routes(routes) if len(routes) == len(blocks) else None,
        )

    def register_negative_hook(self, functional):
        """注册负采样函数，供C++调用"""
        self._neg_hook = functional
    
    def register_neighbor_hook(self, functional):
        """注册邻居采样函数，供C++调用"""
        self._neighbor_hook = functional
        
    #本地采样 + 重建远程请求清单
    def materialize(self, sampler_cfg):
        """Phase 1: 本地采样 + 产出远程请求清单"""
        # ---- 全部走C++扩展，Python零逻辑 ----
        # 1. 时序邻居采样: 在Temporal-CSR上做二分查找
        #    对每个event(src, dst, t), 找t之前最近的k个邻居
        #    返回: sampled_ids, sampled_ts, locality_mask(标记本地/远程)
        if self._neg_hook is not None:
            neg = self._neg_hook(self.neg_pool, len(self.events_src))
        if self._neighbor_hook is not None:
            sampled, locality_mask = self._neighbor_hook(
                self.row_ptr, self.col_idx, self.timestamps,
                self.events_src, self.events_dst, neg, self.events_ts,
                sampler_cfg.num_neighbors, sampler_cfg.num_layers, sampler_cfg.sample_type
            )
        local_mfg = s_C.build_mfg(sampled, locality_mask)
        remote_manifest = _C.extract_remote_manifest(
            sampled, locality_mask, self.dep_manifest
        )
        return local_mfg, remote_manifest  # Python只传递，不处理
    
    def complete(self, local_mfg, remote_data):
        """Phase 2: 合并远程数据，生成最终训练batch"""
        return _C.merge_remote_into_mfg(local_mfg, remote_data)