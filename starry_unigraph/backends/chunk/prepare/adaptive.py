import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from starry_unigraph.cli.main import build_parser


# ------------------------------------------------------------------ #
#  枚举 & 配置                #
# ------------------------------------------------------------------ #

class ModelFamily(Enum):
    DTDG = "dtdg"
    CTDG = "ctdg"


class LoaderMode(Enum):
    TEMPORAL_DECAY = "temporal_decay"  # 时间优先：按时间窗口 + temporal decay 路由
    FULL = "full"                      # 全图全窗口（含temporal decay路由）
    NEIGHBOR_SAMPLE = "neighbor_sample"  # 邻居采样（Sampling底层模型）
    CLUSTER_CHUNK = "cluster_chunk"    # METIS分区chunk


@dataclass
class ChunkConfig:
    model_family: ModelFamily
    batch_size_base: int
    window_size_base: int              # DTDG: 快照数; CTDG: 时间长度
    adaptive_adjustment: bool = True
    info_loss_threshold: float = 0.2   # 跨chunk边比例上限 (ETC/EPIC参考值)
    available_gpu_mem_gb: float = 8.0
    num_clusters: int = 8
    neg_sample_ratio: int = 1
    neighbor_sample_size: Optional[int] = None  # None=不支持邻居采样
    world_size: int = 1                # GPU 机器数量，用于显存评估
    num_per_partition: int = 1         # 每个 partition 内还需切分的 cluster 数下限


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
    edge_indx: torch.Tensor
    edge_feat: torch.Tensor
    edge_ts: torch.Tensor              # [E_local]
    pos_edges: torch.Tensor            # [2, E_pos] 正样本边（局部id）
    neg_edges: torch.Tensor            # [2, E_neg] 负样本边（局部id）

    def save(self, path: str):
        """持久化单个chunk"""
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: str) -> "ChunkData":
        return ChunkData(**torch.load(path))


@dataclass
class DependenceRoute:
    """
    chunk间的依赖路由
    temporal_route: 时序依赖（哪些历史chunk影响当前chunk，来自记忆缓存）
    gnn_route:      空间依赖（哪些邻居chunk需要聚合）
    """
    chunk_id: int
    temporal_deps: list[int]   
    gnn_deps: list[int]                   


class ChunkStore:
    """chunks列表 + 依赖路由的统一存储"""

    def save(
        self,
        chunks: list[ChunkData],
        routes: list[DependenceRoute],
        save_dir: str,
    ) -> None:
        """
        持久化所有chunk数据和路由表
        chunks_list:      List[ChunkData]
        dependence_route: List[DependenceRoute]
        """
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        for chunk in chunks:
            chunk.save(f"{save_dir}/chunk_{chunk.chunk_id}.pt")
        route_data = [r.__dict__ for r in routes]
        # decay_weights tensor → list for json
        for r in route_data:
            r["decay_weights"] = r["decay_weights"].tolist()
        with open(f"{save_dir}/routes.json", "w") as f:
            json.dump(route_data, f)

    def load(self, save_dir: str) -> tuple[list[ChunkData], list[DependenceRoute]]:
        import os, json
        chunks = []
        for fname in sorted(os.listdir(save_dir)):
            if fname.startswith("chunk_") and fname.endswith(".pt"):
                chunks.append(ChunkData.load(f"{save_dir}/{fname}"))
        with open(f"{save_dir}/routes.json") as f:
            route_data = json.load(f)
        routes = [
            DependenceRoute(
                **{**r, "decay_weights": torch.tensor(r["decay_weights"])}
            )
            for r in route_data
        ]
        return chunks, routes


# ------------------------------------------------------------------ #
#  主生成器                                                              #
# ------------------------------------------------------------------ #

class ChunkGenerator:
    """
    统一的 DTDG / CTDG 数据分块与路由生成器DTDG: Full / NeighborSample / ClusterChunk
    CTDG: 信息损失感知分区（参考 ETC / EPIC）
          - ETC:  基于边密度变化检测时间切割点
          - EPIC: 基于信息熵最大化的分区边界
    """

    def __init__(self, config: ChunkConfig):
        self.config = config
        self.loader_mode: Optional[LoaderMode] = None
        self.clusters: Optional[list[list[int]]] = None
        self.clusters_per_partition: int = max(1, config.num_per_partition)

    # ------------------------------------------------------------------ #
    #  Step 1: 模式选择                                                     #
    # ------------------------------------------------------------------ #

    # def analysis_and_select_loader_mode(
    #     self,
    #     edge_index: torch.Tensor,      # [2, E] 全图边
    #     num_nodes: int,
    #     feature_dim: int,
    # ) -> LoaderMode:
    #     """
    #     DTDG 决策树：
    #       1. 全窗口放得进显存 → FULL
    #       2. 单快照放得进显存 且 topology_first=False → NEIGHBOR_SAMPLE
    #       3. 否则 → CLUSTER_CHUNK（动态计算 num_clusters）

    #     CTDG:
    #       固定 FULL 模式（信息损失感知切片）
    #     """
    #     cfg = self.config

    #     if cfg.model_family == ModelFamily.CTDG:
    #         self.loader_mode = LoaderMode.FULL
    #         return self.loader_mode

    #     E_total = edge_index.shape[1]
    #     avail = cfg.available_gpu_mem_gb

    #     # Step 1: 全窗口显存需求
    #     window_mem = self._estimate_window_memory_gb(
    #         num_nodes, E_total, feature_dim, cfg.window_size_base
    #     )
    #     per_gpu_mem = window_mem / cfg.world_size
    #     full_fits = per_gpu_mem < avail * 0.7

    #     if full_fits:
    #         self.loader_mode = LoaderMode.FULL
    #         return self.loader_mode

    #     # Step 2: 单快照显存需求
    #     snap_mem = self._estimate_window_memory_gb(
    #         num_nodes, E_total // max(cfg.window_size_base, 1), feature_dim, 1
    #     )
    #     snap_fits = (snap_mem / cfg.world_size) < avail * 0.7

    #     # Step 3: 决策
    #     if not cfg.topology_first and snap_fits:
    #         self.loader_mode = LoaderMode.NEIGHBOR_SAMPLE
    #         return self.loader_mode

    #     # CLUSTER_CHUNK: 动态计算 num_clusters
    #     node_per_cluster = num_nodes // max(cfg.num_clusters, 1)
    #     edge_per_cluster = E_total * (node_per_cluster / max(num_nodes, 1)) ** 2
    #     cluster_mem = self._estimate_window_memory_gb(
    #         node_per_cluster, int(edge_per_cluster), feature_dim, 1
    #     )
    #     import math
    #     num_clusters = math.ceil(cluster_mem / (avail * 0.7))
    #     num_clusters = max(num_clusters, cfg.num_clusters)
    #     cfg.num_clusters = num_clusters

    #     self.loader_mode = LoaderMode.CLUSTER_CHUNK
    #     return self.loader_mode

    # ------------------------------------------------------------------ #
    #  Step 2: 时间切片                                     #
    # ------------------------------------------------------------------ #

    def generate_time_slice(
        self,
        edge_index: torch.Tensor,      # [2, E]
        edge_ts: torch.Tensor,         # [E]
        slice_config: dict,) -> list[TimeSlice]:
        """DTDG: 按快照id切分为固定窗口
        CTDG: 信息损失感知切分- 参考ETC：检测边密度突变点作为切割边界
              - 参考EPIC：最大化窗口内信息熵，最小化跨窗口信息损失
        """
        if self.config.model_family == ModelFamily.DTDG:
            return self._slice_by_snapshot(edge_index, edge_ts, slice_config)
        else:
            return self._slice_ctdg_info_aware(edge_index, edge_ts, slice_config)

    def _slice_by_snapshot(
        self, edge_index, edge_ts, cfg
    ) -> list[TimeSlice]:
        snaps = max(1, int(cfg.get("window_size", self.config.window_size_base)))
        num_edges = edge_index.shape[1]
        per_snap = max(1, (num_edges + snaps - 1) // snaps) if num_edges > 0 else 1
        slices = []
        for snap_idx in range(snaps):
            start = snap_idx * per_snap
            end = min(num_edges, start + per_snap)
            if start >= end:
                ei = torch.empty((2, 0), dtype=edge_index.dtype)
                ts = torch.empty((0,), dtype=edge_ts.dtype)
                t_start = t_end = float(edge_ts[-1].item()) if num_edges > 0 else 0.0
            else:
                ei = edge_index[:, start:end]
                ts = edge_ts[start:end]
                t_start = float(ts[0].item())
                t_end = float(ts[-1].item())
            slices.append(TimeSlice(
                slice_id=snap_idx,
                edge_index=ei,
                edge_ts=ts,
                t_start=t_start,
                t_end=t_end,
            ))
        return slices


    def _slice_ctdg_info_aware(
        self, edge_index, edge_ts, cfg
    ) -> list[TimeSlice]:
        from starry_unigraph.backends.chunk.prepare.time_split import time_split
        batch_size = int(cfg.get("batch_size", self.config.batch_size_base))
        window_size = int(cfg.get("window_size", self.config.window_size_base))
        group_index, keep_indices = time_split(
            src=edge_index[0],
            dst=edge_index[1],
            ts=edge_ts,
            batch_size=batch_size,
            window_size=window_size,
        )
        kept_ei = edge_index[:, keep_indices]
        kept_ts = edge_ts[keep_indices]
        kept_groups = group_index[keep_indices]
        slices = []
        for gid in kept_groups.unique(sorted=True):
            mask = kept_groups == gid
            ts_slice = kept_ts[mask]
            slices.append(TimeSlice(
                slice_id=int(gid.item()),
                edge_index=kept_ei[:, mask],
                edge_ts=ts_slice,
                t_start=float(ts_slice[0].item()),
                t_end=float(ts_slice[-1].item()),
            ))
        return slices


    # ------------------------------------------------------------------ #
    #  Step 3: 图分区                                                       #
    # ------------------------------------------------------------------ #

    def generate_graph_clusters(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_clusters: Optional[int] = None,
    ) -> list[list[int]]:
        """
        METIS分区 + 自适应调整
        跨cluster边比例超过info_loss_threshold时自动增加cluster数
        """
        n_clust = num_clusters or self.config.num_clusters
        partition = self._metis_partition(edge_index, num_nodes, n_clust)

        if self.config.adaptive_adjustment:
            cross_ratio = self._cross_cluster_edge_ratio(edge_index, partition)
            if cross_ratio > self.config.info_loss_threshold:
                n_clust = int(n_clust * cross_ratio / self.config.info_loss_threshold)
                partition = self._metis_partition(edge_index, num_nodes, n_clust)

        clusters: list[list[int]] = [[] for _ in range(n_clust)]
        for node, cid in enumerate(partition):
            clusters[cid].append(node)

        self.clusters = clusters
        return clusters

    def generate_graph_partitions(
        self,
        edge_index: torch.Tensor,
        time_slice_id: torch.Tensor,   # [E] 每条边所属的slice_id
    ) -> dict[int, list[list[int]]]:
        """
        CTDG场景：每个时间切片独立分区
        返回: {slice_id: clusters}
        """
        result = {}
        for sid in time_slice_id.unique():
            mask = time_slice_id == sid
            sub_ei = edge_index[:, mask]
            num_nodes = sub_ei.max().item() + 1
            result[sid.item()] = self.generate_graph_clusters(sub_ei, num_nodes)
        return result

    def generate_chunks(
        self,
        slices: list[TimeSlice],
        clusters: Optional[list[list[int]]] = None,
    ) -> list[dict]:
        """
        将时间切片 × 节点cluster展开为chunk列表
        FULL/NEIGHBOR模式：每个slice为一个chunk（cluster_id=-1）
        CLUSTER_CHUNK模式：每个slice × cluster为一个chunk
        """
        _clusters = clusters or self.clusters
        chunks_raw = []
        cid_global = 0
        for sl in slices:
            if self.loader_mode == LoaderMode.CLUSTER_CHUNK and _clusters:
                for cluster_id, nodes in enumerate(_clusters):
                    node_set = torch.tensor(nodes)
                    mask = (torch.isin(sl.edge_index[0], node_set) &
                            torch.isin(sl.edge_index[1], node_set))
                    chunks_raw.append({
                        "chunk_id": cid_global,
                        "slice_id": sl.slice_id,
                        "cluster_id": cluster_id,
                        "edge_index": sl.edge_index[:, mask],
                        "edge_ts": sl.edge_ts[mask],
                        "node_set": node_set,
                    })
                    cid_global += 1
            else:
                chunks_raw.append({
                    "chunk_id": cid_global,
                    "slice_id": sl.slice_id,
                    "cluster_id": -1,
                    "edge_index": sl.edge_index,
                    "edge_ts": sl.edge_ts,
                    "node_set": sl.edge_index.unique(),
                })
                cid_global += 1
        return chunks_raw

    # ------------------------------------------------------------------ #
    #  Step 4: 采样预处理                                                   #
    # ------------------------------------------------------------------ #

    def pre_sample_for_chunk(
        self,
        chunks_raw: list[dict],
        num_nodes: int,
    ) -> list[dict]:
        """
        为每个chunk预处理负样本（跨cluster全局采样，避免cluster内偏置）
        recent采样：固定节点集合（chunk内节点即为recent节点集）
        uniform负采样：从全局节点池采，排除正样本
        """
        pos_set_global = set()
        for c in chunks_raw:
            ei = c["edge_index"]
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                pos_set_global.add((s, d))

        for c in chunks_raw:
            ei = c["edge_index"]
            E = ei.shape[1]
            needed = E * self.config.neg_sample_ratio
            neg_src, neg_dst = [], []
            while len(neg_src) < needed:
                s = torch.randint(0, num_nodes, (needed * 2,))
                d = torch.randint(0, num_nodes, (needed * 2,))
                for si, di in zip(s.tolist(), d.tolist()):
                    if (si, di) not in pos_set_global and si != di:
                        neg_src.append(si)
                        neg_dst.append(di)    if len(neg_src) >= needed:
                        break
            c["neg_edges"] = torch.tensor([neg_src[:needed], neg_dst[:needed]])return chunks_raw

    # ------------------------------------------------------------------ #
    #  Step 5: 路由构建                                                     #
    # ------------------------------------------------------------------ #

    def build_route_for_temporal(
        self,
        chunks_raw: list[dict],
        decay_base: float = 0.9,
    ) -> list[DependenceRoute]:
        """
        构建时序依赖路由
        FULL模式：每个chunk依赖所有历史slice（temporal decay加权）
        CLUSTER_CHUNK模式：每个chunk依赖同cluster的历史chunk
        """
        routes = []
        for c in chunks_raw:
            if self.loader_mode == LoaderMode.CLUSTER_CHUNK:
                # 同cluster的历史chunk
                hist = [
                    h for h in chunks_raw
                    if h["cluster_id"] == c["cluster_id"]
                    and h["slice_id"] < c["slice_id"]
                ]
            else:
                hist = [h for h in chunks_raw if h["slice_id"] < c["slice_id"]]

            hist = sorted(hist, key=lambda x: x["slice_id"], reverse=True)
            steps_back = torch.arange(1, len(hist) + 1, dtype=torch.float)
            decay_weights = decay_base ** steps_back

            routes.append(DependenceRoute(
                chunk_id=c["chunk_id"],
                temporal_deps=[h["chunk_id"] for h in hist],
                gnn_deps=[],# 由build_route_for_gnn填充
                decay_weights=decay_weights,
            ))
        return routes

    def build_route_for_gnn(
        self,
        chunks_raw: list[dict],
        routes: list[DependenceRoute],
    ) -> list[DependenceRoute]:
        """
        构建GNN空间依赖路由
        CLUSTER_CHUNK模式：找出与当前chunk有跨cluster边的相邻chunk
        FULL/NEIGHBOR模式：无跨chunk空间依赖
        """
        if self.loader_mode != LoaderMode.CLUSTER_CHUNK:
            return routes

        chunk_map = {c["chunk_id"]: c for c in chunks_raw}
        route_map = {r.chunk_id: r for r in routes}

        for c in chunks_raw:
            neighbor_chunks = set()
            for other in chunks_raw:
                if other["chunk_id"] == c["chunk_id"]:
                    continue
                if other["slice_id"] != c["slice_id"]:
                    continue
                # 检查是否有跨cluster边（节点集合有交叉的邻居）
                c_nodes = set(c["node_set"].tolist())
                o_nodes = set(other["node_set"].tolist())
                ei = c["edge_index"]
                has_cross = any(
                    s in c_nodes and d in o_nodes
                    for s, d in zip(ei[0].tolist(), ei[1].tolist())
                )
                if has_cross:
                    neighbor_chunks.add(other["chunk_id"])
            route_map[c["chunk_id"]].gnn_deps = list(neighbor_chunks)

        return routes

    # ------------------------------------------------------------------ #
    #  Step 6: 数据准备                                                     #
    # ------------------------------------------------------------------ #

    def prepare_chunks_data(
        self,
        chunks_raw: list[dict],
        node_features: torch.Tensor,   # [V, D]
    ) -> list[ChunkData]:
        """
        按chunk切出局部节点特征，重映射节点id，组装ChunkData
        """
        result = []
        for c in chunks_raw:
            nodes = c["node_set"]
            local_map = {n.item(): i for i, n in enumerate(nodes)}
            ei = c["edge_index"]

            def remap(t):
                return torch.tensor([[local_map[x.item()] for x in t[0]],
                                     [local_map[x.item()] for x in t[1]]])

            local_ei = remap(ei)
            neg_ei = c.get("neg_edges", torch.zeros(2, 0, dtype=torch.long))
            # 负样本中不在本chunk节点集合内的过滤掉
            neg_mask = (torch.isin(neg_ei[0], nodes) & torch.isin(neg_ei[1], nodes))
            neg_local = remap(neg_ei[:, neg_mask]) if neg_mask.any() else torch.zeros(2, 0, dtype=torch.long)

            result.append(ChunkData(
                chunk_id=c["chunk_id"],
                slice_id=c["slice_id"],
                cluster_id=c["cluster_id"],
                global_node_ids=nodes,
                x=node_features[nodes],
                local_edge_index=local_ei,
                edge_ts=c["edge_ts"],
                pos_edges=local_ei,
                neg_edges=neg_local,
            ))
        return result

    # ------------------------------------------------------------------ #
    #  内部工具                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_window_memory_gb(V, E, D, W, dtype_bytes=4) -> float:
        node_mem = V * D * W * dtype_bytes
        edge_mem = E * 2 * W * dtype_bytes
        activation_mem = node_mem * 4
        return (node_mem + edge_mem + activation_mem) / (1024 ** 3)

    @staticmethod
    def _metis_partition(edge_index: torch.Tensor, num_nodes: int, n_parts: int) -> list[int]:
        try:
            import pymetis
            adj = [[] for _ in range(num_nodes)]
            for s, d in edge_index.t().tolist():
                adj[s].append(d)
                adj[d].append(s)
            _, membership = pymetis.part_graph(n_parts, adjacency=adj)
            return membership
        except ImportError:
            return [i * n_parts // num_nodes for i in range(num_nodes)]

    @staticmethod
    def _cross_cluster_edge_ratio(edge_index: torch.Tensor, partition: list[int]) -> float:
        p = torch.tensor(partition)
        cross = (p[edge_index[0]] != p[edge_index[1]]).sum().item()
        return cross / max(edge_index.shape[1], 1)
    
    
    def gener
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(argv