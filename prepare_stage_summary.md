# Chunk Prepare Stage Summary

当前 chunk prepare 阶段的核心链路是：节点 partition -> chunk assignment -> per-slice chunk load matrix -> 按时间片负载向量 rebalance -> hot node replica -> placement/PartitionData。外层分三步：`prepare_raw()` 读原始数据，`build_partitions()` 做分区和 chunk prepare，`build_runtime_artifacts()` 写离线文件。

## 1. prepare_raw

位置：`starry_unigraph/preprocess/chunk.py`

输入来自 `session_ctx`：

```python
session_ctx.config["data"]["root"]
session_ctx.config["data"]["name"]
session_ctx.config["train"]["snaps"] 或 chunk.time_slices
session_ctx.config["data"]["slice_config"]
```

执行逻辑：

```text
1. 找 dataset_root
2. load_raw_temporal_events() 读取原始时序边
3. 根据 snaps / slice_config 调 build_snapshot_dataset_from_events()
4. 要求 raw_dataset["dataset"] 非空
5. 写入 provider_state
```

内存输出：

```python
session_ctx.provider_state["raw_events"]
session_ctx.provider_state["raw_dataset"]
session_ctx.provider_state["raw_stats"] = {
    "num_nodes": ...,
    "num_edges": ...,
    "num_snapshots": ...,
}
```

## 2. build_partitions

位置：`starry_unigraph/preprocess/chunk.py`

输入：

```python
raw_events.src
raw_events.dst
raw_events.ts
raw_events.num_nodes
raw_dataset
dist.world_size
chunk 配置
graph.partition
data.graph_mode / graph_family
```

关键配置：

```python
partition_strategy = chunk.partition_strategy or graph.partition or "metis"
hot_topk
hot_ratio / shared_ratio
num_chunks_per_partition / node_clusters
max_imbalance_ratio
max_migrations
build_mem_routes
num_candidates
```

核心调用：

```python
artifacts = prepare_chunks(...)
```

也就是 `starry_unigraph/backends/chunk/prepare/pipeline.py` 里的 `prepare()`。

`prepare()` 显式接收：

```python
graph_family: str = "ctdg"  # ctdg | dtdg | chunk
```

`prepare()` 主要逻辑：

```text
1. 如果传入已有 assignment，则复用 node_to_chunk/chunk layout
2. 否则根据 edge_src/edge_dst 构建 node_partition
   - metis 或 degree-balanced fallback
   - hot node 根据 hot_topk/hot_ratio 选出
3. build_chunk_assignment()
   - node -> global chunk
   - chunk -> initial partition
   - chunk -> owner partition
4. 计算负载
   - 有 time_ptr 时输出 chunk_load_by_slice[time, chunk]
   - 同时保留聚合 load_stats 作为摘要/fallback
   - 无 time_ptr 时只按全图聚合统计
5. rebalance_chunks()
   - 有 chunk_load_by_slice 时按时间片负载向量贪心分配
   - 否则回退到旧的聚合标量负载 rebalance
   - 只改变 chunk owner，不改变 node -> chunk membership
   - 得到最终 node_owner
6. hot/shared 节点保持原 master
7. 可选构建 mem_routes
8. 可选构建 spatial_routes
```

`prepare()` 返回 `PrepareArtifacts`，主要字段：

```python
assignment              # ChunkAssignment
node_owner              # [num_nodes] 最终 owner partition
load_stats              # chunk 负载统计
rebalance_manifest      # chunk 迁移记录
node_partition          # [num_nodes] 初始/master partition
node_to_partition       # [num_nodes]
hot_node_mask           # [num_nodes] bool
hot_node_ids            # hot node ids
replica_mask            # [num_nodes] bool，目前等价 shared/hot mask
graph_family            # ctdg | dtdg | chunk
chunk_load_by_slice     # Optional[Tensor[num_slices, num_chunks]]
partition_strategy
mem_routes              # Optional[list[list[MemoryRouteData]]]
spatial_routes          # Optional[list[list[SpatialRouteData]]]
time_ptr                # [num_snapshots + 1]
```

`build_partitions()` 写入内存：

```python
session_ctx.provider_state["chunk_prepare"] = artifacts
session_ctx.provider_state["partition_manifest"] = {
    "num_parts": ...,
    "partition_algo": ...,
    "num_nodes": ...,
    "num_edges": ...,
    "num_snapshots": ...,
    "num_chunks": ...,
    "num_chunks_per_partition": ...,
    "hot_node_count": ...,
}
```

## 3. build_runtime_artifacts

位置：`starry_unigraph/preprocess/chunk.py`

输入：

```python
raw_events
raw_dataset
raw_stats
provider_state["chunk_prepare"]
provider_state["partition_manifest"]
artifact_root
```

执行逻辑：

```text
1. 生成 edge_ids = arange(num_edges)
2. 构建 placement.pth 内容
3. 对每个 part_id 构建 PartitionData
4. 写 partitions/part_XXX.pth
5. 写 placement.pth
6. 写 rebalance_manifest.json
7. 写 route bundle 文件
8. 写 meta/manifest JSON
```

每个 partition 的边选择逻辑：

```python
owns_dst = node_owner[edge_dst] == part_id
hot_edge = hot_mask[edge_src] | hot_mask[edge_dst]
mask = owns_dst | hot_edge
```

也就是：

```text
目的点属于该 partition 的边会写入该 partition；
涉及 hot/shared node 的边也会复制进去。
```

## 4. placement.pth 内容

构建位置：`starry_unigraph/preprocess/chunk.py`

现在集中保存分布式索引：

```python
{
    "format": "chunk_dist_index_v1",
    "placement_version": 0,
    "dist_index": {...},

    "assignment": ChunkAssignment,
    "node_to_chunk": Tensor[num_nodes],
    "node_owner": Tensor[num_nodes],
    "node_master": Tensor[num_nodes],
    "node_partition": Tensor[num_nodes],
    "replica_mask": BoolTensor[num_nodes],
    "hot_node_ids": Tensor,
    "graph_family": str,
    "chunk_load_by_slice": Optional[Tensor[num_slices, num_chunks]],

    "canonical_nid_dist": Tensor[num_nodes],
    "canonical_eid_dist": Tensor[num_edges],

    "local_node_ids_by_part": List[Tensor],
    "local_nid_dist_by_part": List[Tensor],
    "local_node_counts": List[dict],

    "canonical_edge_ids_by_part": List[Tensor],
    "time_ptr": Tensor[num_snapshots + 1],
}
```

DistIndex 编码在 `starry_unigraph/backends/chunk/data/dist_index.py`：

```text
低 48 bit: local id
bit 48: shared flag
bit 49: cached/shadow flag
高位: partition id，shift=50
```

本地节点布局：

```text
local_id 0 ... shared_count-1                         shared/hot nodes
local_id shared_count ... shared_count+owned_count-1  owned nodes
后面                                                     1-hop shadow nodes
```

其中：

```python
shared_local -> DistIndex(..., shared=True)
shadow_1hop  -> DistIndex(..., cached=True)
```

## 5. 硬盘输出

当前主要输出：

```text
artifact_root/
  placement.pth
  partitions/
    part_000.pth
    part_001.pth
    ...
    manifest.json
    rebalance_manifest.json
  mem_routes_000.pth              # 如果 build_mem_routes=True
  mem_routes_001.pth
  spatial_routes_000.pth          # 如果有 spatial routes
  meta.json
  routes/manifest.json
  snapshots/manifest.json
  clusters/part_000/cluster_manifest.json
  meta/artifacts.json
```

相比之前，已经减少了这些分散文件的新写入：

```text
chunk_assignment.pth
node_owner.pt
node_partition.pt
hot_node_ids.pt
replica_mask.pt
time_ptr.pt
根目录 part_000.pth
mem_routes_000/slice_*.pth
```

这些信息现在集中进 `placement.pth` 或 route bundle 文件。
