# Chunk Backend 修复状态报告

基于其他 agent 的 code review，以下是修复状态：

---

## ✅ 已完成修复（P1 级别）

### 1. Memory timestamp reduce ✅

**问题**：`comm.py:submit_memory()` 不做按 timestamp reduce

**修复**：
- 文件: `starry_unigraph/backends/chunk/data/comm.py:464-502`
- 改动: 在 `await_memory()` 中添加 timestamp reduce 逻辑
- 实现: 使用 `torch.unique()` 去重，保留最新 timestamp 的更新

**代码**：
```python
# Timestamp reduce: keep only the latest update per node
unique_ids, inverse = torch.unique(ids, return_inverse=True)
if unique_ids.numel() < ids.numel():
    # Multiple updates for same nodes - reduce by timestamp
    for i in range(ids.numel()):
        uid = int(inverse[i].item())
        if ts[i] > latest_ts[uid]:
            latest_ts[uid] = ts[i]
            latest_mem[uid] = mem[i]
```

---

### 2. 负采样 dst pool ✅

**问题**：使用 owner pool 而非 dst pool

**修复**：
- 文件: `starry_unigraph/backends/chunk/runtime/loader.py:266-283`
- 改动: 从 `canonical_events.dst` 提取 dst pool
- Fallback: 如果 canonical_events 不存在，使用 owner pool（兼容性）

**代码**：
```python
if canonical_events is not None:
    all_dst = canonical_events.dst.unique(sorted=True)
    local_dst_mask = placement.node_owner[all_dst] == int(rank)
    local_dst_pool = all_dst[local_dst_mask]
    global_dst_pool = all_dst
else:
    # Fallback: use owner pool (compatibility)
    local_dst_pool = torch.nonzero(placement.node_owner == int(rank), as_tuple=False).flatten().long()
    global_dst_pool = torch.arange(int(num_nodes), dtype=torch.long)
```

---

### 3. CTDG 自适应切分配置 ✅

**问题**：已实现但未接入主链路

**修复**：
- 文件: `starry_unigraph/preprocess/chunk.py:616-632`
- 改动: 添加 `time_slice_strategy` 配置项
- 支持: `"adaptive"` (C++ native) 或 `"snapshot"` (legacy)

**配置示例**：
```yaml
chunk:
  time_slice_strategy: "adaptive"  # "adaptive" | "snapshot"
  target_batch_size: 200
  graph_feature: 1.0
  alpha: 1.0
  beta: 0.5
  aggl: 0.0
  use_cpp_adaptive_split: true
```

---

### 4. DTDG ChunkPropagationRoute snapshot 语义 ✅

**问题**：缺少 snapshot index 的 compact row indexing

**修复**：
- 文件: `starry_unigraph/backends/chunk/prepare/propagation_builder.py:18-180`
- 新增函数: `build_propagation_routes_from_snapshots()`
- 支持: `dst_ids/src_ids` 和 `recv_src_rows` 生成

**关键改进**：
```python
def build_propagation_routes_from_snapshots(
    part_data,  # PartitionData with snapshot structure
    node_owner: Tensor,
    num_parts: int,
    num_layers: int = 1,
) -> List[List[List[ChunkPropagationRoute]]]:
    """Build routes using DTDG snapshot compact row indexing.
    
    - Uses dst_ids and src_ids from each snapshot
    - Generates recv_src_rows for compact row indexing
    - Supports DTDG block's [dst, remote_src] row layout
    """
    # ...
    route = ChunkPropagationRoute(
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_index=send_index,
        recv_src_rows=recv_src_rows,  # ✅ New field
        append_recv=True,
    )
```

**使用方式**：
- 新代码应使用 `build_propagation_routes_from_snapshots()`
- 旧代码仍可使用 `build_propagation_routes()`（legacy fallback）

---

## P2 级别（可延后）

### 5. CTDG fetch route 缓存 ⚠️

**状态**：未实现

**建议**：
- 在 `loader.py` 中实现 `_make_dynamic_fetch_plan_cached()`
- Cache key: `(placement_version, block_id, remote_read_index_hash)`

---

### 6. Artifact layout 规范化 ⚠️

**状态**：未实现

**建议**：
- 输出独立的 `chunks/chunk_load_by_slice.pth`
- 输出 `events/dst_pool_*.pth`
- 输出 `indices/snapshot_index_*.pth`

---

### 7. 模型目录迁移 ⚠️

**状态**：未实现

**建议**：
- 创建 `models/encoders/` 和 `models/heads/`
- 迁移 `task_head.py` 中的 heads

---

## 设计合理（无需修改）

### 8. MemoryRouteData 全局去重 ✅

**Review 意见**：Phase 1 全局去重可能导致重复写回

**实际情况**：
- Phase 1 全局去重是为了**复用**（重分区时不需要重建）
- Phase 2 按 `node_owner` 排序并分配
- 每个 rank 只发送它 **own** 的节点更新

**结论**：设计正确，无需修改

---

## 修复总结

| 优先级 | 问题 | 状态 | 文件 |
|--------|------|------|------|
| P1 | Memory timestamp reduce | ✅ 已修复 | `comm.py` |
| P1 | 负采样 dst pool | ✅ 已修复 | `loader.py` |
| P1 | CTDG 自适应切分配置 | ✅ 已修复 | `chunk.py` |
| P1 | DTDG propagation route | ✅ 已修复 | `propagation_builder.py` |
| P2 | CTDG fetch route 缓存 | ⚠️ 可延后 | `loader.py` |
| P2 | Artifact layout | ⚠️ 可延后 | `chunk.py` |
| P2 | 模型目录迁移 | ⚠️ 可延后 | `models/` |
| N/A | MemoryRouteData 全局去重 | ✅ 设计合理 | 无需修改 |

---

## 测试建议

### 1. Memory timestamp reduce
```python
# 测试场景：同一节点多个更新
ids = torch.tensor([1, 2, 1, 3])
ts = torch.tensor([1.0, 2.0, 3.0, 4.0])
mem = torch.randn(4, 128)

# 预期：node 1 保留 ts=3.0 的更新
```

### 2. 负采样 dst pool
```python
# 测试场景：验证 pool 是 dst 而非 owner
assert set(local_dst_pool.tolist()).issubset(set(canonical_events.dst.tolist()))
```

### 3. CTDG 自适应切分
```python
# 配置文件
chunk:
  time_slice_strategy: "adaptive"
  target_batch_size: 200

# 验证：time_ptr 应该基于 staleness 而非均匀切分
```

### 4. DTDG propagation route
```python
# 测试场景：验证 recv_src_rows 对齐 compact layout
route = build_propagation_routes_from_snapshots(part_data, node_owner, num_parts)
assert route[0][0][0].recv_src_rows is not None
```

---

## 下一步

1. ✅ **所有 P1 修复已完成**
2. ⚠️ **P2 优化可根据需求延后**
3. 📝 **建议添加单元测试验证修复**
4. 📚 **更新用户文档说明新配置项**
