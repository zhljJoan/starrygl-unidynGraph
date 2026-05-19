"""Chunk Decay Endpoints: CSR vs ind2ptr 对比分析

## 问题背景

STGraphLoader 需要根据 chunk decay 策略计算每个历史层的 truncation endpoint。
例如：chunk_decay = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]（最近的 chunk 优先）
- Layer 0 (最旧): 只保留 chunk 9 的节点
- Layer 1: 保留 chunk 9 + 8 的节点
- Layer 2: 保留 chunk 9 + 8 + 7 的节点
- ...
- Layer 9 (最新): 保留所有 chunks

需要计算每层的 endpoint（节点总数）。

---

## FlareDTDG 原始方法：ind2ptr

### 数据结构
```python
node_to_chunk: Tensor[num_nodes]  # 每个节点属于哪个 chunk
chunk_index: int                   # 当前 rank 的 chunk ID
```

### 算法流程
```python
# 1. 按 chunk 排序节点
chunk_order = node_to_chunk[perm]  # [num_nodes]，每个节点的 chunk ID
inds, perm = chunk_order.sort()     # inds: sorted chunk IDs, perm: 排序后的节点索引

# 2. 使用 torch_sparse.ind2ptr 计算 chunk boundaries
ends = torch.ops.torch_sparse.ind2ptr(inds, num_chunks)
# ends[c] = 前 c 个 chunks 的节点总数

# 3. 根据 chunk_decay 提取 endpoints
ends_list = [ends[chunk_decay[k]] for k in range(num_decay)]
```

### 示例
```python
# 假设有 10 个节点，3 个 chunks
node_to_chunk = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # 每个节点的 chunk ID
perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]           # 假设已按 chunk 排序

# 排序后
inds = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # sorted chunk IDs

# ind2ptr 计算
ends = torch_sparse.ind2ptr(inds, num_chunks=3)
# ends = [0, 2, 5, 10]
#        ^  ^  ^   ^
#        |  |  |   chunk 2 结束位置（10 个节点）
#        |  |  chunk 1 结束位置（5 个节点）
#        |  chunk 0 结束位置（2 个节点）
#        起始位置

# 如果 chunk_decay = [2, 1, 0]（最近的优先）
ends_list = [ends[2], ends[1], ends[0]]  # [10, 5, 2]
```

### 问题
1. **依赖 torch_sparse**：需要额外安装 `torch_sparse` 库
2. **需要排序**：必须先对节点按 chunk 排序，O(N log N)
3. **间接查询**：`ind2ptr` 是从 flat array 构建 CSR pointer，再查询
4. **内存开销**：需要存储 `inds` (sorted chunk IDs) 和 `perm` (permutation)

---

## 当前方法：直接使用 chunk_ptr CSR

### 数据结构
```python
ChunkAssignment:
    chunk_ptr: Tensor[num_chunks + 1]   # CSR pointer
    chunk_nodes: Tensor[num_nodes]      # CSR data (node IDs)
```

`chunk_ptr` 是 chunk -> nodes 的 CSR 表示：
- `chunk_ptr[c]` = chunk c 的起始位置
- `chunk_ptr[c+1]` = chunk c 的结束位置
- `chunk_nodes[chunk_ptr[c]:chunk_ptr[c+1]]` = chunk c 的所有节点

### 算法流程
```python
# 直接从 chunk_ptr 读取
chunk_ptr = chunk_assignment.chunk_ptr  # [num_chunks + 1]

ends_list = []
for k in chunk_decay:
    # endpoint = 前 k+1 个 chunks 的节点总数
    end = int(chunk_ptr[k + 1].item())
    ends_list.append(end)
```

### 示例
```python
# 假设有 10 个节点，3 个 chunks
# Chunk 0: nodes [0, 1]
# Chunk 1: nodes [2, 3, 4]
# Chunk 2: nodes [5, 6, 7, 8, 9]

chunk_ptr = [0, 2, 5, 10]
#            ^  ^  ^   ^
#            |  |  |   chunk 2 结束（10 个节点）
#            |  |  chunk 1 结束（5 个节点）
#            |  chunk 0 结束（2 个节点）
#            起始

chunk_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 按 chunk 排序的节点 ID

# 如果 chunk_decay = [2, 1, 0]（最近的优先）
ends_list = [
    chunk_ptr[2 + 1],  # chunk 2 结束位置 = 10
    chunk_ptr[1 + 1],  # chunk 1 结束位置 = 5
    chunk_ptr[0 + 1],  # chunk 0 结束位置 = 2
]
# ends_list = [10, 5, 2]
```

### 优势
1. **无需额外依赖**：不需要 `torch_sparse`
2. **O(1) 查询**：直接索引 `chunk_ptr`，无需排序
3. **内存高效**：`chunk_ptr` 已经在 `ChunkAssignment` 中，无需额外存储
4. **语义清晰**：`chunk_ptr[k+1]` 直接表示"前 k+1 个 chunks 的节点总数"

---

## 核心区别总结

| 维度 | FlareDTDG (ind2ptr) | 当前实现 (chunk_ptr CSR) |
|------|---------------------|--------------------------|
| **数据结构** | flat `node_to_chunk` | CSR `chunk_ptr` + `chunk_nodes` |
| **预处理** | 需要排序 O(N log N) | 已在 prepare 阶段构建 |
| **查询复杂度** | O(N) (ind2ptr) + O(1) (索引) | O(1) (直接索引) |
| **依赖** | `torch_sparse` | 无 |
| **内存** | 需要 `inds` + `perm` | 只需 `chunk_ptr` |
| **语义** | 间接：flat array -> CSR -> 查询 | 直接：CSR 查询 |

---

## 为什么 FlareDTDG 使用 ind2ptr？

FlareDTDG 的设计背景：
1. **没有预构建 ChunkAssignment**：FlareDTDG 的 chunk 是 runtime 动态分配的，没有 prepare 阶段的 CSR 结构
2. **节点按 chunk 排序**：FlareDTDG 在 loader 中动态对节点排序，所以需要 `ind2ptr` 从 sorted array 构建 CSR
3. **灵活性**：可以在 runtime 动态调整 chunk 分配

StarryUniGraph 的设计：
1. **预构建 ChunkAssignment**：在 prepare 阶段已经构建好 CSR 结构
2. **静态 chunk 分配**：chunk 分配在 prepare 阶段确定，runtime 不变
3. **性能优化**：利用预构建的 CSR 结构，避免 runtime 排序和 ind2ptr 计算

---

## 代码对比

### FlareDTDG
```python
# flare2/data/stc_loader.py:460-480
if self.chunk_index is not None and chunk_decay is not None:
    chunk_order = self.chunk_index.to(self.device, non_blocking=True)
    inds, perm = chunk_order.sort(dim=0)
    ends = torch.ops.torch_sparse.ind2ptr(inds, self.chunk_count)
    ends_list = [ends[t] for t in reversed(chunk_decay)]
    ends_list.extend([None] * num_full_snaps)
else:
    perm = None
    ends_list = [None] * num_full_snaps
```

### StarryUniGraph
```python
# backends/chunk/runtime/stg_loader.py:700-730
def _build_ends_list(self) -> Optional[List[Optional[int]]]:
    if self.chunk_decay is None or self.chunk_assignment is None:
        return None
    
    chunk_ptr = self.chunk_assignment.chunk_ptr  # [num_chunks + 1]
    ends: List[Optional[int]] = []
    
    for k in self.chunk_decay:
        end = int(chunk_ptr[k + 1].item())
        ends.append(end)
    
    ends.extend([None] * self.num_full_snaps)
    return ends
```

---

## Runtime chunk_order Override

当前实现还支持 runtime 动态覆盖 chunk 顺序（用于负载均衡）：

```python
def _build_ends_list_from_order(self, chunk_order: Tensor) -> List[Optional[int]]:
    """Runtime override: chunk_order[c] = priority rank of chunk c."""
    chunk_ptr = self.chunk_assignment.chunk_ptr
    chunk_sizes = chunk_ptr[1:] - chunk_ptr[:-1]  # [num_chunks]
    
    # Sort chunks by priority
    sorted_chunks = chunk_order.argsort()
    sorted_sizes = chunk_sizes[sorted_chunks]
    cumulative = sorted_sizes.cumsum(0)
    
    ends = [int(cumulative[i].item()) for i in range(num_decay)]
    ends.extend([None] * self.num_full_snaps)
    return ends
```

这样可以在 runtime 根据负载动态调整 chunk 优先级，而不需要重新构建 CSR。

---

## 结论

**CSR 方法优于 ind2ptr 方法**，因为：
1. ✅ 无需额外依赖
2. ✅ O(1) 查询，无需排序
3. ✅ 内存高效
4. ✅ 语义清晰
5. ✅ 支持 runtime 动态覆盖

FlareDTDG 使用 ind2ptr 是因为它没有预构建的 CSR 结构，而 StarryUniGraph 在 prepare 阶段已经构建好了 `ChunkAssignment`，所以可以直接利用 CSR 结构，避免 runtime 开销。
"""
