# 🔴 预处理问题诊断

## 关键发现

### 1. 精度差距巨大
- **你的项目**: 84% AP
- **MemShare**: 95% AP
- **差距**: 11% (非常大！)

### 2. DTDG也有同样问题
你提到"dtdg经过当前预处理之后精度也偏低"，这强烈说明是**预处理的问题**，而不是模型实现的问题。

### 3. WIKI数据集的特殊性

WIKI数据集是一个**二部图**：
```
src节点: 1 - 8227 (8227个节点)
dst节点: 8228 - 9227 (1000个节点)
总节点数: 9227
节点ID范围: 1 - 9227 (从1开始，不是0！)
```

## 可能的预处理问题

### 问题1: 节点ID从1开始，但代码假设从0开始

如果你的代码假设节点ID从0开始，会导致：
- 节点0永远不会被访问（浪费一个位置）
- 节点9227会越界（如果num_nodes=9227）
- 或者节点9227的memory/mailbox永远不会被初始化（如果num_nodes=9228）

**检查方法**：
```python
# 在你的代码中打印
print(f"src min: {batch.src.min()}, max: {batch.src.max()}")
print(f"dst min: {batch.dst.min()}, max: {batch.dst.max()}")
print(f"num_nodes: {num_nodes}")
```

**正确的处理**：
- 要么将节点ID减1（变成0-based）
- 要么设置num_nodes = max(node_id) + 1 = 9228

### 问题2: num_nodes计算错误

可能的错误：
```python
# 错误1: 使用unique().numel()
num_nodes = torch.cat([src, dst]).unique().numel()  # = 9227

# 错误2: 使用max() + 1但节点ID从1开始
num_nodes = max(src.max(), dst.max()) + 1  # = 9228，但节点0浪费了

# 正确: 如果节点ID从1开始，需要减1
src = src - 1  # 变成0-based
dst = dst - 1
num_nodes = max(src.max(), dst.max()) + 1  # = 9227
```

### 问题3: 数据分割错误

WIKI数据集使用`ext_roll`或`int_roll`来标记train/val/test：
```
0 = train
1 = val
2 = test
```

如果你的预处理使用了错误的分割方式（比如按时间比例分割），会导致：
- 训练集和测试集的分布不同
- 精度大幅下降

**检查方法**：
```python
# 读取原始数据
df = pd.read_csv('WIKI/edges.csv')
print(df['ext_roll'].value_counts())
# 应该看到0, 1, 2三个值

# 检查你的split
print(f"Train edges: {(split == 0).sum()}")
print(f"Val edges: {(split == 1).sum()}")
print(f"Test edges: {(split == 2).sum()}")
```

### 问题4: 边特征维度错误

WIKI数据集没有原始边特征，使用随机特征：
```json
"random_node_feat_dim": 172
```

但这个名字容易混淆 - 实际上是**边特征**的维度！

**检查方法**：
```python
print(f"edge_feat shape: {edge_feat.shape}")
# 应该是 [num_edges, 172]
```

## 立即诊断步骤

### Step 1: 检查节点ID范围

在你的训练代码中添加：
```python
# 在第一个batch
if batch_idx == 0:
    print(f"=== 第一个batch的节点ID ===")
    print(f"src: min={batch.src.min()}, max={batch.src.max()}")
    print(f"dst: min={batch.dst.min()}, max={batch.dst.max()}")
    print(f"roots: min={batch.roots.min()}, max={batch.roots.max()}")
    print(f"num_nodes (配置): {num_nodes}")
    print(f"memory shape: {memory.shape}")
    print(f"mailbox shape: {mailbox.shape}")
```

**预期结果**：
- 如果节点ID从0开始：min=0, max=9226
- 如果节点ID从1开始：min=1, max=9227

### Step 2: 检查数据分割

```python
import pandas as pd
import torch

# 读取原始数据
df = pd.read_csv('/mnt/data/zlj/starrygl-data/raw/TGL-DATA/WIKI/edges.csv')
print("=== 原始数据分割 ===")
print(df['ext_roll'].value_counts().sort_index())

# 读取你的预处理后的数据
# 检查split是否匹配
```

### Step 3: 对比MemShare的预处理

运行MemShare的预处理，保存中间结果：
```python
# 在MemShare中
print(f"MemShare num_nodes: {graph.num_nodes}")
print(f"MemShare src range: {graph.edge_index[0].min()} - {graph.edge_index[0].max()}")
print(f"MemShare dst range: {graph.edge_index[1].min()} - {graph.edge_index[1].max()}")
```

然后对比你的项目的相同信息。

## 最可能的问题

根据经验，**最可能的问题是**：

1. **节点ID没有减1** (概率60%)
   - 原始数据从1开始
   - 代码假设从0开始
   - 导致节点0的memory/mailbox从未被使用
   - 节点9227可能越界或未初始化

2. **数据分割方式错误** (概率30%)
   - 应该使用ext_roll/int_roll
   - 但可能使用了时间比例分割
   - 导致train/val/test分布不同

3. **num_nodes计算错误** (概率10%)
   - 可能是9227（少了1个）
   - 或者是9228（多了1个）

## 建议

**立即执行Step 1**，打印第一个batch的节点ID范围，这会立即告诉我们问题所在。

如果节点ID从1开始，你需要：
1. 在预处理时将所有节点ID减1
2. 或者在模型中处理这个offset

请运行Step 1的代码并告诉我结果！
