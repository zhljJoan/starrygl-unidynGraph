# 负采样Weight机制 - 详细分析和修复方案

## MemShare的train_ratio计算公式

### 关键代码
```python
# 在LocalNegativeSampling.sample()中
prob = self.ada_param.beta  # 采样远程节点的概率
remote_ratio = self.local_dst.shape[0] / self.dst_node_list.shape[0]  # 本地节点占比

# 计算权重
self.train_ratio_pos = 1.0 / (1 - prob + prob * remote_ratio) if ((prob < 1) & (prob > 0)) else 1
self.train_ratio_neg = 1.0 / (prob * remote_ratio) if ((prob < 1) & (prob > 0)) else 1
```

### 公式推导

假设：
- `prob` = β = 采样远程节点的概率（来自ada_param.beta）
- `remote_ratio` = r = 本地节点数 / 全局节点数
- 每个batch采样N个负样本

**实际采样分布**：
- 采样到本地节点的概率：`(1 - β) + β * r`
  - `(1 - β)`: 直接从本地采样的概率
  - `β * r`: 从全局采样但恰好是本地节点的概率
- 采样到远程节点的概率：`β * (1 - r)`

**期望的均匀分布**：
- 每个节点应该有相同的被采样概率：`1 / total_nodes`

**权重计算**：
为了补偿采样偏差，需要对loss加权：
```
train_ratio_pos = 1 / [(1 - β) + β * r]  # 本地节点的权重
train_ratio_neg = 1 / [β * (1 - r)]      # 远程节点的权重
```

但MemShare的代码中使用的是：
```python
train_ratio_pos = 1.0 / (1 - prob + prob * remote_ratio)
train_ratio_neg = 1.0 / (prob * remote_ratio)  # 注意：这里是 prob * remote_ratio，不是 prob * (1 - remote_ratio)
```

**这里有个问题**：`train_ratio_neg`的公式似乎不对。让我重新理解：

实际上，`remote_ratio`的定义是：
```python
remote_ratio = self.local_dst.shape[0] / self.dst_node_list.shape[0]
```
这是**本地节点占全局节点的比例**，不是远程节点的比例。

所以：
- 本地节点占比：`r = local_count / total_count`
- 远程节点占比：`1 - r`

**重新推导**：

采样策略：
```python
p = torch.rand(size=(num_samples,))
s = torch.where(p <= prob, sr, sl)  # prob概率采样远程(sr)，否则采样本地(sl)
```

实际采样到本地节点的概率：
- 直接从本地采样：`(1 - prob)`
- 从远程采样但恰好是本地：`prob * r`
- 总计：`(1 - prob) + prob * r`

实际采样到远程节点的概率：
- 从远程采样且确实是远程：`prob * (1 - r)`

但是，MemShare的代码中`train_ratio_neg = 1.0 / (prob * remote_ratio)`，这里的`remote_ratio`实际上是本地占比`r`，所以：
```
train_ratio_neg = 1.0 / (prob * r)
```

**这个公式看起来不对！** 让我再检查一下...

实际上，我理解错了。让我重新看采样代码：
```python
sr = self.dst_node_list[...]  # 从全局节点列表采样
sl = self.local_dst[...]      # 从本地节点列表采样
s = torch.where(p <= prob, sr, sl)
```

所以：
- `prob`概率从**全局**采样（sr）
- `(1-prob)`概率从**本地**采样（sl）

从全局采样时：
- 采样到本地节点的概率：`r`
- 采样到远程节点的概率：`1 - r`

总的采样概率：
- 本地节点：`(1 - prob) + prob * r`
- 远程节点：`prob * (1 - r)`

**权重应该是采样概率的倒数**：
```
train_ratio_pos = 1 / [(1 - prob) + prob * r]
train_ratio_neg = 1 / [prob * (1 - r)]
```

但MemShare用的是：
```python
train_ratio_neg = 1.0 / (prob * remote_ratio)  # remote_ratio = r
```

这意味着MemShare把`remote_ratio`当作远程节点的占比`(1-r)`来用了？

**让我重新检查remote_ratio的定义**...

## 实际情况

看代码：
```python
remote_ratio = self.local_dst.shape[0] / self.dst_node_list.shape[0]
```

- `self.local_dst`: 本地节点列表
- `self.dst_node_list`: 全局节点列表
- `remote_ratio`: **本地节点占比**

所以`remote_ratio`实际上是本地占比`r`，不是远程占比。

那么MemShare的公式：
```python
train_ratio_pos = 1.0 / (1 - prob + prob * remote_ratio)  # = 1 / [(1-β) + β*r]  ✓ 正确
train_ratio_neg = 1.0 / (prob * remote_ratio)             # = 1 / (β*r)  ✗ 错误？
```

**等等**，让我再看一遍采样逻辑...

```python
sr = self.dst_node_list[torch.randint(len(self.dst_node_list), (num_samples,))]  # 全局采样
sl = self.local_dst[torch.randint(len(self.local_dst), (num_samples,))]          # 本地采样
s = torch.where(p <= prob, sr, sl)
```

哦！我明白了！

- `sr`是从**全局列表**中uniform采样
- `sl`是从**本地列表**中uniform采样

当`p <= prob`时，选择`sr`（全局采样的结果）。

**关键点**：`sr`可能是本地节点，也可能是远程节点！

所以实际上：
- 采样到本地节点：`(1-prob) * 1 + prob * r = (1-prob) + prob*r`
- 采样到远程节点：`prob * (1-r)`

但是，在loss计算时：
```python
is_local = DistIndex(mfgs[0][0].srcdata['ID'][metadata['dst_neg_index']]).part == rank
weight = torch.where(is_local, ones * train_ratio_pos, ones * train_ratio_neg)
```

这里判断的是**实际采样到的节点**是本地还是远程。

所以：
- `train_ratio_pos`：给**实际是本地节点**的负样本的权重
- `train_ratio_neg`：给**实际是远程节点**的负样本的权重

**正确的公式应该是**：
```
train_ratio_pos = 1 / [(1-prob) + prob*r]      # 本地节点的采样概率
train_ratio_neg = 1 / [prob * (1-r)]           # 远程节点的采样概率
```

但MemShare用的是：
```python
train_ratio_neg = 1.0 / (prob * remote_ratio)  # = 1 / (prob * r)
```

**这确实是错的！** 除非...

让我再看一次变量名...`remote_ratio`...

哦！我一直理解反了！让我重新看：

```python
remote_ratio = self.local_dst.shape[0] / self.dst_node_list.shape[0]
```

等等，这个命名很confusing。`remote_ratio`实际上是**本地占比**。

让我假设MemShare的命名是对的，那么：
- `remote_ratio` = 远程节点占比 = `1 - r`

那么：
```python
self.local_dst.shape[0] / self.dst_node_list.shape[0]
```
应该是远程节点数 / 全局节点数？

**不对**，`self.local_dst`明明是本地节点列表...

## 结论

MemShare的代码中存在命名混淆：
- `remote_ratio`实际上是**本地节点占比**
- 但在公式中被当作**远程节点占比**使用

**可能的情况**：
1. 代码有bug，但恰好在某些情况下work
2. 我理解错了某些变量的含义
3. 这个公式是经验性的，不是理论推导的

## 你的项目应该如何实现？

### 方案1：完全复制MemShare的实现（推荐）

即使MemShare的公式可能有问题，为了复现精度，应该**完全复制**它的实现：

```python
# 在负采样器中
class YourNegativeSampler:
    def __init__(self, local_nodes, global_nodes, prob):
        self.local_nodes = local_nodes
        self.global_nodes = global_nodes
        self.prob = prob
        
        # 计算ratio（注意：这里用本地占比，但变量名叫remote_ratio）
        self.remote_ratio = len(local_nodes) / len(global_nodes)
        
        # 计算train_ratio（完全复制MemShare的公式）
        if 0 < prob < 1:
            self.train_ratio_pos = 1.0 / (1 - prob + prob * self.remote_ratio)
            self.train_ratio_neg = 1.0 / (prob * self.remote_ratio)
        else:
            self.train_ratio_pos = 1.0
            self.train_ratio_neg = 1.0
    
    def sample(self, num_samples):
        p = torch.rand(num_samples)
        global_samples = self.global_nodes[torch.randint(len(self.global_nodes), (num_samples,))]
        local_samples = self.local_nodes[torch.randint(len(self.local_nodes), (num_samples,))]
        samples = torch.where(p <= self.prob, global_samples, local_samples)
        return samples

# 在训练循环中
neg_samples = negative_sampler.sample(batch_size)
is_local = dist_index_part(neg_samples) == dist.get_rank()

weight = torch.where(
    is_local,
    torch.ones_like(pred_neg) * negative_sampler.train_ratio_pos,
    torch.ones_like(pred_neg) * negative_sampler.train_ratio_neg
)

neg_criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
neg_loss = neg_criterion(pred_neg, torch.zeros_like(pred_neg))
```

### 方案2：理论正确的实现

如果你想用理论正确的公式：

```python
# 本地节点占比
local_ratio = len(local_nodes) / len(global_nodes)

# 采样概率
prob_local = (1 - prob) + prob * local_ratio
prob_remote = prob * (1 - local_ratio)

# 权重（采样概率的倒数）
train_ratio_pos = 1.0 / prob_local
train_ratio_neg = 1.0 / prob_remote
```

## 建议

**为了复现MemShare的精度，使用方案1**（完全复制MemShare的实现），即使公式看起来不太对。

之后如果想改进，可以尝试方案2，但需要重新调参。
