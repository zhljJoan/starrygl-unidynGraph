# 负采样Weight机制详解

## MemShare的Weight机制

### 问题背景
在分布式训练中，负样本可能来自：
1. **本地分区**的节点（local negative）
2. **远程分区**的节点（remote negative）

由于数据分区，每个rank只能看到部分节点。如果不加权重，会导致：
- 本地负样本被过度惩罚（因为它们更容易被采样到）
- 远程负样本权重不足

### MemShare的解决方案

```python
# 在训练循环中
if args.local_neg_sample is False:  # 使用全局负采样
    # 判断每个负样本是本地还是远程
    is_local = DistIndex(mfgs[0][0].srcdata['ID'][metadata['dst_neg_index']]).part == torch.distributed.get_rank()
    
    # 根据本地/远程设置不同的权重
    weight = torch.where(
        is_local,
        ones * metadata['train_ratio_pos'],  # 本地负样本的权重
        ones * metadata['train_ratio_neg']   # 远程负样本的权重
    ).reshape(-1, 1)
    
    # 使用加权的loss
    neg_criterion = torch.nn.BCEWithLogitsLoss(weight)
    loss += neg_criterion(pred_neg, torch.zeros_like(pred_neg))
else:  # 使用本地负采样
    # 不需要权重，因为所有负样本都来自本地
    loss += criterion(pred_neg, torch.zeros_like(pred_neg))
```

### train_ratio的含义

- `train_ratio_pos`: 本地负样本的权重（通常较大）
- `train_ratio_neg`: 远程负样本的权重（通常较小）

这两个比例反映了：
- 本地节点在全局中的占比
- 远程节点在全局中的占比

### 为什么需要这个机制？

**示例**：
- 假设有2个GPU，每个GPU负责50%的节点
- 如果使用全局负采样（从所有节点中采样）：
  - 采样到本地节点的概率：50%
  - 采样到远程节点的概率：50%

但在实际训练中：
- 本地节点的梯度会被**直接计算**
- 远程节点的梯度需要**通信同步**

如果不加权重：
- 本地节点会被过度更新（因为每个batch都会看到它们）
- 远程节点更新不足

**Weight的作用**：
```python
# 假设world_size=2，每个rank有50%的节点
train_ratio_pos = 1.0 / 0.5 = 2.0  # 本地节点权重
train_ratio_neg = 1.0 / 0.5 = 2.0  # 远程节点权重

# 实际上，这个比例可能根据实际分区情况动态计算
```

## 你的项目是否实现了这个机制？

### 检查方法

1. **查看train_loop.py中的loss计算**：
```bash
grep -A 20 "pred_neg" src/atc_starrygl_lib/ctdg/train_loop.py
```

2. **查看是否有weight参数**：
```bash
grep -r "BCEWithLogitsLoss" src/atc_starrygl_lib/ctdg/ -B 5 -A 5
```

3. **查看负采样器的实现**：
```bash
grep -r "negative_sampler\|NegativeSampler" src/atc_starrygl_lib/sampling/ -A 10
```

## 如果你的项目没有这个机制

### 影响
- **严重**：在分布式训练中，会导致模型收敛到次优解
- **单机训练**：影响较小或没有影响

### 修复方案

需要在训练循环中添加类似的weight机制：

```python
# 在train_loop.py中
def train_epoch(...):
    for batch in dataloader:
        pred_pos, pred_neg = model(batch)
        
        # 正样本loss（不需要权重）
        pos_loss = criterion(pred_pos, torch.ones_like(pred_pos))
        
        # 负样本loss（需要权重）
        if use_global_negative_sampling:
            # 判断负样本是本地还是远程
            neg_node_ids = batch.negative_dst  # 负样本节点ID
            is_local = dist_index_part(neg_node_ids) == dist.get_rank()
            
            # 计算权重
            local_ratio = batch.metadata.get('local_ratio', 1.0)
            remote_ratio = batch.metadata.get('remote_ratio', 1.0)
            
            weight = torch.where(
                is_local,
                torch.ones_like(pred_neg) * local_ratio,
                torch.ones_like(pred_neg) * remote_ratio
            )
            
            # 使用加权loss
            neg_criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
            neg_loss = neg_criterion(pred_neg, torch.zeros_like(pred_neg))
        else:
            # 本地负采样，不需要权重
            neg_loss = criterion(pred_neg, torch.zeros_like(pred_neg))
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
```

## 如何计算train_ratio？

### 方法1：基于分区大小
```python
# 在数据加载时计算
local_node_count = len(local_nodes)
total_node_count = all_reduce_sum(local_node_count)
local_ratio = total_node_count / local_node_count
```

### 方法2：基于采样概率
```python
# 如果负采样是uniform的
world_size = dist.get_world_size()
local_ratio = world_size  # 因为每个rank负责1/world_size的节点
remote_ratio = world_size / (world_size - 1)  # 远程节点来自其他world_size-1个rank
```

## 验证是否是这个问题

### 实验1：单机单卡测试
```bash
# 如果单机单卡精度一致，说明是分布式训练的问题
CUDA_VISIBLE_DEVICES=0 python train.py --world_size=1
```

### 实验2：对比loss值
```python
# 在训练循环中打印
print(f"Pos loss: {pos_loss.item():.4f}")
print(f"Neg loss: {neg_loss.item():.4f}")
print(f"Local neg count: {is_local.sum().item()}")
print(f"Remote neg count: {(~is_local).sum().item()}")
```

如果你的项目中负样本loss异常大或异常小，很可能就是这个weight机制的问题。

## 总结

这个weight机制是MemShare在分布式训练中的一个**关键设计**，用于平衡本地和远程负样本的贡献。如果你的项目没有实现这个机制，在多GPU训练时会导致：

1. 模型收敛到次优解
2. 精度低于MemShare
3. 训练不稳定

**优先级：🔴 非常高** - 这很可能是导致精度差异的主要原因。
