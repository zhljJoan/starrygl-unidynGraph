# 精度差异调试清单

## 已修复的问题
✅ Mailbox更新逻辑 - 移除了时间戳检查，匹配MemShare的循环队列行为

## 已验证一致的部分
✅ Mailbox消息构建 - `[mem_src, mem_dst, edge_feat]`
✅ Memory updater - GRU实现一致
✅ combine_node_feature - 特征组合逻辑一致
✅ mem_input reshape - `mailbox.reshape(N, -1)` 一致
✅ 配置参数 - memory_dim, mailbox_size, hidden_dim等都一致
✅ _latest_by_row vs scatter_max - 行为完全一致

## 需要进一步检查的点

### 1. 训练流程细节
- [ ] 优化器配置（Adam参数、weight_decay）
- [ ] 学习率调度器
- [ ] Gradient clipping
- [ ] 随机种子设置（包括cudnn.deterministic）

### 2. 数据预处理
- [ ] 数据分割方式（train/val/test split）
- [ ] 时间戳归一化
- [ ] 特征归一化
- [ ] 边的顺序是否一致

### 3. 负采样
- [ ] 负采样策略（uniform vs local）
- [ ] 负样本数量
- [ ] 负样本池的构建方式

### 4. Memory/Mailbox初始化
- [ ] 初始值（zeros vs random）
- [ ] dtype（float32 vs float64）
- [ ] device placement

### 5. 分布式训练细节
- [ ] 数据分区方式
- [ ] 通信同步点
- [ ] Gradient同步方式

### 6. 模型初始化
- [ ] 权重初始化方式
- [ ] Bias初始化
- [ ] LayerNorm/BatchNorm参数

## 调试建议

### 方法1: 逐层对比
在第一个batch的每一步打印中间结果：
```python
# 在memory updater中
print(f"Batch 0 - mem_input: {b.srcdata['mem_input'][:5]}")
print(f"Batch 0 - updated_memory: {updated_memory[:5]}")
```

### 方法2: 保存checkpoint对比
```python
# MemShare
torch.save(model.state_dict(), 'memshare_epoch0.pth')

# ATC_StarryglLib
torch.save(encoder.state_dict(), 'atc_epoch0.pth')

# 对比权重
import torch
m1 = torch.load('memshare_epoch0.pth')
m2 = torch.load('atc_epoch0.pth')
for k in m1.keys():
    if k in m2:
        diff = (m1[k] - m2[k]).abs().max()
        print(f"{k}: max_diff={diff}")
```

### 方法3: 固定随机种子测试
确保两个项目使用完全相同的随机种子：
```python
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 方法4: 单机单卡测试
先在单机单卡环境下对比，排除分布式训练的影响。

## 需要的信息

请提供以下信息以进一步诊断：

1. **精度数值对比**：
   - MemShare的Test AP/AUC: ?
   - ATC_StarryglLib修复前的Test AP/AUC: ?
   - ATC_StarryglLib修复后的Test AP/AUC: ?

2. **训练配置**：
   - 使用的GPU数量: ?
   - Batch size: ?
   - 训练的epoch数: ?

3. **数据集信息**：
   - 使用的数据集: WikiTalk
   - 数据分区方式: ?
   - 节点数/边数: ?

4. **观察到的现象**：
   - Loss曲线是否相似？
   - 收敛速度是否一致？
   - 是否有异常的梯度或loss值？
