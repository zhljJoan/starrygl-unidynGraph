# 精度差异问题总结与下一步行动

## 当前状态
- ✅ 已修复mailbox更新逻辑（移除时间戳检查）
- ⚠️ 精度略有改善但仍有差距

## 关键发现

### 1. Mailbox逻辑差异（已修复）
- **问题**: ATC_StarryglLib检查时间戳，拒绝旧消息；MemShare直接覆盖
- **影响**: 中等 - 会导致使用不同的mailbox内容
- **状态**: 已修复

### 2. 混合精度训练
- **MemShare**: 代码中有`scaler = torch.cuda.amp.GradScaler()`，但`with autocast()`被注释
- **结论**: MemShare **没有使用**混合精度训练
- **建议**: 确认你的项目也没有使用混合精度

### 3. 其他已验证一致的部分
- Mailbox消息构建
- Memory updater实现
- 配置参数（memory_dim, mailbox_size等）
- mem_input reshape

## 可能的剩余差异点

### 高优先级（最可能导致精度差异）

#### 1. 数据预处理和分区
```python
# 需要检查的点：
- 数据分割的随机种子
- 分区算法的实现细节
- 时间戳的处理方式
- 边的排序方式
```

**检查方法**:
```python
# 在两个项目中打印第一个batch的内容
print("First batch src:", batch.src[:10])
print("First batch dst:", batch.dst[:10])
print("First batch ts:", batch.ts[:10])
```

#### 2. 负采样
```python
# MemShare使用weight来处理跨分区的负样本
weight = torch.where(
    DistIndex(mfgs[0][0].srcdata['ID'][metadata['dst_neg_index']]).part == rank,
    ones * metadata['train_ratio_pos'],
    ones * metadata['train_ratio_neg']
).reshape(-1, 1)
neg_creterion = torch.nn.BCEWithLogitsLoss(weight)
```

**你的项目是否也有这个weight机制？**

#### 3. 模型权重初始化
PyTorch的默认初始化依赖于操作顺序，即使设置了随机种子，如果模块创建顺序不同，初始权重也会不同。

**检查方法**:
```python
# 在训练前保存初始权重
torch.save(model.state_dict(), 'init_weights.pth')

# 对比两个项目的初始权重
import torch
w1 = torch.load('memshare_init.pth')
w2 = torch.load('atc_init.pth')
for k in w1.keys():
    if k in w2:
        diff = (w1[k] - w2[k]).abs().max().item()
        if diff > 1e-6:
            print(f"{k}: diff={diff}")
```

### 中优先级

#### 4. 训练循环细节
- Gradient accumulation
- Gradient clipping
- 参数更新顺序

#### 5. 分布式训练
- AllReduce的时机
- 通信同步点
- 数据分区的一致性

### 低优先级

#### 6. 数值精度
- float32 vs float64
- 累加顺序（浮点数累加不满足结合律）

## 推荐的调试步骤

### Step 1: 单机单卡对比（排除分布式影响）
```bash
# 使用1个GPU，相同的数据和配置
CUDA_VISIBLE_DEVICES=0 python train.py --world_size=1
```

### Step 2: 固定所有随机性
```python
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### Step 3: 逐层对比第一个batch
```python
# 在memory updater的forward中添加
if batch_idx == 0:
    print(f"[Rank {rank}] mem_input mean: {b.srcdata['mem_input'].mean():.6f}")
    print(f"[Rank {rank}] mem mean: {b.srcdata['mem'].mean():.6f}")
    print(f"[Rank {rank}] updated_memory mean: {updated_memory.mean():.6f}")
    
    # 保存到文件
    torch.save({
        'mem_input': b.srcdata['mem_input'],
        'mem': b.srcdata['mem'],
        'updated_memory': updated_memory,
    }, f'debug_batch0_rank{rank}.pth')
```

### Step 4: 对比loss曲线
```python
# 记录每个batch的loss
losses = []
for batch in dataloader:
    loss = train_step(batch)
    losses.append(loss.item())

# 保存并对比
torch.save(losses, 'losses.pth')
```

### Step 5: 检查负采样的weight
这是一个**非常可疑**的点。MemShare对跨分区的负样本使用了不同的权重：

```python
# 检查你的项目是否有类似逻辑
if args.local_neg_sample is False:
    weight = torch.where(
        DistIndex(...).part == rank,
        ones * train_ratio_pos,
        ones * train_ratio_neg
    )
    neg_criterion = torch.nn.BCEWithLogitsLoss(weight)
```

## 需要你提供的信息

1. **精度数值**:
   - MemShare Test AP: ?
   - ATC修复前 Test AP: ?
   - ATC修复后 Test AP: ?
   - 差距是多少？（例如：0.95 vs 0.93）

2. **训练设置**:
   - GPU数量: ?
   - 是否使用分布式训练: ?
   - Batch size: ?

3. **负采样**:
   - 你的项目是否实现了跨分区负样本的weight机制？
   - 负采样策略是什么？（uniform/local）

4. **观察**:
   - Loss曲线是否相似？
   - 第一个epoch的loss是否接近？
   - 训练是否稳定？

## 最可能的原因（按概率排序）

1. **负采样的weight机制不同** (60%)
2. **数据预处理/分区方式不同** (20%)
3. **模型初始化不同** (10%)
4. **其他训练细节** (10%)

请先检查负采样的weight机制，这很可能是主要原因。
