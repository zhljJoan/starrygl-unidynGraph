# 🔴 精度差异的根本原因 - 负采样策略错误

## 问题诊断

### 当前配置
```json
{
  "runtime": {
    "negative_sampler_policy": "uniform"  // ❌ 错误！
  }
}
```

### MemShare使用的策略
MemShare使用**local/global混合采样**，带有weight机制来平衡本地和远程负样本。

### 你的项目已经实现了正确的代码
在`src/atc_starrygl_lib/sampling/negative.py`中：
- ✅ `MemShareLocalNegativeSampler` 类已实现
- ✅ `_memshare_local_global_weight` 函数已实现
- ✅ Weight计算公式与MemShare完全一致

**但是配置文件中没有启用它！**

## 修复方案

### 方法1：修改配置文件（推荐）

编辑 `configs/ctdg_wiki_tgn_speed_hot01_bs600.json`：

```json
{
  "runtime": {
    "negative_ratio": 1,
    "negative_sampler_policy": "memshare_local",  // 改为memshare_local
    "negative_memshare_beta": 0.1,  // 添加beta参数（默认0.1）
    "negative_test_policy": "global"  // 测试时使用全局采样
  }
}
```

### 方法2：使用命令行参数

如果你的训练脚本支持命令行参数覆盖配置：

```bash
python train.py \
  --config configs/ctdg_wiki_tgn_speed_hot01_bs600.json \
  --negative_sampler_policy memshare_local \
  --negative_memshare_beta 0.1
```

## Beta参数说明

`beta`是采样远程节点的概率：
- `beta = 0.0`: 只从本地节点采样（不推荐）
- `beta = 0.1`: 10%概率从全局采样，90%从本地采样（**MemShare默认值**）
- `beta = 1.0`: 只从全局采样（等价于uniform）

### MemShare的Beta值

查看MemShare的配置：
```bash
cd /home/zlj/MemShare-public/MemShare
grep -r "beta\|probability" config/TGN_large.yml
```

通常MemShare使用`beta = 0.1`或动态调整。

## 完整的配置对比

### MemShare配置
```yaml
# config/TGN_large.yml
sample:
  - mode: 'recent'
    fanout: 20
    history: 1
    
# 负采样在代码中动态设置
# beta = ada_param.beta (通常为0.1)
```

### 你的配置应该改为
```json
{
  "runtime": {
    "negative_ratio": 1,
    "negative_sampler_policy": "memshare_local",
    "negative_memshare_beta": 0.1,
    "negative_test_policy": "global",
    "build_sampler": true,
    "policy": "recent",
    "fanout": [20],
    "history": 1
  }
}
```

## 验证修复

### 1. 检查日志
修改配置后，运行训练并检查日志中是否有weight相关的输出：

```python
# 在训练循环中添加调试输出
if batch_idx == 0:
    if batch.neg_weight is not None:
        print(f"✓ neg_weight is set: mean={batch.neg_weight.mean():.4f}")
        print(f"  local samples: {(batch.neg_weight < 2.0).sum().item()}")
        print(f"  remote samples: {(batch.neg_weight >= 2.0).sum().item()}")
    else:
        print("✗ neg_weight is None - using uniform sampling!")
```

### 2. 对比loss值
修复后，第一个epoch的loss应该与MemShare更接近。

### 3. 对比精度
修复后，Test AP/AUC应该与MemShare一致（误差<0.01）。

## 为什么这个问题会导致精度差异？

### Uniform采样的问题
```python
# Uniform采样：所有节点被采样的概率相同
# 但在分布式训练中：
# - 本地节点的梯度会被直接计算
# - 远程节点的梯度需要通信同步
# 
# 结果：本地节点被过度更新，远程节点更新不足
```

### MemShare的解决方案
```python
# 1. 混合采样：90%本地 + 10%全局
#    减少通信开销，同时保证全局覆盖
#
# 2. Weight机制：
#    local_weight = 1 / (0.9 + 0.1 * local_ratio)
#    remote_weight = 1 / (0.1 * local_ratio)
#    
#    补偿采样偏差，确保每个节点的有效更新率相同
```

### 影响程度
- **单机单卡**：影响较小（因为所有节点都是"本地"）
- **多GPU分布式**：影响**非常大**（可能导致AP差距>0.05）

## 预期结果

修复后：
- ✅ 训练loss应该与MemShare一致
- ✅ 收敛速度应该相似
- ✅ Test AP/AUC应该匹配（误差<0.01）

## 如果修复后仍有差异

如果修复后精度仍有小幅差异（<0.01），可能的原因：
1. 随机种子不同
2. 数据预处理的细微差异
3. 浮点数累加顺序不同

但如果差异仍然很大（>0.02），请检查：
1. Beta值是否正确（应该是0.1）
2. 是否在分布式训练中正确设置了local_dst_pool
3. 数据分区是否一致

## 立即行动

1. **修改配置文件**：
   ```bash
   cd /home/zlj/ATC_StarryglLib
   # 备份原配置
   cp configs/ctdg_wiki_tgn_speed_hot01_bs600.json configs/ctdg_wiki_tgn_speed_hot01_bs600.json.bak
   
   # 编辑配置，添加：
   # "negative_sampler_policy": "memshare_local",
   # "negative_memshare_beta": 0.1,
   ```

2. **重新训练**：
   ```bash
   python train.py --config configs/ctdg_wiki_tgn_speed_hot01_bs600.json
   ```

3. **对比结果**：
   记录修复前后的Test AP/AUC，应该有显著提升。
