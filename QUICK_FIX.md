# 🚀 快速修复指南

## 问题
ATC_StarryglLib的精度低于MemShare-public

## 根本原因
**负采样策略配置错误** - 使用了`uniform`而不是`memshare_local`

## 已修复的内容

### ✅ 1. Mailbox更新逻辑
**文件**: `src/atc_starrygl_lib/memory/mailbox.py`
- 移除了时间戳检查，匹配MemShare的循环队列行为

### ✅ 2. 负采样配置
**文件**: `configs/ctdg_wiki_tgn_speed_hot01_bs600.json`
```json
{
  "runtime": {
    "negative_sampler_policy": "memshare_local",  // 改为memshare_local
    "negative_memshare_beta": 0.1,                // 添加beta参数
    "negative_test_policy": "global"              // 添加测试策略
  }
}
```

## 立即测试

```bash
cd /home/zlj/ATC_StarryglLib

# 重新训练
python train.py --config configs/ctdg_wiki_tgn_speed_hot01_bs600.json

# 或者如果使用分布式训练
torchrun --nproc_per_node=2 train.py --config configs/ctdg_wiki_tgn_speed_hot01_bs600.json
```

## 验证修复

在训练开始时，应该看到类似输出：
```
✓ Using MemShare negative sampling
  neg_weight mean: 1.5234
  local samples: 450
  remote samples: 150
```

如果看到`✗ WARNING: neg_weight is None!`，说明配置没有生效。

## 预期结果

- ✅ Test AP/AUC应该与MemShare一致（误差<0.01）
- ✅ 训练loss曲线应该相似
- ✅ 收敛速度应该接近

## 如果仍有问题

1. 检查是否使用了正确的配置文件
2. 检查beta值是否为0.1
3. 查看`SOLUTION_SUMMARY.md`获取详细信息
4. 查看`FIX_NEGATIVE_SAMPLING.md`获取调试步骤

## 关键文件

- `SOLUTION_SUMMARY.md` - 完整解决方案
- `FIX_NEGATIVE_SAMPLING.md` - 负采样修复详解
- `WEIGHT_MECHANISM_EXPLAINED.md` - Weight机制原理
- `ACCURACY_DIFF_ANALYSIS.md` - 完整对比分析

---

**修复完成！现在可以重新训练了。** 🎉
