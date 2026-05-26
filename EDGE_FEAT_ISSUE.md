# 🔍 关键问题总结

## 当前状态
- **你的精度**: 84% AP
- **MemShare精度**: 95% AP (你自己运行MemShare原始代码得到)
- **差距**: 11%
- **测试环境**: 单机单卡
- **DTDG**: 用你的预处理也偏低

## 已排除的问题
1. ✅ Mailbox更新逻辑 - 已修复但精度没变化
2. ✅ 负采样策略 - 已修复但精度没变化  
3. ✅ 分布式训练问题 - 单机单卡也有问题
4. ✅ 节点ID offset - 只会浪费空间，不影响精度
5. ✅ 数据分割 - 使用ext_roll，且数据已排序
6. ✅ 数据分割比例 - 70%/15%/15%一致

## 需要立即检查的

### 🔴 边特征问题（最可疑）

你的配置：
```json
{
  "random_node_feat_dim": 172,  // 生成节点特征
  "random_edge_feat_dim": 0      // 没有边特征！
}
```

**问题**：WIKI数据集没有原始边特征，需要生成随机边特征。但你的配置生成的是**节点特征**，不是**边特征**！

**MemShare的处理**：
- MemShare使用`graph.efeat`作为边特征
- 维度是172

**可能的错误**：
1. 你的代码可能没有生成边特征
2. 或者生成了但没有正确使用
3. 或者mailbox消息中缺少边特征

### 验证方法

在你的训练代码中添加：

```python
# 在第一个batch
if batch_idx == 0:
    print("=" * 60)
    print("特征维度检查")
    print("=" * 60)
    if hasattr(batch, 'edge_feat') and batch.edge_feat is not None:
        print(f"✓ edge_feat存在: {batch.edge_feat.shape}")
        print(f"  mean: {batch.edge_feat.mean():.6f}")
        print(f"  std: {batch.edge_feat.std():.6f}")
    else:
        print("✗ edge_feat不存在或为None")
    
    if hasattr(batch, 'node_feat') and batch.node_feat is not None:
        print(f"node_feat: {batch.node_feat.shape}")
    
    # 检查mailbox消息维度
    print(f"\nmailbox_msg_dim配置: {config['runtime']['mailbox_msg_dim']}")
    print(f"预期: 2 * memory_dim + edge_feat_dim")
    print(f"    = 2 * 100 + 172 = 372")
    print("=" * 60)
```

**如果edge_feat不存在或为None，这就是问题所在！**

## 可能的修复

### 方案1：修改配置（如果是配置问题）

```json
{
  "random_node_feat_dim": 0,      // 不需要节点特征
  "random_edge_feat_dim": 172,    // 生成边特征
  "random_edge_feat_seed": 0
}
```

### 方案2：检查代码（如果是代码问题）

检查以下位置：
1. 预处理时是否生成了edge_feat
2. 数据加载时是否正确加载edge_feat
3. Batch构建时是否包含edge_feat
4. Mailbox消息构建时是否使用edge_feat

## 下一步

**请立即运行上面的验证代码**，检查edge_feat是否存在。

如果edge_feat不存在，这很可能就是导致精度从95%降到84%的根本原因！

边特征对于TGNN非常重要，因为：
- 边特征包含交互的上下文信息
- Mailbox消息需要边特征
- Memory更新依赖边特征
- 缺少边特征会导致模型学习能力大幅下降
