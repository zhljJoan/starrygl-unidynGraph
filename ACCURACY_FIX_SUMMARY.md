# 精度差异修复总结

## 问题根源

通过系统性对比 ATC_StarryglLib 和 MemShare-public 的实现，发现了导致精度差异的**关键问题**：

### Mailbox更新逻辑的差异

**MemShare的行为**：
- 使用循环队列方式更新mailbox
- **不检查**新消息的时间戳是否比已有消息更新
- 允许旧消息覆盖新消息

**ATC_StarryglLib的原始行为**：
- 在更新mailbox前检查时间戳
- 只有当新消息的时间戳**大于**mailbox中最旧消息的时间戳时才写入
- 拒绝旧消息，保护新消息不被覆盖

### 具体场景示例

```python
# 场景：节点0的mailbox已有 ts=2.0 的消息，现在收到 ts=1.5 的旧消息

# MemShare结果：
mailbox[0] = ts=1.5  # 旧消息覆盖了新消息

# ATC_StarryglLib原始结果：
mailbox[0] = ts=2.0  # 拒绝旧消息，保留新消息
```

这导致训练过程中memory updater读取到不同的mailbox内容，产生不同的memory更新，最终影响模型精度。

## 修复方案

修改了 `src/atc_starrygl_lib/memory/mailbox.py` 中的两个方法：

### 1. `append_rows()` 方法

**修改前**：
```python
if reduce == "max_ts":
    oldest = self.mailbox_ts[row].min(dim=1).values
    keep = ts_in > oldest  # 检查是否比已有消息更新
    if not keep.any():
        return
    row = row[keep]
    message = message[keep]
    ts_in = ts_in[keep]
```

**修改后**：
```python
# NOTE: MemShare-compatible behavior - do NOT check against existing timestamps
# Removing this check to match MemShare behavior for accuracy reproduction
elif reduce != "append" and reduce != "max_ts":
    raise ValueError(f"unknown mailbox reduce mode: {reduce!r}")
```

### 2. `project_append_rows()` 方法

同样移除了时间戳检查逻辑，保持与MemShare一致的行为。

## 修改的文件

- `src/atc_starrygl_lib/memory/mailbox.py`
  - 修改了 `append_rows()` 方法（第57-82行）
  - 修改了 `project_append_rows()` 方法（第84-110行）

## 验证步骤

1. 重新编译项目（如果有C++组件）：
   ```bash
   cd build && make -j
   ```

2. 运行训练脚本，验证精度是否与MemShare一致：
   ```bash
   # 使用相同的配置和数据集
   python tools/wiki_ctdg_edge_predict_smoke.py
   ```

3. 对比关键指标：
   - Training loss
   - Validation AP
   - Test AP
   - Test AUC

## 预期结果

修复后，ATC_StarryglLib应该能够复现与MemShare-public相同的精度，因为：
1. Mailbox消息构建逻辑已经一致（`[mem_src, mem_dst, edge_feat]`）
2. Memory updater实现一致（GRU + time encoding）
3. Mailbox更新逻辑现在也一致（循环队列，不检查时间戳）

## 其他已验证的一致性

✅ Mailbox消息构建：两个项目都使用 `[mem_src, mem_dst, edge_feat]` 拼接
✅ Memory updater：GRU/RNN/Transformer实现一致
✅ Time encoding：使用相同的时间编码方式
✅ combine_node_feature：特征组合逻辑一致
✅ _latest_by_row vs scatter_max：行为完全一致

## 注意事项

这个修改使得mailbox允许旧消息覆盖新消息，这在分布式训练中可能由于消息到达顺序不确定而发生。虽然这看起来不太合理，但这正是MemShare的行为，为了复现精度必须保持一致。

如果未来需要改进这个行为，应该同时修改两个项目，并重新验证精度。
