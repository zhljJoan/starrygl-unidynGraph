# 精度差异分析报告

## 对比项目
- **当前项目**: ATC_StarryglLib
- **参考项目**: ~/MemShare-public/MemShare

## 已发现的关键差异

### 1. Mailbox消息构建逻辑 ✓ (一致)
两个项目的mailbox消息构建逻辑相同：
- **MemShare**: `[mem_src, mem_dst, edge_feat]` 和 `[mem_dst, mem_src, edge_feat]`
- **ATC_StarryglLib**: `_build_mailbox_messages()` 实现相同逻辑
- 消息维度: `2*memory_dim + edge_feat_dim = 2*100 + 172 = 372` ✓

### 2. Memory Updater实现 ✓ (一致)
GRU Memory Updater的核心逻辑一致：
```python
# 两个项目都使用相同的更新方式
time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
```

### 3. combine_node_feature逻辑 ✓ (一致)
两个项目都支持combine_node_feature：
```python
if self.memory_param.get('combine_node_feature'):
    if self.dim_node_feat > 0:
        if self.dim_node_feat == self.dim_hid:
            b.srcdata['h'] += updated_memory  # 或 b.srcdata['h'] = b.srcdata['h'] + updated_memory
```

### 4. 需要进一步检查的差异点

#### 4.1 Mailbox的reduce操作
- **MemShare**: 使用 `torch_scatter.scatter_max` 按timestamp选择最新的消息
  ```python
  unq_index, inv = torch.unique(index, return_inverse=True)
  max_ts, idx = torch_scatter.scatter_max(mail_ts, inv, 0)
  mail = mail[idx]
  ```
- **ATC_StarryglLib**: 使用 `_latest_by_row()` 函数
  - 需要验证这个函数是否与scatter_max行为完全一致

#### 4.2 Mailbox的append_rows逻辑
- **MemShare**: 直接使用循环索引更新mailbox
  ```python
  pos = self.next_mail_pos[index] % mailbox_size
  self.mailbox[index, pos] = mail
  self.mailbox_ts[index, pos] = mail_ts
  self.next_mail_pos[index] = (pos + 1) % mailbox_size
  ```
- **ATC_StarryglLib**: 使用 `append_rows()` 方法，包含 `max_ts` reduce
  - 需要验证reduce="max_ts"的行为是否正确

#### 4.3 Memory/Mailbox的初始化
- 需要检查两个项目中memory和mailbox的初始化是否一致
- 特别是dtype (float32 vs float64)

#### 4.4 训练流程差异
- **MemShare**: 使用DGL的MFG (Message Flow Graph)
- **ATC_StarryglLib**: 使用自定义的采样输出和batch结构
- 需要验证数据流是否完全对齐

#### 4.5 时间戳处理
- 需要检查时间戳的精度和范围是否一致
- 检查时间编码(TimeEncode)的实现是否完全相同

## 🔴 已发现的关键问题

### **Mailbox更新逻辑的根本差异**

**MemShare的行为**:
```python
def set_mailbox_local(self, index, source, source_ts, Reduce_Op=None):
    if Reduce_Op == 'max' and self.num_parts > 1:
        # 只在同一批次内对重复节点取最大时间戳
        unq_id, inv = index.unique(return_inverse=True)
        max_ts, id = torch_scatter.scatter_max(source_ts, inv, dim=0)
        source_ts = max_ts
        source = source[id]
        index = unq_id
    
    # 直接覆盖，不检查是否比已有消息更新
    self.mailbox_ts.accessor.data[index, self.next_mail_pos[index]] = source_ts
    self.mailbox.accessor.data[index, self.next_mail_pos[index]] = source
```

**ATC_StarryglLib的行为**:
```python
def append_rows(self, rows, msg, ts, *, reduce="max_ts"):
    # 先对同一批次内的重复节点取最新
    if reduce == "max_ts" and row.numel() > 1:
        row, message, ts_in = _latest_by_row(row, message, ts_in)
    
    # 🔴 关键差异：检查是否比mailbox中已有的最旧消息更新
    if reduce == "max_ts":
        oldest = self.mailbox_ts[row].min(dim=1).values
        keep = ts_in > oldest  # 只保留比已有消息更新的
        if not keep.any():
            return  # 如果都是旧消息，直接返回
        row = row[keep]
        message = message[keep]
        ts_in = ts_in[keep]
    
    # 写入mailbox
    pos = self.next_pos[row] % k
    self.mailbox[row, pos] = message
    self.mailbox_ts[row, pos] = ts_in
```

**影响**:
- MemShare: 允许旧消息覆盖新消息（循环队列，不检查时间戳）
- ATC_StarryglLib: 拒绝旧消息，只接受更新的消息

**示例场景**:
```
节点0的mailbox已有: ts=2.0的消息
现在收到: ts=1.5的旧消息

MemShare结果: mailbox[0] = ts=1.5 (旧消息覆盖了新消息)
ATC_StarryglLib结果: mailbox[0] = ts=2.0 (拒绝旧消息)
```

这会导致训练过程中memory updater读取到不同的mailbox内容，从而产生不同的更新结果，最终影响模型精度。

## 解决方案

### 方案1: 移除max_ts检查（匹配MemShare行为）

修改 `src/atc_starrygl_lib/memory/mailbox.py` 的 `append_rows` 方法，移除时间戳检查：

```python
def append_rows(self, rows: Tensor, msg: Tensor, ts: Tensor, *, reduce: str = "max_ts") -> None:
    row = rows.long().to(self.mailbox.device)
    message = msg.to(device=self.mailbox.device, dtype=self.mailbox.dtype)
    ts_in = ts.to(device=self.mailbox_ts.device, dtype=self.mailbox_ts.dtype).reshape(-1)
    
    # 只对同一批次内的重复节点去重
    if reduce == "max_ts" and row.numel() > 1:
        row, message, ts_in = _latest_by_row(row, message, ts_in)
    
    # 🔴 移除这段检查，直接写入
    # if reduce == "max_ts":
    #     oldest = self.mailbox_ts[row].min(dim=1).values
    #     keep = ts_in > oldest
    #     if not keep.any():
    #         return
    #     row = row[keep]
    #     message = message[keep]
    #     ts_in = ts_in[keep]
    
    k = int(self.mailbox.size(1))
    pos = self.next_pos[row] % k
    self.mailbox[row, pos] = message
    self.mailbox_ts[row, pos] = ts_in
    self.next_pos[row] = (pos + 1) % k
```

### 方案2: 添加配置选项

添加一个配置项 `mailbox_check_timestamp`，允许用户选择行为：
- `False`: 匹配MemShare（不检查时间戳）
- `True`: 保持当前行为（检查时间戳）

## 下一步调查方向

1. ✅ **对比_latest_by_row与scatter_max的行为** - 已验证一致
2. ✅ **检查mailbox append时的max_ts过滤逻辑** - 已发现关键差异
3. **验证memory/mailbox的dtype一致性**
4. **对比时间编码实现**
5. **检查训练时的数据预处理和batch构建**
6. **验证负采样策略是否一致**

## 建议的调试方法

1. 在两个项目中打印相同batch的中间结果：
   - 采样后的节点ID
   - Memory读取的值
   - Mailbox读取的值
   - Memory更新后的值
   - Loss值

2. 逐层对比：
   - 第一个batch的第一层输出
   - 确保每一步的数值完全一致

3. 检查配置文件：
   - 确保所有超参数完全相同
   - 特别是learning rate, dropout, batch_size等
