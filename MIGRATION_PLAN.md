# CTDG/DTDG 迁移到 backends 的完整计划

## 目标结构

```
starry_unigraph/
├── backends/
│   ├── ctdg/
│   │   ├── __init__.py
│   │   ├── preprocess.py (from preprocess/ctdg.py)
│   │   └── runtime/
│   │       └── (from runtime/online/)
│   ├── dtdg/
│   │   ├── __init__.py
│   │   ├── preprocess.py (from preprocess/dtdg.py)
│   │   ├── dtdg_prepare.py (from preprocess/dtdg_prepare.py)
│   │   └── runtime/
│   │       └── (from runtime/flare/)
│   └── chunk/
├── runtime/
│   ├── chunk/ (NEW - kept independent)
│   ├── modules/
│   ├── route/
│   ├── store/
│   └── (other base utilities)
└── preprocess/
    ├── chunk.py (NEW - kept independent)
    ├── base.py
    └── __init__.py
```

## 迁移步骤

### Phase 1: 准备 backends 目录结构

```bash
# 创建 CTDG backend 完整结构
mkdir -p starry_unigraph/backends/ctdg/runtime
mkdir -p starry_unigraph/backends/dtdg/runtime

# 复制 online/ 到 backends/ctdg/runtime
cp -r starry_unigraph/runtime/online/* starry_unigraph/backends/ctdg/runtime/
cp -r starry_unigraph/runtime/online/__init__.py starry_unigraph/backends/ctdg/runtime/

# 复制 flare/ 到 backends/dtdg/runtime
cp -r starry_unigraph/runtime/flare/* starry_unigraph/backends/dtdg/runtime/
cp -r starry_unigraph/runtime/flare/__init__.py starry_unigraph/backends/dtdg/runtime/
```

### Phase 2: 迁移预处理器

```bash
# 复制预处理器到 backends
cp starry_unigraph/preprocess/ctdg.py starry_unigraph/backends/ctdg/preprocess.py
cp starry_unigraph/preprocess/dtdg.py starry_unigraph/backends/dtdg/preprocess.py
cp starry_unigraph/preprocess/dtdg_prepare.py starry_unigraph/backends/dtdg/dtdg_prepare.py
```

### Phase 3: 创建 backends 导出接口

**starry_unigraph/backends/ctdg/__init__.py**
```python
from .preprocess import CTDGPreprocessor
from .runtime.session import CTDGSession
from .runtime import (
    CTDGOnlineRuntime,
    # ... other exports
)
__all__ = ["CTDGPreprocessor", "CTDGSession", "CTDGOnlineRuntime"]
```

**starry_unigraph/backends/dtdg/__init__.py**
```python
from .preprocess import FlareDTDGPreprocessor
from .dtdg_prepare import (
    build_dtdg_partitions,
    build_flare_partition_data_list,
    # ... other exports
)
from .runtime.session_loader import FlareRuntimeLoader
from .runtime import (
    build_flare_model,
    # ... other exports
)
__all__ = ["FlareDTDGPreprocessor", "FlareRuntimeLoader", "build_flare_model"]
```

### Phase 4: 更新导入路径

所有引用需要更新的地方：

1. **starry_unigraph/session.py**
   - `from starry_unigraph.runtime.online import CTDGSession`
     → `from starry_unigraph.backends.ctdg.runtime.session import CTDGSession`
   - `from starry_unigraph.runtime.flare import ...`
     → `from starry_unigraph.backends.dtdg.runtime import ...`
   - `from starry_unigraph.preprocess.dtdg import ...`
     → `from starry_unigraph.backends.dtdg.preprocess import ...`

2. **backends/ctdg/preprocess.py** (内部更新)
   - `from starry_unigraph.runtime.online.data import TGTemporalDataset`
     → `from starry_unigraph.backends.ctdg.runtime.data import TGTemporalDataset`
   - `from starry_unigraph.runtime.online.route import CTDGFeatureRoute`
     → `from starry_unigraph.backends.ctdg.runtime.route import CTDGFeatureRoute`

3. **backends/dtdg/preprocess.py** (内部更新)
   - `from starry_unigraph.preprocess.dtdg_prepare import ...`
     → `from .dtdg_prepare import ...`
   - `from starry_unigraph.runtime.flare import ...`
     → `from starry_unigraph.backends.dtdg.runtime import ...`

4. **backends/dtdg/runtime/session_loader.py** (内部更新)
   - `from starry_unigraph.data.partition import PartitionData`
     → 保持不变（在 starry_unigraph/data 下）

5. **tests/**
   - 更新所有 `from starry_unigraph.runtime.online/flare` 为 `from starry_unigraph.backends.ctdg/dtdg.runtime`

### Phase 5: 清理旧目录

```bash
# 删除已迁移的文件
rm starry_unigraph/preprocess/ctdg.py
rm starry_unigraph/preprocess/dtdg.py
rm starry_unigraph/preprocess/dtdg_prepare.py

# 删除旧的运行时目录（注意先确认已完全复制）
rm -rf starry_unigraph/runtime/online/
rm -rf starry_unigraph/runtime/flare/
```

### Phase 6: 验证

```bash
# 运行 CTDG 测试
python -m pytest tests/test_session.py::test_build_runtime_ctdg_and_dtdg -v

# 运行 DTDG 测试
python -m pytest tests/test_session.py::test_build_runtime_dtdg -v

# 运行 chunk 测试
python -m pytest tests/test_session.py::test_build_runtime_chunked_dtdg -v
```

## 关键点

1. **导入路径更新是关键** — 需要确保所有 import 都正确
2. **backends 内部相对导入** — 使用相对导入 (from . or from ..) 减少依赖
3. **保持 runtime/chunk 独立** — 不要让 chunk 依赖 CTDG/DTDG 代码
4. **session.py 作为统一入口** — 只需更新导入，分发逻辑不变

## 快速检查清单

- [ ] backends/ctdg/ 完整复制和重组
- [ ] backends/dtdg/ 完整复制和重组
- [ ] 所有 import 路径更新
- [ ] session.py 导入正确
- [ ] CTDG 测试通过
- [ ] DTDG 测试通过
- [ ] Chunk 测试通过
- [ ] 旧目录已删除

## 问题排查

### ImportError: cannot import name XXX from starry_unigraph.runtime.online
→ 检查 backends/ctdg/runtime/__init__.py 是否正确导出

### ModuleNotFoundError: No module named 'starry_unigraph.runtime.online'
→ 检查 session.py 和其他文件的导入路径是否已更新

### Circular import
→ 检查是否有循环依赖，特别是 backends 内部的相对导入
