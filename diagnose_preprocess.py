#!/usr/bin/env python3
"""
快速诊断预处理问题
运行此脚本检查节点ID范围和数据分割
"""

import pandas as pd
import torch
import json
from pathlib import Path

print("=" * 60)
print("预处理问题诊断")
print("=" * 60)

# 1. 检查原始数据
print("\n[1] 原始数据检查")
data_path = Path("/mnt/data/zlj/starrygl-data/raw/TGL-DATA/WIKI/edges.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"✓ 找到数据文件: {data_path}")
    print(f"  总边数: {len(df)}")
    print(f"  src范围: {df['src'].min():.0f} - {df['src'].max():.0f}")
    print(f"  dst范围: {df['dst'].min():.0f} - {df['dst'].max():.0f}")

    all_nodes = pd.concat([df['src'], df['dst']]).unique()
    print(f"  唯一节点数: {len(all_nodes)}")
    print(f"  节点ID最小值: {all_nodes.min():.0f}")
    print(f"  节点ID最大值: {all_nodes.max():.0f}")

    if all_nodes.min() == 1:
        print("  ⚠️  节点ID从1开始，不是0！")
    else:
        print("  ✓ 节点ID从0开始")

    # 检查数据分割
    if 'ext_roll' in df.columns:
        print(f"\n  数据分割 (ext_roll):")
        split_counts = df['ext_roll'].value_counts().sort_index()
        for split_id, count in split_counts.items():
            split_name = ['train', 'val', 'test'][int(split_id)]
            print(f"    {split_name} ({split_id}): {count} 条边")
    elif 'int_roll' in df.columns:
        print(f"\n  数据分割 (int_roll):")
        split_counts = df['int_roll'].value_counts().sort_index()
        for split_id, count in split_counts.items():
            split_name = ['train', 'val', 'test'][int(split_id)]
            print(f"    {split_name} ({split_id}): {count} 条边")
else:
    print(f"✗ 未找到数据文件: {data_path}")

# 2. 检查配置
print("\n[2] 配置检查")
config_path = Path("/home/zlj/ATC_StarryglLib/configs/ctdg_wiki_tgn_speed_hot01_bs600.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)

    print(f"✓ 找到配置文件: {config_path}")
    print(f"  数据源: {config['graph']['source']}")
    print(f"  random_node_feat_dim: {config['graph']['random_node_feat_dim']}")

    if 'num_nodes' in config.get('graph', {}):
        print(f"  num_nodes: {config['graph']['num_nodes']}")
    else:
        print(f"  num_nodes: 未配置（自动推断）")

    # 检查预处理配置
    preprocess = config.get('preprocess', {})
    print(f"\n  预处理配置:")
    print(f"    use_new_pipeline: {preprocess.get('use_new_pipeline', False)}")
    print(f"    train_ratio: {preprocess.get('train_ratio', 0.7)}")
    print(f"    val_ratio: {preprocess.get('val_ratio', 0.15)}")
else:
    print(f"✗ 未找到配置文件: {config_path}")

# 3. 预期的正确处理
print("\n[3] 正确的预处理应该是:")
if data_path.exists():
    print(f"  选项A: 将节点ID减1 (变成0-based)")
    print(f"    src: 0 - {df['src'].max() - 1:.0f}")
    print(f"    dst: {df['dst'].min() - 1:.0f} - {df['dst'].max() - 1:.0f}")
    print(f"    num_nodes: {len(all_nodes)}")

    print(f"\n  选项B: 保持节点ID不变，但num_nodes要多1")
    print(f"    src: {df['src'].min():.0f} - {df['src'].max():.0f}")
    print(f"    dst: {df['dst'].min():.0f} - {df['dst'].max():.0f}")
    print(f"    num_nodes: {int(all_nodes.max()) + 1}")
    print(f"    (注意: 节点0会浪费)")

# 4. 检查预处理后的数据
print("\n[4] 预处理后的数据检查")
artifact_dirs = [
    Path("/home/zlj/ATC_StarryglLib/.artifacts"),
    Path("/home/zlj/ATC_StarryglLib/artifacts"),
]

found_artifacts = False
for artifact_dir in artifact_dirs:
    if artifact_dir.exists():
        print(f"✓ 找到artifacts目录: {artifact_dir}")
        found_artifacts = True

        # 查找预处理后的文件
        for file in artifact_dir.rglob("*.pt"):
            print(f"  - {file.relative_to(artifact_dir)}")
        for file in artifact_dir.rglob("*.pth"):
            print(f"  - {file.relative_to(artifact_dir)}")
        break

if not found_artifacts:
    print("✗ 未找到artifacts目录")
    print("  可能还没有运行过预处理")

# 5. 诊断建议
print("\n" + "=" * 60)
print("诊断建议")
print("=" * 60)

if data_path.exists() and all_nodes.min() == 1:
    print("\n⚠️  检测到节点ID从1开始！")
    print("\n这很可能是精度低的原因。请检查:")
    print("\n1. 在训练代码中添加调试输出:")
    print("   ```python")
    print("   # 在第一个batch")
    print("   if batch_idx == 0:")
    print("       print(f'src: {batch.src.min()} - {batch.src.max()}')")
    print("       print(f'dst: {batch.dst.min()} - {batch.dst.max()}')")
    print("       print(f'num_nodes: {num_nodes}')")
    print("       print(f'memory.shape: {memory.shape}')")
    print("   ```")
    print("\n2. 如果src.min()=1，说明节点ID没有减1")
    print("   需要修改预处理代码，将节点ID减1")
    print("\n3. 如果src.min()=0，说明已经减1了")
    print("   那么问题可能在其他地方")

print("\n" + "=" * 60)
