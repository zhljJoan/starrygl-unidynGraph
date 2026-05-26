#!/usr/bin/env python3
"""
验证边特征是否被正确加载
"""

import sys
sys.path.insert(0, '/home/zlj/ATC_StarryglLib/src')

from atc_starrygl_lib.preprocess.dataset import build_dataset

print("=" * 60)
print("验证边特征加载")
print("=" * 60)

# 使用你的配置加载数据
dataset = build_dataset(
    data="/mnt/data/zlj/starrygl-data/raw/TGL-DATA/WIKI",
    mode="event",
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=600,
    random_node_feat_dim=172,
    random_node_feat_seed=0,
)

print(f"\n数据集信息:")
print(f"  num_nodes: {dataset['num_nodes']}")
print(f"  num_edges: {len(dataset['src'])}")

if dataset.get('node_feat') is not None:
    print(f"  node_feat: {dataset['node_feat'].shape}")
else:
    print(f"  node_feat: None")

if dataset.get('edge_feat') is not None:
    print(f"  edge_feat: {dataset['edge_feat'].shape}")
    print(f"    mean: {dataset['edge_feat'].mean():.6f}")
    print(f"    std: {dataset['edge_feat'].std():.6f}")
    print(f"    前5条边的前5维:")
    print(dataset['edge_feat'][:5, :5])
else:
    print(f"  edge_feat: None")
    print("\n🔴 严重问题：edge_feat没有被加载！")
    print("这会导致精度大幅下降！")

print("\n" + "=" * 60)
