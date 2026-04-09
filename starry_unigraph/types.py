from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DistributedContext:
    backend: str
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    local_world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    init_method: str = "env://"
    launcher: str = "single_process"
    initialized: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


@dataclass
class PredictionResult:
    predictions: list[Any]
    targets: list[Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionContext:
    config: dict[str, Any]
    project_root: Path
    dataset_path: Path | None
    artifact_root: Path
    dist: DistributedContext
    warnings: list[str] = field(default_factory=list)
    provider_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeBundle:
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedArtifacts:
    # 指向 <artifact_root>/meta/artifacts.json，由 GraphPreprocessor.emit_artifacts() 写入
    meta_path: Path
    # 已存在的子目录映射，键为目录名，值为绝对路径。
    # 可能包含的目录：
    #   "meta"       — artifacts.json 所在目录
    #   "partitions" — 分区 manifest（partitions/manifest.json）
    #   "routes"     — 路由 manifest（routes/manifest.json）
    #   "sampling"   — 采样索引（sampling/index.json）
    #   "snapshots"  — 快照文件（chunked pipeline）
    #   "flare"      — flare 分区文件（flare/manifest.json + flare/part_NNN.pth）
    #   "clusters"   — 聚类文件（chunked pipeline）
    directories: dict[str, Path]
    # artifacts.json 的内容，由各 preprocessor 写入，公共字段包括：
    #   "graph_mode"          — "dtdg" 或 "ctdg"
    #   "artifact_version"    — int，用于版本校验（当前为 1）
    #   "num_parts"           — 分布式分区数
    # DTDG 额外字段：
    #   "partition_algo"      — 分区算法名称
    #   "snapshot_count"      — 快照总数
    #   "feature_dim"         — 节点特征维度
    #   "edge_feat_dim"       — 边特征维度（通常为 0）
    #   "label_dim"           — 标签维度
    #   "task_type"           — 任务类型（如 "link_prediction"）
    #   "pipeline"            — "flare_native" 或 "chunked"
    #   "snapshot_route_plan" — {"route_type": ..., "cache_policy": ...}
    # CTDG 额外字段：
    #   "num_nodes"           — 节点数
    #   "num_edges"           — 边数
    #   "feature_route_plan"  — {"route_type": ..., "world_size": ...}
    provider_meta: dict[str, Any]
