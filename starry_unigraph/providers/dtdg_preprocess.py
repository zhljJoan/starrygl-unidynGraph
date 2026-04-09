from __future__ import annotations

from pathlib import Path
from typing import Any

from starry_unigraph.core import DTDGPartitionBook, SnapshotRoutePlan
from starry_unigraph.data import build_snapshot_dataset_from_events, load_raw_temporal_events
from starry_unigraph.preprocess import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.providers.common import ARTIFACT_VERSION
from starry_unigraph.types import PreparedArtifacts, SessionContext

from .dtdg_data import (
    build_dtdg_partitions,
    build_flare_partition_data_list,
    build_partition_input_from_raw_dataset,
)
from .dtdg_common import dtdg_pipeline


class BaseDTDGPreprocessor(GraphPreprocessor):
    graph_mode = "dtdg"
    artifact_dirs = ("meta", "partitions", "routes", "snapshots", "flare", "clusters")

    def prepare_raw(self, session_ctx: SessionContext) -> None:
        # config["data"]["root"]：原始数据集根目录（可被 session_ctx.dataset_path 覆盖）
        # config["data"]["name"]：数据集名称，用于定位具体文件
        # config["train"]["snaps"]：将事件流切分成的快照总数
        dataset_root = (
            session_ctx.dataset_path
            if session_ctx.dataset_path is not None
            else Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        )
        dataset_root.mkdir(parents=True, exist_ok=True)

        dataset_name = session_ctx.config["data"]["name"]
        snaps = int(session_ctx.config["train"]["snaps"])
        raw_events = load_raw_temporal_events(root=dataset_root, dataset_name=dataset_name, config=session_ctx.config)
        raw_dataset = build_snapshot_dataset_from_events(events=raw_events, snaps=snaps)

        session_ctx.provider_state["raw_dataset"] = raw_dataset
        session_ctx.provider_state["raw_stats"] = {
            "num_nodes": raw_dataset["num_nodes"],
            "num_edges": raw_dataset["num_edges"],
            "num_snapshots": raw_dataset["num_snapshots"],
        }

    def build_partitions(self, session_ctx: SessionContext) -> None:
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        # config["dist"]["world_size"]：分布式进程数，决定分区数量
        # config["graph"]["partition"]：分区算法（如 "metis"、"random"）
        num_parts = int(session_ctx.config["dist"]["world_size"])
        algo = str(session_ctx.config["graph"]["partition"])
        partition_input = build_partition_input_from_raw_dataset(raw_dataset)
        partition_result = build_dtdg_partitions(
            graph_data=partition_input,
            num_parts=num_parts,
            algo=algo,
            config=session_ctx.config,
        )
        session_ctx.provider_state["partition_result"] = partition_result
        session_ctx.provider_state["partition_manifest"] = {
            "num_parts": num_parts,
            "partition_algo": algo,
            "num_nodes": raw_dataset["num_nodes"],
            "num_edges": raw_dataset["num_edges"],
            "num_snapshots": raw_dataset["num_snapshots"],
        }

    def _partition_book(self, session_ctx: SessionContext) -> DTDGPartitionBook:
        return DTDGPartitionBook(
            # config["dist"]["world_size"]：分区数
            num_parts=session_ctx.config["dist"]["world_size"],
            # config["graph"]["partition"]：分区算法
            partition_algo=session_ctx.config["graph"]["partition"],
            # config["train"]["snaps"]：快照总数
            snapshot_count=session_ctx.config["train"]["snaps"],
        )

    def _route_plan(self, session_ctx: SessionContext) -> SnapshotRoutePlan:
        return SnapshotRoutePlan(
            # config["graph"]["route"]：路由类型（如 "full"、"sampled"）
            route_type=session_ctx.config["graph"]["route"],
            # config["runtime"]["cache"]：缓存策略（如 "none"、"lru"）
            cache_policy=session_ctx.config["runtime"]["cache"],
        )

    def _provider_meta(self, session_ctx: SessionContext, route_plan: SnapshotRoutePlan) -> dict[str, Any]:
        partition_book = self._partition_book(session_ctx)
        return {
            "graph_mode": self.graph_mode,
            **partition_book.describe(),
            # config["model"]["hidden_dim"]：节点特征维度
            "feature_dim": session_ctx.config["model"]["hidden_dim"],
            "edge_feat_dim": 0,
            "label_dim": 1,
            # config["model"]["task"]：任务类型（如 "link_prediction"、"node_classification"）
            "task_type": session_ctx.config["model"]["task"],
            "artifact_version": ARTIFACT_VERSION,
            "snapshot_route_plan": route_plan.describe(),
            # config["dtdg"]["pipeline"]：pipeline 类型（"flare_native" 或 "chunked"）
            "pipeline": dtdg_pipeline(session_ctx),
        }


class FlareDTDGPreprocessor(BaseDTDGPreprocessor):
    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        partition_result = session_ctx.provider_state["partition_result"]
        route_plan = self._route_plan(session_ctx)
        partition_data_list = build_flare_partition_data_list(
            raw_dataset=raw_dataset,
            partition_result=partition_result,
            route_plan=route_plan,
            config=session_ctx.config,
        )

        outputs: list[ArtifactOutput] = [
            ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
            ArtifactOutput("routes/manifest.json", route_plan.describe()),
            ArtifactOutput(
                "flare/manifest.json",
                {
                    "pipeline": "flare_native",
                    "num_parts": len(partition_data_list),
                    "files": [f"part_{part_id:03d}.pth" for part_id in range(len(partition_data_list))],
                },
            ),
        ]
        for part_id, partition_data in enumerate(partition_data_list):
            outputs.append(ArtifactOutput(f"flare/part_{part_id:03d}.pth", partition_data, serializer="torch"))

        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(provider_meta=self._provider_meta(session_ctx, route_plan), outputs=outputs),
        )


class ChunkedDTDGPreprocessor(BaseDTDGPreprocessor):
    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        partition_book = self._partition_book(session_ctx)
        route_plan = self._route_plan(session_ctx)
        num_parts = partition_book.num_parts
        snapshot_count = partition_book.snapshot_count
        # config["preprocess"]["cluster"]["num_per_partition"]：每个分区内的聚类数
        cluster_per_partition = int(session_ctx.config["preprocess"]["cluster"]["num_per_partition"])
        # config["preprocess"]["chunk"]["window_multiple"]：chunk 跨度 = window_multiple × window_size
        window_multiple = int(session_ctx.config["preprocess"]["chunk"]["window_multiple"])
        # config["model"]["window"]["size"]：时序窗口大小（快照数）
        window_size = int(session_ctx.config["model"]["window"]["size"])
        chunk_span = max(1, window_multiple * window_size)

        outputs: list[ArtifactOutput] = [
            ArtifactOutput("partitions/manifest.json", partition_book.describe()),
            ArtifactOutput("routes/manifest.json", route_plan.describe()),
        ]
        snapshot_manifest: list[dict[str, Any]] = []

        for part_id in range(num_parts):
            cluster_manifest = []
            cluster_snapshots = max(1, snapshot_count // cluster_per_partition)
            for cluster_id in range(cluster_per_partition):
                cluster_start = cluster_id * cluster_snapshots
                cluster_end = snapshot_count if cluster_id == cluster_per_partition - 1 else min(
                    snapshot_count, cluster_start + cluster_snapshots
                )
                chunks = []
                chunk_id = 0
                for chunk_start in range(cluster_start, cluster_end, chunk_span):
                    chunk_end = min(cluster_end, chunk_start + chunk_span)
                    chunk_payload = {
                        "partition_id": part_id,
                        "cluster_id": cluster_id,
                        "chunk_id": chunk_id,
                        "snapshot_begin": chunk_start,
                        "snapshot_end": chunk_end,
                        "window_count": chunk_end - chunk_start,
                        "route_plan": route_plan.describe(),
                    }
                    chunk_path = f"snapshots/part_{part_id:03d}/cluster_{cluster_id:03d}/chunk_{chunk_id:03d}.pth"
                    outputs.append(ArtifactOutput(chunk_path, chunk_payload, serializer="torch"))
                    chunks.append(chunk_payload | {"data_path": chunk_path})
                    snapshot_manifest.append(chunk_payload | {"data_path": chunk_path})
                    chunk_id += 1
                cluster_manifest.append(
                    {
                        "partition_id": part_id,
                        "cluster_id": cluster_id,
                        "snapshot_begin": cluster_start,
                        "snapshot_end": cluster_end,
                        "chunks": chunks,
                    }
                )
            outputs.append(
                ArtifactOutput(
                    f"clusters/part_{part_id:03d}/cluster_manifest.json",
                    {"pipeline": "chunked", "partition_id": part_id, "clusters": cluster_manifest},
                )
            )

        outputs.append(
            ArtifactOutput(
                "snapshots/manifest.json",
                {"pipeline": "chunked", "chunk_count": len(snapshot_manifest), "chunks": snapshot_manifest},
            )
        )
        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(provider_meta=self._provider_meta(session_ctx, route_plan), outputs=outputs),
        )
