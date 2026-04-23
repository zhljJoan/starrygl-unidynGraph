"""DTDG preprocessing pipeline: raw events -> partitions -> Flare artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starry_unigraph.data import build_snapshot_dataset_from_events, load_raw_temporal_events
from starry_unigraph.preprocess.base import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.types import PreparedArtifacts, SessionContext

from .types import DTDGPartitionBook, SnapshotRoutePlan
from .dtdg_prepare import (
    build_dtdg_partitions,
    build_flare_partition_data_list,
)

ARTIFACT_VERSION = 1


def read_artifact_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_artifacts(
    prepared: PreparedArtifacts,
    expected_graph_mode: str,
    expected_num_parts: int | None = None,
) -> dict[str, Any]:
    if not prepared.meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts metadata: {prepared.meta_path}")
    meta = read_artifact_meta(prepared.meta_path)
    if meta.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError(
            f"Artifact version mismatch: expected {ARTIFACT_VERSION}, got {meta.get('artifact_version')}"
        )
    if meta.get("graph_mode") != expected_graph_mode:
        raise RuntimeError(
            f"Artifact graph_mode mismatch: expected {expected_graph_mode}, got {meta.get('graph_mode')}"
        )
    if expected_num_parts is not None and meta.get("num_parts") != expected_num_parts:
        raise RuntimeError(
            f"Artifact num_parts mismatch: expected {expected_num_parts}, got {meta.get('num_parts')}"
        )
    partitions = prepared.directories["partitions"] / "manifest.json"
    routes = prepared.directories["routes"] / "manifest.json"
    if not partitions.exists():
        raise FileNotFoundError(f"Missing partition manifest: {partitions}")
    if not routes.exists():
        raise FileNotFoundError(f"Missing route manifest: {routes}")
    return meta


def load_prepared_from_disk(artifact_root: Path) -> PreparedArtifacts:
    meta_path = artifact_root / "meta" / "artifacts.json"
    if not meta_path.exists():
        raise RuntimeError("Data not prepared. Call prepare_data() first.")
    provider_meta = read_artifact_meta(meta_path)
    known_dirs = ("meta", "partitions", "routes", "sampling", "snapshots", "flare", "clusters")
    directories = {
        name: artifact_root / name
        for name in known_dirs
        if (artifact_root / name).exists()
    }
    return PreparedArtifacts(meta_path=meta_path, directories=directories, provider_meta=provider_meta)


class BaseDTDGPreprocessor(GraphPreprocessor):
    graph_mode = "dtdg"
    artifact_dirs = ("meta", "partitions", "routes", "snapshots", "flare")

    def prepare_raw(self, session_ctx: SessionContext) -> None:
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
            "node_feat_dim": raw_dataset.get("node_feat_dim", 0),
            "node_feature_source": raw_dataset.get("node_feature_source", "unknown"),
        }

    def build_partitions(self, session_ctx: SessionContext) -> None:
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        num_parts = int(session_ctx.config["dist"]["world_size"])
        algo = str(session_ctx.config["graph"]["partition"])
        partition_result = build_dtdg_partitions(
            graph_data=raw_dataset,
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
            num_parts=session_ctx.config["dist"]["world_size"],
            partition_algo=session_ctx.config["graph"]["partition"],
            snapshot_count=session_ctx.config["train"]["snaps"],
        )

    def _route_plan(self, session_ctx: SessionContext) -> SnapshotRoutePlan:
        return SnapshotRoutePlan(
            route_type=session_ctx.config["graph"]["route"],
            cache_policy=session_ctx.config["runtime"]["cache"],
        )

    def _provider_meta(self, session_ctx: SessionContext, route_plan: SnapshotRoutePlan) -> dict[str, Any]:
        partition_book = self._partition_book(session_ctx)
        return {
            "graph_mode": self.graph_mode,
            **partition_book.describe(),
            "feature_dim": session_ctx.provider_state["raw_stats"].get("node_feat_dim", 0),
            "edge_feat_dim": 0,
            "label_dim": 1,
            "task_type": session_ctx.config["model"]["task"],
            "artifact_version": ARTIFACT_VERSION,
            "snapshot_route_plan": route_plan.describe(),
            "pipeline": "flare_native",
            "node_feature_source": session_ctx.provider_state["raw_stats"].get("node_feature_source", "unknown"),
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
