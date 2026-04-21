"""CTDG preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starry_unigraph.preprocess.base import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.types import PreparedArtifacts, SessionContext

ARTIFACT_VERSION = 1


class CTDGPreprocessor(GraphPreprocessor):
    graph_mode = "ctdg"

    def prepare_raw(self, session_ctx: SessionContext) -> None:
        from .runtime.data import TGTemporalDataset
        dataset_root = session_ctx.dataset_path or Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        dataset = TGTemporalDataset(
            dataset_root,
            session_ctx.config["data"]["name"],
            split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
            config=session_ctx.config,
        )
        session_ctx.provider_state["ctdg_dataset_stats"] = dataset.describe()

    def build_partitions(self, session_ctx: SessionContext) -> None:
        stats = session_ctx.provider_state["ctdg_dataset_stats"]
        session_ctx.provider_state["partition_manifest"] = {
            "graph_mode": "ctdg",
            "num_parts": int(session_ctx.config["dist"]["world_size"]),
            "partition_algo": str(session_ctx.config["graph"]["partition"]),
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
        }

    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        from .runtime.route import CTDGFeatureRoute
        stats = session_ctx.provider_state["ctdg_dataset_stats"]
        feature_route = CTDGFeatureRoute(
            route_type=str(session_ctx.config["graph"]["route"]),
            world_size=int(session_ctx.config["dist"]["world_size"]),
        )
        provider_meta = {
            "graph_mode": self.graph_mode,
            "artifact_version": ARTIFACT_VERSION,
            "num_parts": int(session_ctx.config["dist"]["world_size"]),
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
            "feature_dim": stats["edge_feat_dim"],
            "task_type": session_ctx.config["model"]["task"],
            "feature_route_plan": feature_route.describe(),
        }
        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(
                provider_meta=provider_meta,
                outputs=[
                    ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
                    ArtifactOutput("routes/manifest.json", feature_route.describe()),
                    ArtifactOutput("sampling/index.json", {"dataset": session_ctx.config["data"]["name"], **stats}),
                ],
            ),
        )
