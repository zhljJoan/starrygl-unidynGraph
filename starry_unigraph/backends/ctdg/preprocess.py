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
        from .runtime.data import TGTemporalDataset
        from .preprocess_partition import speed_partition

        stats = session_ctx.provider_state["ctdg_dataset_stats"]
        partition_algo = str(session_ctx.config["graph"]["partition"])
        num_parts = int(session_ctx.config["dist"]["world_size"])

        # If SPEED partitioning configured, compute it now
        node_parts = None
        if partition_algo.lower() == "speed":
            try:
                dataset_root = session_ctx.dataset_path or Path(session_ctx.config["data"]["root"]).expanduser().resolve()
                dataset = TGTemporalDataset(
                    dataset_root,
                    session_ctx.config["data"]["name"],
                    split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
                    config=session_ctx.config,
                )
                print(f"Computing SPEED partitioning for {num_parts} partitions...")
                node_parts = speed_partition(dataset, num_parts, config=session_ctx.config)
                session_ctx.provider_state["node_parts"] = node_parts
                print(f"SPEED partitioning complete: {node_parts.unique().numel()} partitions assigned")
            except Exception as e:
                print(f"Warning: SPEED partitioning failed ({e}), falling back to round-robin")
                node_parts = None

        session_ctx.provider_state["partition_manifest"] = {
            "graph_mode": "ctdg",
            "num_parts": num_parts,
            "partition_algo": partition_algo,
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
            "has_node_parts": node_parts is not None,
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
            "partition_algo": str(session_ctx.config["graph"]["partition"]),
        }

        outputs = [
            ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
            ArtifactOutput("routes/manifest.json", feature_route.describe()),
            ArtifactOutput("sampling/index.json", {"dataset": session_ctx.config["data"]["name"], **stats}),
        ]

        # Save node_parts tensor if available
        if "node_parts" in session_ctx.provider_state:
            node_parts = session_ctx.provider_state["node_parts"]
            outputs.append(ArtifactOutput("partitions/node_parts.pt", node_parts))
            provider_meta["has_node_parts"] = True

        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(
                provider_meta=provider_meta,
                outputs=outputs,
            ),
        )
