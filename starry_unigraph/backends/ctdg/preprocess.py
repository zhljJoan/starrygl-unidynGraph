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
        from .preprocess_partition import build_round_robin_node_parts, derive_edge_parts, speed_partition

        stats = session_ctx.provider_state["ctdg_dataset_stats"]
        partition_algo = str(session_ctx.config["graph"]["partition"])
        num_parts = int(session_ctx.config["dist"]["world_size"])

        dataset_root = session_ctx.dataset_path or Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        dataset = TGTemporalDataset(
            dataset_root,
            session_ctx.config["data"]["name"],
            split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
            config=session_ctx.config,
        )

        node_parts = None
        edge_parts = None
        if partition_algo.lower() == "speed":
            try:
                print(f"Computing SPEED partitioning for {num_parts} partitions...")
                node_parts, edge_parts = speed_partition(dataset, num_parts, config=session_ctx.config)
                session_ctx.provider_state["node_parts"] = node_parts
                session_ctx.provider_state["edge_parts"] = edge_parts
                print(f"SPEED partitioning complete: {node_parts.unique().numel()} partitions assigned")
            except Exception as e:
                print(f"Warning: SPEED partitioning failed ({e}), falling back to round-robin")
                node_parts = None
                edge_parts = None

        if node_parts is None and num_parts > 1:
            node_parts = build_round_robin_node_parts(dataset.num_nodes, num_parts)
            edge_parts = derive_edge_parts(dataset.src, node_parts)
            session_ctx.provider_state["node_parts"] = node_parts
            session_ctx.provider_state["edge_parts"] = edge_parts

        if edge_parts is not None:
            batch_plan: dict[str, dict[str, int]] = {}
            for split_name in ("train", "val", "test"):
                split_eids = dataset.split_event_ids(split_name)
                local_counts = []
                for part_id in range(num_parts):
                    local_counts.append(int((edge_parts[split_eids] == part_id).sum().item()))
                batch_plan[split_name] = {
                    "num_events": int(split_eids.numel()),
                    "global_batches_per_local_batch": num_parts,
                    "max_local_events_per_part": max(local_counts) if local_counts else 0,
                    "min_local_events_per_part": min(local_counts) if local_counts else 0,
                }
            session_ctx.provider_state["batch_plan"] = batch_plan

        session_ctx.provider_state["partition_manifest"] = {
            "graph_mode": "ctdg",
            "num_parts": num_parts,
            "partition_algo": partition_algo,
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
            "has_node_parts": node_parts is not None,
            "has_edge_parts": edge_parts is not None,
            "edge_owner": "src_node_partition" if edge_parts is not None else None,
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
            "node_feat_dim": stats.get("node_feat_dim", 0),
            "task_type": session_ctx.config["model"]["task"],
            "feature_route_plan": feature_route.describe(),
            "partition_algo": str(session_ctx.config["graph"]["partition"]),
            "edge_owner": "src_node_partition",
            "batch_alignment": "global_time_window",
        }

        outputs = [
            ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
            ArtifactOutput("routes/manifest.json", feature_route.describe()),
            ArtifactOutput("sampling/index.json", {"dataset": session_ctx.config["data"]["name"], **stats}),
        ]

        # Save node_parts tensor if available
        if "node_parts" in session_ctx.provider_state:
            node_parts = session_ctx.provider_state["node_parts"]
            outputs.append(ArtifactOutput("partitions/node_parts.pt", node_parts, serializer="torch"))
            provider_meta["has_node_parts"] = True
        if "edge_parts" in session_ctx.provider_state:
            edge_parts = session_ctx.provider_state["edge_parts"]
            outputs.append(ArtifactOutput("partitions/edge_parts.pt", edge_parts, serializer="torch"))
            provider_meta["has_edge_parts"] = True
        if "batch_plan" in session_ctx.provider_state:
            outputs.append(ArtifactOutput("sampling/boundaries.json", session_ctx.provider_state["batch_plan"]))

        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(
                provider_meta=provider_meta,
                outputs=outputs,
            ),
        )
