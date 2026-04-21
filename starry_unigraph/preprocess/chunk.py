"""Chunk preprocessing pipeline: raw events -> snapshots -> chunk artifacts.

Independent from DTDG(Flare) and CTDG(online) — chunk is a separate graph mode
with its own prepare->runtime->train flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from starry_unigraph.data import build_snapshot_dataset_from_events, load_raw_temporal_events
from starry_unigraph.preprocess.base import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.types import PreparedArtifacts, SessionContext

ARTIFACT_VERSION = 1


class ChunkPreprocessor(GraphPreprocessor):
    """Preprocessor for chunk graph mode.

    Builds chunk artifacts (snapshots/ and clusters/ directories)
    independent from Flare and CTDG pipelines.
    """

    graph_mode = "chunk"
    artifact_dirs = ("meta", "partitions", "routes", "snapshots", "clusters")

    def prepare_raw(self, session_ctx: SessionContext) -> None:
        """Load raw temporal events and build snapshot dataset."""
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
        """Build partition metadata for chunks."""
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        num_parts = int(session_ctx.config["dist"]["world_size"])

        session_ctx.provider_state["partition_manifest"] = {
            "num_parts": num_parts,
            "partition_algo": str(session_ctx.config["graph"]["partition"]),
            "num_nodes": raw_dataset["num_nodes"],
            "num_edges": raw_dataset["num_edges"],
            "num_snapshots": raw_dataset["num_snapshots"],
        }

    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
        """Build chunk-specific artifacts (snapshots/ and clusters/)."""
        raw_dataset = session_ctx.provider_state["raw_dataset"]
        raw_stats = session_ctx.provider_state["raw_stats"]
        num_parts = int(session_ctx.config["dist"]["world_size"])

        provider_meta = {
            "graph_mode": self.graph_mode,
            "artifact_version": ARTIFACT_VERSION,
            "num_parts": num_parts,
            "num_nodes": raw_stats["num_nodes"],
            "num_edges": raw_stats["num_edges"],
            "num_snapshots": raw_stats["num_snapshots"],
            "feature_dim": session_ctx.config["model"]["hidden_dim"],
            "edge_feat_dim": 0,
            "label_dim": 1,
            "task_type": session_ctx.config["model"]["task"],
        }

        outputs: list[ArtifactOutput] = [
            ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
            ArtifactOutput(
                "routes/manifest.json",
                {
                    "route_type": str(session_ctx.config["graph"]["route"]),
                    "cache_policy": str(session_ctx.config["runtime"]["cache"]),
                },
            ),
            ArtifactOutput(
                "snapshots/manifest.json",
                {
                    "graph_mode": "chunk",
                    "snapshot_count": raw_dataset["num_snapshots"],
                    "num_parts": num_parts,
                },
            ),
        ]

        for part_id in range(num_parts):
            outputs.append(
                ArtifactOutput(
                    f"clusters/part_{part_id:03d}/cluster_manifest.json",
                    {
                        "partition_id": part_id,
                        "graph_mode": "chunk",
                        "cluster_count": 0,
                    },
                )
            )

        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(provider_meta=provider_meta, outputs=outputs),
        )
