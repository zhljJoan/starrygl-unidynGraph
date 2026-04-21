"""ChunkRuntimeLoader: session.py interface for chunked DTDG.

Provides a minimal runtime loader interface compatible with FlareRuntimeLoader,
accepting prepared chunked artifacts and exposing iteration/state methods
matching the Flare contract for transparent reuse of training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from starry_unigraph.backends.dtdg.runtime.session_loader import DTDGBatch, DTDGWindowState
from starry_unigraph.backends.dtdg.types import SnapshotRoutePlan
from starry_unigraph.backends.dtdg.runtime.state import STGraphBlob


@dataclass
class ChunkRuntimeLoader:
    """High-level chunked DTDG loader for session integration.

    Mirrors :class:`FlareRuntimeLoader` public interface:
    - :meth:`iter_train` — yields :class:`STGraphBlob`
    - :meth:`iter_eval` — yields :class:`DTDGBatch`
    - :meth:`iter_predict` — yields :class:`DTDGBatch`
    - :meth:`build_snapshot_index` — returns metadata dict
    - :meth:`dump_state` — returns runtime state dict
    - :meth:`describe_window_state` — returns window metadata
    - :meth:`describe_route_cache` — returns route metadata

    Created via :meth:`from_prepared_artifacts` which accepts
    prepared_dir, device, rank, world_size, config — matching
    FlareRuntimeLoader.from_partition_data signature.
    """

    chunk_manifest: dict[str, Any]
    cluster_manifest: dict[str, Any]
    window_state: DTDGWindowState
    route_plan: SnapshotRoutePlan
    partition_id: int
    device: str
    rank: int
    world_size: int
    cursor: int = 0

    @classmethod
    def from_prepared_artifacts(
        cls,
        prepared_dir: str | Path,
        device: str,
        rank: int,
        world_size: int,
        config: dict[str, Any],
    ) -> ChunkRuntimeLoader:
        """Initialize from prepared chunk artifacts.

        Args:
            prepared_dir: Root directory of prepared artifacts.
            device: Target device (e.g., "cuda:0", "cpu").
            rank: Distributed rank.
            world_size: Total ranks.
            config: Full config dict.

        Returns:
            ChunkRuntimeLoader instance.

        Raises:
            FileNotFoundError: If required manifests are missing.
        """
        # Load manifests (to be implemented by chunk internals)
        chunk_manifest = {}  # Loaded from snapshots/manifest.json
        cluster_manifest = {}  # Loaded from clusters/part_XXX/cluster_manifest.json

        route_plan = SnapshotRoutePlan(
            route_type=str(config["graph"]["route"]),
            cache_policy=str(config["runtime"]["cache"]),
        )
        window_state = DTDGWindowState(window_size=int(config["model"]["window"]["size"]))

        return cls(
            chunk_manifest=chunk_manifest,
            cluster_manifest=cluster_manifest,
            window_state=window_state,
            route_plan=route_plan,
            partition_id=rank,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    def iter_train(self, split: str = "train") -> Iterable[STGraphBlob]:
        """Iterate training snapshots yielding STGraphBlob."""
        raise NotImplementedError("Chunk materialization layer")

    def iter_eval(self, split: str = "val") -> Iterable[DTDGBatch]:
        """Iterate evaluation snapshots yielding DTDGBatch."""
        raise NotImplementedError("Chunk materialization layer")

    def iter_predict(self, split: str = "test") -> Iterable[DTDGBatch]:
        """Iterate prediction snapshots yielding DTDGBatch."""
        raise NotImplementedError("Chunk materialization layer")

    def build_snapshot_index(self) -> dict[str, Any]:
        """Return snapshot metadata with pipeline='chunked'."""
        return {"pipeline": "chunked"}

    def dump_state(self) -> dict[str, Any]:
        """Return runtime state for observability."""
        return {"cursor": self.cursor}

    def describe_window_state(self) -> dict[str, Any]:
        """Return window state metadata."""
        return self.window_state.describe()

    def describe_route_cache(self) -> dict[str, Any]:
        """Return route cache metadata."""
        return self.route_plan.describe()

    def run_train_step(self, runtime: Any, blob: Any) -> dict[str, Any]:
        """Execute one training step (delegates to internal step handler).

        For now, raises NotImplementedError. To be implemented by chunk internals.
        """
        raise NotImplementedError("Chunk materialization + step execution layer")

    def run_eval_step(self, runtime: Any, batch: Any) -> dict[str, Any]:
        """Execute one evaluation step (delegates to internal step handler).

        For now, raises NotImplementedError. To be implemented by chunk internals.
        """
        raise NotImplementedError("Chunk materialization + step execution layer")

    def run_predict_step(self, runtime: Any, batch: Any) -> dict[str, Any]:
        """Execute one prediction step (delegates to internal step handler).

        For now, raises NotImplementedError. To be implemented by chunk internals.
        """
        raise NotImplementedError("Chunk materialization + step execution layer")
