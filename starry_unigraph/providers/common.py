from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starry_unigraph.preprocess import GraphPreprocessor
from starry_unigraph.runtime import ExecutionAdapter, GraphProvider, RuntimeAdapter
from starry_unigraph.types import PreparedArtifacts, RuntimeBundle, SessionContext


ARTIFACT_VERSION = 1


def ensure_artifact_dirs(session_ctx: SessionContext, graph_mode: str) -> dict[str, Path]:
    root = session_ctx.artifact_root
    directories = {
        "meta": root / "meta",
        "partitions": root / "partitions",
        "routes": root / "routes",
        "sampling": root / "sampling",
        "snapshots": root / "snapshots",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    if graph_mode == "ctdg":
        directories["snapshots"].mkdir(parents=True, exist_ok=True)
    return directories


def write_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_meta(meta_path: Path) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def validate_artifacts(
    prepared: PreparedArtifacts,
    expected_graph_mode: str,
    expected_num_parts: int | None = None,
) -> dict[str, Any]:
    if not prepared.meta_path.exists():
        raise FileNotFoundError(f"Missing artifacts metadata: {prepared.meta_path}")
    meta = read_meta(prepared.meta_path)
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


@dataclass
class SimpleRuntimeAdapter(RuntimeAdapter):
    provider_name: str

    def init_model(self, session_ctx: SessionContext) -> Any:
        return {
            "provider": self.provider_name,
            "model_name": session_ctx.config["model"]["name"],
            "device": session_ctx.config["runtime"]["device"],
        }

    def init_optimizer(self, session_ctx: SessionContext, model: Any) -> Any:
        return {
            "type": "mock_optimizer" if session_ctx.config["runtime"]["backend"] == "mock" else "native_optimizer",
            "lr": session_ctx.config.get("train", {}).get("lr", 1e-3),
            "model": model["model_name"],
        }

    def load_runtime_state(self, runtime: RuntimeBundle, state: dict[str, Any]) -> None:
        runtime.state.update(state)

    def dump_runtime_state(self, runtime: RuntimeBundle) -> dict[str, Any]:
        return dict(runtime.state)


@dataclass
class SimpleExecutionAdapter(ExecutionAdapter):
    graph_mode: str

    def train_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        runtime.state["cursor"] = batch["index"]
        runtime.state["last_split"] = "train"
        chain = batch["chain"]
        return {
            "loss": float(batch["index"] + 1),
            "predictions": [f"{self.graph_mode}:train:{chain}:{batch['index']}"],
            "targets": [batch["index"]],
            "meta": {"chain": chain},
        }

    def eval_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        runtime.state["cursor"] = batch["index"]
        runtime.state["last_split"] = "eval"
        return {
            "loss": float(batch["index"]) / 10.0,
            "predictions": [f"{self.graph_mode}:eval:{batch['index']}"],
            "targets": [batch["index"]],
            "meta": {"split": batch["split"]},
        }

    def predict_step(self, runtime: RuntimeBundle, batch: Any) -> dict[str, Any]:
        runtime.state["cursor"] = batch["index"]
        runtime.state["last_split"] = "predict"
        return {
            "predictions": [f"{self.graph_mode}:predict:{batch['index']}"],
            "targets": [batch.get("target")] if "target" in batch else None,
            "meta": {"split": batch["split"]},
        }


class BaseProvider(GraphProvider):
    provider_key = "base"

    def __init__(self, task_adapter: Any):
        super().__init__(task_adapter=task_adapter)
        self.runtime_adapter = SimpleRuntimeAdapter(self.provider_key)
        self.execution_adapter = SimpleExecutionAdapter(self.graph_mode)

    def _require_prepared(self) -> PreparedArtifacts:
        if self.prepared is None:
            raise RuntimeError("Data not prepared. Call session.prepare_data() first.")
        return self.prepared

    def _load_prepared_from_disk(self, session_ctx: SessionContext) -> PreparedArtifacts:
        meta_path = session_ctx.artifact_root / "meta" / "artifacts.json"
        if not meta_path.exists():
            raise RuntimeError("Data not prepared. Call session.prepare_data() first.")
        provider_meta = read_meta(meta_path)
        known_dirs = ("meta", "partitions", "routes", "sampling", "snapshots", "flare", "clusters")
        directories = {name: session_ctx.artifact_root / name for name in known_dirs if (session_ctx.artifact_root / name).exists()}
        prepared = PreparedArtifacts(meta_path=meta_path, directories=directories, provider_meta=provider_meta)
        self.prepared = prepared
        return prepared

    def _build_runtime_common(self, session_ctx: SessionContext) -> RuntimeBundle:
        prepared = self.prepared if self.prepared is not None else self._load_prepared_from_disk(session_ctx)
        validate_artifacts(
            prepared,
            expected_graph_mode=self.graph_mode,
            expected_num_parts=session_ctx.dist.world_size,
        )
        model = self.runtime_adapter.init_model(session_ctx)
        optimizer = self.runtime_adapter.init_optimizer(session_ctx, model)
        self.runtime = RuntimeBundle(model=model, optimizer=optimizer, state={"graph_mode": self.graph_mode})
        return self.runtime

    def run_train_step(self, batch: Any) -> dict[str, Any]:
        return self.execution_adapter.train_step(self.runtime, batch)

    def run_eval_step(self, batch: Any) -> dict[str, Any]:
        return self.execution_adapter.eval_step(self.runtime, batch)

    def run_predict_step(self, batch: Any) -> dict[str, Any]:
        return self.execution_adapter.predict_step(self.runtime, batch)
