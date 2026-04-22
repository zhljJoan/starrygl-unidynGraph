from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist

from starry_unigraph.checkpoint import load_checkpoint, save_checkpoint
from starry_unigraph.config.schema import detect_graph_mode, load_config, validate_config
from starry_unigraph.distributed import apply_distributed_env, build_distributed_context
from starry_unigraph.backends.ctdg.preprocess import CTDGPreprocessor
from starry_unigraph.backends.dtdg import FlareDTDGPreprocessor, FlareRuntimeLoader, init_flare_training, run_flare_eval_step, run_flare_predict_step, run_flare_train_step
from starry_unigraph.backends.dtdg.preprocess import load_prepared_from_disk, validate_artifacts
from starry_unigraph.preprocess.chunk import ChunkPreprocessor
from starry_unigraph.registry import ModelRegistry, TaskRegistry
from starry_unigraph.runtime.chunk import ChunkRuntimeLoader
from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
from starry_unigraph.types import PredictionResult, PreparedArtifacts, RuntimeBundle, SessionContext
from starry_unigraph.runtime.engine import PipelineEngine
from starry_unigraph.runtime.backend_adapters import CTDGGraphBackend, FlareGraphBackend, ChunkGraphBackend, DummyStateManager


class SchedulerSession:
    def __init__(self, session_ctx: SessionContext, model_spec: Any, task_adapter: Any):
        self.ctx = session_ctx
        self.model_spec = model_spec
        self.task_adapter = task_adapter
        self.current_epoch = 0
        self.global_step = 0
        self.runtime = RuntimeBundle()
        self.snapshot_core: FlareRuntimeLoader | ChunkRuntimeLoader | None = None
        self.ctdg_session: CTDGSession | None = None
        self.prepared: PreparedArtifacts | None = None
        # New: unified pipeline engine (optional, for refactored code path)
        self.pipeline_engine: PipelineEngine | None = None

    @classmethod
    def from_config(
        cls,
        config_or_path: str | Path | dict[str, Any],
        dataset_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "SchedulerSession":
        config = apply_distributed_env(load_config(config_or_path, overrides=overrides))
        warnings = validate_config(config)
        model_spec = ModelRegistry.resolve(
            model_name=config["model"]["name"],
            family=config["model"]["family"],
        )
        task_cls = TaskRegistry.resolve(config["model"]["task"])
        artifact_root = Path("artifacts") / config["data"]["name"]
        ctx = SessionContext(
            config=config,
            project_root=Path.cwd(),
            dataset_path=Path(dataset_path).expanduser().resolve() if dataset_path else None,
            artifact_root=artifact_root.resolve(),
            dist=build_distributed_context(config),
            warnings=warnings,
        )
        return cls(session_ctx=ctx, model_spec=model_spec, task_adapter=task_cls())

    def prepare_data(self) -> PreparedArtifacts:
        from starry_unigraph.config.schema import detect_graph_mode
        graph_mode = detect_graph_mode(self.ctx.config)
        if graph_mode == "ctdg":
            preprocessor = CTDGPreprocessor()
        elif graph_mode == "dtdg":
            preprocessor = FlareDTDGPreprocessor()
        elif graph_mode == "chunk":
            preprocessor = ChunkPreprocessor()
        else:
            raise ValueError(f"Unknown graph_mode for preprocessing: {graph_mode!r}")
        self.prepared = preprocessor.run(self.ctx)
        self.ctx.prepared_artifacts = self.prepared
        return self.prepared

    def build_runtime(self) -> RuntimeBundle:
        if self.prepared is None:
            self.prepared = load_prepared_from_disk(self.ctx.artifact_root)
        self.ctx.prepared_artifacts = self.prepared

        graph_mode = self.prepared.provider_meta.get("graph_mode")

        base_device = str(self.ctx.config["runtime"]["device"])
        if base_device == "cuda" and dist.is_available() and dist.is_initialized():
            device = f"cuda:{self.ctx.dist.local_rank}"
        else:
            device = base_device

        if graph_mode == "ctdg":
            # CTDG online runtime (standalone)
            self.ctdg_session = CTDGSession()
            self.ctdg_session.build_runtime(self.ctx)
            self.runtime = self.ctdg_session._runtime
        elif graph_mode == "dtdg":
            # DTDG Flare runtime (standalone)
            validate_artifacts(self.prepared, expected_graph_mode="dtdg", expected_num_parts=self.ctx.dist.world_size)
            flare_dir = self.prepared.directories["flare"]
            if not (flare_dir / "manifest.json").exists():
                raise FileNotFoundError(f"Missing flare manifest: {flare_dir / 'manifest.json'}")
            part_id = min(self.ctx.dist.rank, self.ctx.dist.world_size - 1)
            partition_path = flare_dir / f"part_{part_id:03d}.pth"
            if not partition_path.exists():
                raise FileNotFoundError(f"Missing flare partition artifact: {partition_path}")
            partition_data = torch.load(partition_path, weights_only=False)
            self.snapshot_core = FlareRuntimeLoader.from_partition_data(
                data=partition_data,
                device=device,
                rank=self.ctx.dist.rank,
                world_size=self.ctx.dist.world_size,
                config=self.ctx.config,
            )
            self.runtime = RuntimeBundle(state={"graph_mode": "dtdg"})
            init_flare_training(runtime=self.runtime, session_ctx=self.ctx, partition_data=partition_data, device=device)
            self.runtime.state.update(
                {
                    "flare_partition_path": str(partition_path),
                    "window_state": self.snapshot_core.describe_window_state(),
                    "snapshot_state": self.snapshot_core.dump_state(),
                    "route_cache": self.snapshot_core.describe_route_cache(),
                }
            )
        elif graph_mode == "chunk":
            # Chunk runtime (independent, with internal CTDG/DTDG branching)
            validate_artifacts(self.prepared, expected_graph_mode="chunk", expected_num_parts=self.ctx.dist.world_size)
            self.snapshot_core = ChunkRuntimeLoader.from_prepared_artifacts(
                prepared_dir=self.ctx.artifact_root,
                device=device,
                rank=self.ctx.dist.rank,
                world_size=self.ctx.dist.world_size,
                config=self.ctx.config,
            )
            partition_data = None  # TODO: get from chunk manifest for model init
            self.runtime = RuntimeBundle(state={"graph_mode": "chunk"})
            init_flare_training(runtime=self.runtime, session_ctx=self.ctx, partition_data=partition_data, device=device)
            self.runtime.state.update(
                {
                    "chunk_manifest": self.snapshot_core.chunk_manifest,
                    "window_state": self.snapshot_core.describe_window_state(),
                    "snapshot_state": self.snapshot_core.dump_state(),
                    "route_cache": self.snapshot_core.describe_route_cache(),
                }
            )
        else:
            raise ValueError(f"Unknown graph_mode: {graph_mode}")
        return self.runtime

    def build_pipeline_engine(self, model: Any = None) -> PipelineEngine:
        """Build unified PipelineEngine (new architecture, optional).

        This creates a PipelineEngine that composes GraphBackend + TaskAdapter + StateManager + Model.
        For now, this is opt-in. Existing code path via run_epoch() still works.

        Args:
            model: Neural network model (if None, will be loaded from model_spec)

        Returns:
            PipelineEngine ready for run_epoch() calls

        Note:
            This is part of the Graph Mode × Task Type orthogonal separation refactoring.
        """
        if self.prepared is None:
            self.prepared = load_prepared_from_disk(self.ctx.artifact_root)

        graph_mode = self.prepared.provider_meta.get("graph_mode")
        base_device = str(self.ctx.config["runtime"]["device"])
        if base_device == "cuda" and dist.is_available() and dist.is_initialized():
            device = f"cuda:{self.ctx.dist.local_rank}"
        else:
            device = base_device

        # Build GraphBackend and StateManager adapters based on graph_mode
        if graph_mode == "ctdg":
            if self.ctdg_session is None:
                self.ctdg_session = CTDGSession()
                self.ctdg_session.build_runtime(self.ctx)
            backend = CTDGGraphBackend(self.ctdg_session)
        elif graph_mode == "dtdg":
            if self.snapshot_core is None:
                # Build FlareRuntimeLoader (same as build_runtime)
                validate_artifacts(self.prepared, expected_graph_mode="dtdg", expected_num_parts=self.ctx.dist.world_size)
                flare_dir = self.prepared.directories["flare"]
                part_id = min(self.ctx.dist.rank, self.ctx.dist.world_size - 1)
                partition_path = flare_dir / f"part_{part_id:03d}.pth"
                partition_data = torch.load(partition_path, weights_only=False)
                self.snapshot_core = FlareRuntimeLoader.from_partition_data(
                    data=partition_data,
                    device=device,
                    rank=self.ctx.dist.rank,
                    world_size=self.ctx.dist.world_size,
                    config=self.ctx.config,
                )
            backend = FlareGraphBackend(self.snapshot_core)
        elif graph_mode == "chunk":
            if self.snapshot_core is None:
                validate_artifacts(self.prepared, expected_graph_mode="chunk", expected_num_parts=self.ctx.dist.world_size)
                self.snapshot_core = ChunkRuntimeLoader.from_prepared_artifacts(
                    prepared_dir=self.ctx.artifact_root,
                    device=device,
                    rank=self.ctx.dist.rank,
                    world_size=self.ctx.dist.world_size,
                    config=self.ctx.config,
                )
            backend = ChunkGraphBackend(self.snapshot_core)
        else:
            raise ValueError(f"Unknown graph_mode: {graph_mode}")

        # Create PipelineEngine
        state_manager = DummyStateManager()  # Placeholder; could be replaced with real state managers
        self.pipeline_engine = PipelineEngine(
            backend=backend,
            state_manager=state_manager,
            model=model or self.model_spec,  # Placeholder model
            task_adapter=self.task_adapter,
            device=device,
        )

        return self.pipeline_engine

    def run_epoch(self, split: str = "train") -> dict[str, Any]:
        if self.ctdg_session is not None:
            # CTDG path (standalone)
            if split == "train":
                iterator = self.ctdg_session.iter_train(self.ctx)
            else:
                iterator = self.ctdg_session.iter_eval(self.ctx, split=split)

            start_time = time.perf_counter()
            outputs = []
            if split == "train":
                for batch in iterator:
                    outputs.append(self.ctdg_session.train_step(batch))
                    self.global_step += 1
            else:
                for batch in iterator:
                    outputs.append(self.ctdg_session.eval_step(batch))
            elapsed = time.perf_counter() - start_time

            if split == "train":
                self.current_epoch += 1

            losses = [float(item["loss"]) for item in outputs if "loss" in item]
            metric_accumulator: dict[str, list[float]] = {}
            for item in outputs:
                metrics = item.get("meta", {}).get("metrics", {})
                for key, value in metrics.items():
                    metric_accumulator.setdefault(key, []).append(float(value))

            loss_sum = float(sum(losses))
            step_count = len(losses) if split != "train" else len(outputs)
            total_loss_sum, total_steps, max_elapsed = self._distributed_epoch_stats(loss_sum, step_count, elapsed)
            return {
                "split": split,
                "steps": total_steps,
                "local_steps": step_count,
                "loss": total_loss_sum / total_steps if total_steps else 0.0,
                "local_loss": loss_sum / step_count if step_count else 0.0,
                "metrics": {
                    key: float(sum(values) / len(values))
                    for key, values in metric_accumulator.items()
                    if values
                },
                "elapsed_s": max_elapsed,
                "local_elapsed_s": elapsed,
                "outputs": outputs,
            }
        else:
            # DTDG or Chunk path (snapshot_core dispatch)
            assert self.snapshot_core is not None, "Call build_runtime() first"
            if split == "train":
                iterator = self.snapshot_core.iter_train(split=split)
            else:
                self.runtime.state.pop("eval_rnn_state", None)
                iterator = self.snapshot_core.iter_eval(split=split)

            start_time = time.perf_counter()
            outputs = []
            if split == "train":
                for batch in iterator:
                    # snapshot_core handles its own step dispatch (Flare or Chunk)
                    output = self.snapshot_core.run_train_step(self.runtime, batch)
                    outputs.append(output)
                    self.global_step += 1
            else:
                for batch in iterator:
                    output = self.snapshot_core.run_eval_step(self.runtime, batch)
                    outputs.append(output)
            elapsed = time.perf_counter() - start_time

            if split == "train":
                self.current_epoch += 1

            losses = [self.task_adapter.compute_loss(item) for item in outputs if "loss" in item]
            metric_accumulator: dict[str, list[float]] = {}
            for item in outputs:
                metrics = item.get("meta", {}).get("metrics", {})
                for key, value in metrics.items():
                    metric_accumulator.setdefault(key, []).append(float(value))

            loss_sum = float(sum(losses))
            step_count = len(losses) if split != "train" else len(outputs)
            total_loss_sum, total_steps, max_elapsed = self._distributed_epoch_stats(loss_sum, step_count, elapsed)
            return {
                "split": split,
                "steps": total_steps,
                "local_steps": step_count,
                "loss": total_loss_sum / total_steps if total_steps else 0.0,
                "local_loss": loss_sum / step_count if step_count else 0.0,
                "metrics": {
                    key: float(sum(values) / len(values))
                    for key, values in metric_accumulator.items()
                    if values
                },
                "elapsed_s": max_elapsed,
                "local_elapsed_s": elapsed,
                "outputs": outputs,
            }

            loss_sum = float(sum(losses))
            step_count = len(losses) if split != "train" else len(outputs)
            total_loss_sum, total_steps, max_elapsed = self._distributed_epoch_stats(loss_sum, step_count, elapsed)
            return {
                "split": split,
                "steps": total_steps,
                "local_steps": step_count,
                "loss": total_loss_sum / total_steps if total_steps else 0.0,
                "local_loss": loss_sum / step_count if step_count else 0.0,
                "metrics": {
                    key: float(sum(values) / len(values))
                    for key, values in metric_accumulator.items()
                    if values
                },
                "elapsed_s": max_elapsed,
                "local_elapsed_s": elapsed,
                "outputs": outputs,
            }

    def run_task(self) -> dict[str, Any]:
        self.build_runtime()
        train_summary = []
        eval_summary = []
        loop_start = time.perf_counter()
        for epoch in range(self.ctx.config["train"]["epochs"]):
            train_summary.append(self.run_epoch(split="train"))
            if (epoch + 1) % self.ctx.config["train"]["eval_interval"] == 0:
                eval_summary.append(self.run_epoch(split="val"))
        loop_elapsed = time.perf_counter() - loop_start
        train_elapsed = sum(item.get("local_elapsed_s", 0.0) for item in train_summary)
        train_elapsed_global = sum(item.get("elapsed_s", 0.0) for item in train_summary)
        return {
            "epochs": self.current_epoch,
            "train": train_summary,
            "eval": eval_summary,
            "train_total_s": train_elapsed_global,
            "train_total_local_s": train_elapsed,
            "run_task_total_s": loop_elapsed,
            "warnings": self.ctx.warnings,
        }

    def evaluate(self, split: str = "test") -> dict[str, Any]:
        return self.run_epoch(split=split)

    def predict(self, split: str = "test") -> PredictionResult:
        if self.ctdg_session is not None:
            # CTDG path (standalone)
            predictions = []
            targets = []
            for batch in self.ctdg_session.iter_predict(self.ctx, split=split):
                output = self.ctdg_session.predict_step(batch)
                predictions.extend(output.get("predictions", []))
                if output.get("targets") is not None:
                    targets.extend(output.get("targets", []))
            return PredictionResult(
                predictions=predictions,
                targets=targets or None,
                meta={"split": split, "graph_mode": self.model_spec.graph_mode},
            )
        else:
            # DTDG or Chunk path (snapshot_core dispatch)
            assert self.snapshot_core is not None, "Call build_runtime() first"
            self.runtime.state.pop("eval_rnn_state", None)
            predictions = []
            targets = []
            meta = {"split": split, "graph_mode": self.model_spec.graph_mode}
            for batch in self.snapshot_core.iter_predict(split=split):
                # snapshot_core handles its own predict dispatch (Flare or Chunk)
                output = self.snapshot_core.run_predict_step(self.runtime, batch)
                predictions.extend(output.get("predictions", []))
                if output.get("targets") is not None:
                    targets.extend(output.get("targets", []))
            return PredictionResult(
                predictions=predictions,
                targets=targets or None,
                meta=meta,
            )

    def _distributed_epoch_stats(self, loss_sum: float, step_count: int, elapsed: float) -> tuple[float, int, float]:
        if not self.ctx.dist.is_distributed:
            return loss_sum, step_count, elapsed
        if not dist.is_initialized():
            return loss_sum, step_count, elapsed
        device = "cuda" if str(self.ctx.config["runtime"]["device"]).startswith("cuda") else "cpu"
        stats = torch.tensor([loss_sum, float(step_count), elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(stats[0:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(stats[2:3], op=dist.ReduceOp.MAX)
        return float(stats[0].item()), int(stats[1].item()), float(stats[2].item())

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.ctdg_session is not None:
            self.ctdg_session.save_checkpoint(path)
            return Path(path)
        else:
            payload = {
                "model_state": self.runtime.model,
                "optimizer_state": self.runtime.optimizer,
                "scheduler_state": self.runtime.scheduler,
                "config": self.ctx.config,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }
            return save_checkpoint(path, payload)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        if self.ctdg_session is not None:
            self.ctdg_session.load_checkpoint(path)
            return {}
        else:
            payload = load_checkpoint(path)
            if self.runtime.model is None:
                self.build_runtime()
            self.runtime.model = payload.get("model_state")
            self.runtime.optimizer = payload.get("optimizer_state")
            self.runtime.scheduler = payload.get("scheduler_state")
            self.current_epoch = int(payload.get("epoch", 0))
            self.global_step = int(payload.get("global_step", 0))
            return payload
