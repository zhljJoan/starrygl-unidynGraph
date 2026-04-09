from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from starry_unigraph.checkpoint import load_checkpoint, save_checkpoint
from starry_unigraph.config.schema import detect_graph_mode, load_config, validate_config
from starry_unigraph.distributed import apply_distributed_env, build_distributed_context
from starry_unigraph.providers import CTDGProvider, DTDGProvider  # noqa: F401
from starry_unigraph.registry import ModelRegistry, ProviderRegistry, TaskRegistry
from starry_unigraph.types import PredictionResult, PreparedArtifacts, RuntimeBundle, SessionContext


class SchedulerSession:
    def __init__(self, session_ctx: SessionContext, provider: Any, model_spec: Any, task_adapter: Any):
        self.ctx = session_ctx
        self.provider = provider
        self.model_spec = model_spec
        self.task_adapter = task_adapter
        self.current_epoch = 0
        self.global_step = 0

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
        graph_mode = detect_graph_mode(config)
        provider_cls = ProviderRegistry.resolve(graph_mode)
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
        provider = provider_cls(task_adapter=task_cls())
        return cls(session_ctx=ctx, provider=provider, model_spec=model_spec, task_adapter=task_cls())

    def prepare_data(self) -> PreparedArtifacts:
        return self.provider.prepare_data(self.ctx)

    def build_runtime(self) -> RuntimeBundle:
        return self.provider.build_runtime(self.ctx)

    def run_epoch(self, split: str = "train") -> dict[str, Any]:
        iterator = (
            self.provider.build_train_iterator(self.ctx, split=split)
            if split == "train"
            else self.provider.build_eval_iterator(self.ctx, split=split)
        )
        start_time = time.perf_counter()
        outputs = []
        if split == "train":
            for batch in iterator:
                outputs.append(self.provider.run_train_step(batch))
                self.global_step += 1
        else:
            for batch in iterator:
                outputs.append(self.provider.run_eval_step(batch))
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

    def run_task(self) -> dict[str, Any]:
        #self.prepare_data()
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
        #if self.provider.runtime.model is None:
        #    self.build_runtime()
        predictions = []
        targets = []
        meta = {"split": split, "graph_mode": self.model_spec.graph_mode}
        for batch in self.provider.build_predict_iterator(self.ctx, split=split):
            output = self.provider.run_predict_step(batch)
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
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return loss_sum, step_count, elapsed
        device = "cuda" if str(self.ctx.config["runtime"]["device"]).startswith("cuda") else "cpu"
        stats = torch.tensor([loss_sum, float(step_count), elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(stats[0:2], op=dist.ReduceOp.SUM)
        dist.all_reduce(stats[2:3], op=dist.ReduceOp.MAX)
        return float(stats[0].item()), int(stats[1].item()), float(stats[2].item())

    def save_checkpoint(self, path: str | Path) -> Path:
        runtime_state = self.provider.runtime_adapter.dump_runtime_state(self.provider.runtime)
        payload = {
            "model_state": self.provider.runtime.model,
            "optimizer_state": self.provider.runtime.optimizer,
            "scheduler_state": self.provider.runtime.scheduler,
            "runtime_state": runtime_state,
            "provider_meta": getattr(self.provider.prepared, "provider_meta", {}),
            "config": self.ctx.config,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        return save_checkpoint(path, payload)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = load_checkpoint(path)
        if self.provider.runtime.model is None:
            self.build_runtime()
        self.provider.runtime.model = payload.get("model_state")
        self.provider.runtime.optimizer = payload.get("optimizer_state")
        self.provider.runtime.scheduler = payload.get("scheduler_state")
        self.provider.runtime_adapter.load_runtime_state(self.provider.runtime, payload.get("runtime_state", {}))
        self.current_epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))
        return payload
