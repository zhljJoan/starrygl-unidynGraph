from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.distributed as dist

from starry_unigraph.core import DTDGBatch, DTDGKernel, DTDGRuntimeState, SnapshotRoutePlan
from starry_unigraph.preprocess import GraphPreprocessor
from starry_unigraph.providers.common import BaseProvider
from starry_unigraph.registry import ProviderRegistry
from starry_unigraph.types import PreparedArtifacts, RuntimeBundle, SessionContext

from .dtdg_common import dtdg_pipeline
from .dtdg_loaders import ChunkedDTDGLoader, FlareRuntimeLoader
from .dtdg_preprocess import ChunkedDTDGPreprocessor, FlareDTDGPreprocessor
from .dtdg_train import init_flare_training, run_flare_eval_step, run_flare_predict_step, run_flare_train_step


class DTDGProvider(BaseProvider):
    graph_mode = "dtdg"
    provider_key = "dtdg"

    def __init__(self, task_adapter: Any):
        super().__init__(task_adapter=task_adapter)
        self.preprocessor: GraphPreprocessor | None = None
        self.snapshot_core: Any | None = None
        self.kernel: DTDGKernel | None = None

    def _resolve_preprocessor(self, session_ctx: SessionContext) -> GraphPreprocessor:
        pipeline = dtdg_pipeline(session_ctx)
        if pipeline == "flare_native":
            return FlareDTDGPreprocessor()
        if pipeline == "chunked":
            return ChunkedDTDGPreprocessor()
        raise ValueError(f"Unsupported dtdg.pipeline={pipeline!r}")

    def prepare_data(self, session_ctx: SessionContext) -> PreparedArtifacts:
        self.preprocessor = self._resolve_preprocessor(session_ctx)
        self.prepared = self.preprocessor.run(session_ctx)
        session_ctx.provider_state["prepared_meta"] = self.prepared.provider_meta
        return self.prepared

    def _build_flare_runtime(self, session_ctx: SessionContext, runtime: RuntimeBundle) -> RuntimeBundle:
        #加载预处理数据和optimazer配置、model配置
        prepared = self._require_prepared()
        flare_manifest_path = prepared.directories["flare"] / "manifest.json"
        if not flare_manifest_path.exists():
            raise FileNotFoundError(f"Missing flare manifest: {flare_manifest_path}")

        part_id = min(session_ctx.dist.rank, session_ctx.dist.world_size - 1)

        partition_path = prepared.directories["flare"] / f"part_{part_id:03d}.pth"
        if not partition_path.exists():
            raise FileNotFoundError(f"Missing flare partition artifact: {partition_path}")

        partition_data = torch.load(partition_path, weights_only=False)

        # In distributed mode, pin each rank to its local GPU (cuda:local_rank)
        base_device = str(session_ctx.config["runtime"]["device"])
        if base_device == "cuda" and dist.is_available() and dist.is_initialized():
            device = f"cuda:{session_ctx.dist.local_rank}"
        else:
            device = base_device

        #初始化dataloader数据
        loader = FlareRuntimeLoader.from_partition_data(
            data=partition_data,
            device=device,
            rank=session_ctx.dist.rank,
            world_size=session_ctx.dist.world_size,
            config=session_ctx.config,
        )
        self.snapshot_core = loader
        #self.kernel = DTDGKernel(snapshot_core=loader, runtime_state=DTDGRuntimeState(snapshot_cursor=0))
        #实例化模型和优化器
        init_flare_training(runtime=runtime, session_ctx=session_ctx, partition_data=partition_data, device=device)
        runtime.state.update(
            {
                "dtdg_pipeline": "flare_native",
                "flare_partition_path": str(partition_path),
                "window_state": loader.describe_window_state(),
                "snapshot_state": loader.dump_state(),
                "route_cache": loader.describe_route_cache(),
            }
        )
        return runtime

    def _build_chunked_runtime(self, session_ctx: SessionContext, runtime: RuntimeBundle) -> RuntimeBundle:
        snapshot_manifest_path = self.prepared.directories["snapshots"] / "manifest.json"
        if not snapshot_manifest_path.exists():
            raise FileNotFoundError(f"Missing chunked snapshot manifest: {snapshot_manifest_path}")

        import json

        manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        chunk_manifest = [item for item in manifest["chunks"] if item["partition_id"] == session_ctx.dist.rank]
        route_plan = SnapshotRoutePlan(
            route_type=session_ctx.config["graph"]["route"],
            cache_policy=session_ctx.config["runtime"]["cache"],
        )
        loader = ChunkedDTDGLoader(
            snaps=session_ctx.config["train"]["snaps"],
            window_size=session_ctx.config["model"]["window"]["size"],
            route_plan=route_plan,
            chunk_manifest=chunk_manifest,
            split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
        )
        self.snapshot_core = loader
        self.kernel = DTDGKernel(snapshot_core=loader, runtime_state=DTDGRuntimeState(snapshot_cursor=loader.cursor))
        runtime.state.update(
            {
                "dtdg_pipeline": "chunked",
                "chunk_manifest": chunk_manifest,
                "window_state": {"window_size": loader.window_size, "stored_windows": 0},
                "snapshot_state": loader.dump_state(),
                "route_cache": route_plan.describe(),
            }
        )
        return runtime

    def build_runtime(self, session_ctx: SessionContext) -> RuntimeBundle:
        runtime = self._build_runtime_common(session_ctx)
        pipeline = dtdg_pipeline(session_ctx)
        if pipeline == "flare_native":
            return self._build_flare_runtime(session_ctx, runtime)
        if pipeline == "chunked":
            return self._build_chunked_runtime(session_ctx, runtime)
        raise ValueError(f"Unsupported dtdg.pipeline={pipeline!r}")

    def build_train_iterator(self, session_ctx: SessionContext, split: str = "train") -> Iterable[Any]:
        if self.snapshot_core is not None and hasattr(self.snapshot_core, "iter_train"):
            yield from self.snapshot_core.iter_train(split=split)
            return
        if dtdg_pipeline(session_ctx) == "flare_native":
            assert self.snapshot_core is not None
            yield from self.snapshot_core.iter_train(split=split)
            return
        assert self.kernel is not None
        yield from self.kernel.iter_batches(split=split, count=3)

    def build_eval_iterator(self, session_ctx: SessionContext, split: str = "val") -> Iterable[Any]:
        if self.runtime.state.get("dtdg_pipeline") == "flare_native":
            # Reset stateful RNN state at the start of each eval pass
            self.runtime.state.pop("eval_rnn_state", None)
            assert self.snapshot_core is not None
            yield from self.snapshot_core.iter_eval(split=split)
            return
        if self.snapshot_core is not None and hasattr(self.snapshot_core, "iter_eval"):
            yield from self.snapshot_core.iter_eval(split=split)
            return
        assert self.kernel is not None
        for batch in self.kernel.iter_batches(split=split, count=2):
            batch.chain = "snapshot_eval"
            yield batch

    def build_predict_iterator(self, session_ctx: SessionContext, split: str = "test") -> Iterable[Any]:
        if self.runtime.state.get("dtdg_pipeline") == "flare_native":
            # Reset stateful RNN state at the start of each predict pass
            self.runtime.state.pop("eval_rnn_state", None)
            assert self.snapshot_core is not None
            yield from self.snapshot_core.iter_predict(split=split)
            return
        if self.snapshot_core is not None and hasattr(self.snapshot_core, "iter_predict"):
            yield from self.snapshot_core.iter_predict(split=split)
            return
        assert self.kernel is not None
        for batch in self.kernel.iter_batches(split=split, count=2):
            batch.chain = "snapshot_predict"
            yield batch

    def run_train_step(self, batch: Any) -> dict[str, Any]:
        if self.runtime.state.get("dtdg_pipeline") == "flare_native":
            assert self.runtime.model is not None
            return run_flare_train_step(self.runtime, batch, {"meta": {}})
        assert self.kernel is not None
        assert isinstance(batch, DTDGBatch)
        output = self.kernel.execute_train(batch).to_payload()
        kernel_state = self.kernel.dump_state()
        self.runtime.state["window_state"] = kernel_state["window"]
        self.runtime.state["snapshot_state"] = kernel_state["snapshot"]
        route_cache = dict(self.runtime.state.get("route_cache", {}))
        route_cache.update(kernel_state["snapshot"].get("route_plan", {}))
        self.runtime.state["route_cache"] = route_cache
        self.runtime.state["executor_state"] = kernel_state["runtime"]
        self.runtime.state["cursor"] = batch.index
        self.runtime.state["last_split"] = "train"
        return output

    def run_eval_step(self, batch: Any) -> dict[str, Any]:
        if self.runtime.state.get("dtdg_pipeline") == "flare_native":
            assert self.runtime.model is not None
            return run_flare_eval_step(self.runtime, batch, {"meta": {}})
        assert self.kernel is not None
        assert isinstance(batch, DTDGBatch)
        output = self.kernel.execute_eval(batch).to_payload()
        self.runtime.state["executor_state"] = self.kernel.dump_state()["runtime"]
        self.runtime.state["cursor"] = batch.index
        self.runtime.state["last_split"] = "eval"
        return output

    def run_predict_step(self, batch: Any) -> dict[str, Any]:
        if self.runtime.state.get("dtdg_pipeline") == "flare_native":
            assert self.runtime.model is not None
            return run_flare_predict_step(self.runtime, batch, {"meta": {}})
        assert self.kernel is not None
        assert isinstance(batch, DTDGBatch)
        output = self.kernel.execute_predict(batch).to_payload()
        self.runtime.state["executor_state"] = self.kernel.dump_state()["runtime"]
        self.runtime.state["cursor"] = batch.index
        self.runtime.state["last_split"] = "predict"
        return output


ProviderRegistry.register("dtdg", DTDGProvider)
