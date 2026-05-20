from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch

from atc_starrygl_lib.core.types import ArtifactBundle, Batch, RuntimeContext
from atc_starrygl_lib.features import CTDGFeatureRuntime, FeatureStore
from atc_starrygl_lib.comm.dynamic import DynamicFetchComm, DynamicPushComm
from atc_starrygl_lib.memory import MailboxRuntime, MailboxStore, MemoryRuntime, MemoryStore
from atc_starrygl_lib.runtime.index import DistIndexTables
from atc_starrygl_lib.sampling import MemShareNativeSamplerFactory, NativeSamplerConfig, TemporalGraphData
from atc_starrygl_lib.sampling.negative import NegativeSampler, NegativeSamplingRequest, PoolNegativeSampler, RandomNegativeSampler
from atc_starrygl_lib.sampling.temporal import RootSet, TemporalSamplingRequest


class MemShareTemporalSamplingBackend:
    """Thin bridge for the existing MemShare-compatible CTDG runtime."""

    def __init__(self) -> None:
        self._session = None
        self._runtime = None
        self._prepared_by = "legacy"

    def prepare(self, ctx: RuntimeContext) -> ArtifactBundle:
        graph_cfg = dict(ctx.config.get("graph", {}))
        prep_cfg = dict(ctx.config.get("preprocess", {}))
        if bool(prep_cfg.get("use_new_pipeline", False)):
            mode = str(prep_cfg.get("mode", "event")).strip().lower()
            if mode != "event":
                raise ValueError("CTDG new preprocess pipeline currently supports event mode only")
            bundle = self._prepare_new_pipeline(ctx, graph_cfg=graph_cfg, prep_cfg=prep_cfg)
            self._prepared_by = "new_pipeline"
            return bundle

        from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
        from starry_unigraph.types import SessionContext

        session_config = self._build_session_config_with_event_time_ptr(ctx.config, graph_cfg=graph_cfg, prep_cfg=prep_cfg)

        session_ctx = SessionContext(
            config=session_config,
            project_root=ctx.artifact_root.parent,
            artifact_root=ctx.artifact_root,
        )
        session = CTDGSession()
        prepared = session.prepare_data(session_ctx)
        self._session = session
        self._prepared_by = "legacy"
        return ArtifactBundle(
            root=ctx.artifact_root,
            graph_mode="ctdg",
            files={name: path for name, path in prepared.directories.items()},
            meta=dict(prepared.provider_meta),
        )

    def _build_session_config_with_event_time_ptr(self, config: object, *, graph_cfg: dict, prep_cfg: dict) -> dict:
        session_config = dict(config)
        if str(prep_cfg.get("mode", "event")).strip().lower() != "event":
            return session_config
        if prep_cfg.get("batch_size") is None and prep_cfg.get("num_windows") is None:
            return session_config
        source = graph_cfg.get("source") or graph_cfg.get("path") or prep_cfg.get("source")
        if source is None:
            return session_config
        from atc_starrygl_lib.preprocess.dataset import build_dataset

        dataset = build_dataset(
            data=source,
            mode="event",
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            batch_size=prep_cfg.get("batch_size"),
            num_windows=prep_cfg.get("num_windows"),
            hot_ratio=float(prep_cfg.get("hot_ratio", 0.0)),
            hot_topk=int(prep_cfg.get("hot_topk", 0)),
            node_count_weight=float(prep_cfg.get("node_count_weight", 1.0)),
            speed_beta=float(prep_cfg.get("speed_beta", 0.5)),
            speed_topk_type=str(prep_cfg.get("speed_topk_type", "degree")),
            random_node_feat_dim=int(graph_cfg.get("random_node_feat_dim", prep_cfg.get("random_node_feat_dim", 0))),
            random_node_feat_seed=int(graph_cfg.get("random_node_feat_seed", prep_cfg.get("random_node_feat_seed", 0))),
        )
        next_prep = dict(prep_cfg)
        next_prep["time_ptr_2"] = dataset["time_ptr_2"].tolist()
        next_prep["split"] = dataset["split"].tolist()
        session_config["preprocess"] = next_prep
        return session_config

    def _prepare_new_pipeline(self, ctx: RuntimeContext, *, graph_cfg: dict, prep_cfg: dict) -> ArtifactBundle:
        from atc_starrygl_lib.preprocess.pipeline import run_preprocess_pipeline

        source = graph_cfg.get("source") or graph_cfg.get("path") or prep_cfg.get("source")
        if source is None:
            raise ValueError("new CTDG preprocess pipeline requires graph.source (or graph.path/preprocess.source)")
        out_dir = Path(ctx.artifact_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_preprocess_pipeline(
            data=source,
            out_dir=out_dir,
            world_size=int(ctx.world_size),
            algorithm=str(prep_cfg.get("partition_algorithm", "speed_partition")),
            chunks_per_rank=int(prep_cfg.get("chunks_per_rank", 1)),
            mode="event",
            build_feature=bool(prep_cfg.get("build_feature", True)),
            build_partition_data=bool(prep_cfg.get("build_partition_data", False)),
            train_ratio=float(prep_cfg.get("train_ratio", 0.7)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            batch_size=prep_cfg.get("batch_size"),
            num_windows=prep_cfg.get("num_windows"),
            hot_ratio=float(prep_cfg.get("hot_ratio", 0.0)),
            hot_topk=int(prep_cfg.get("hot_topk", 0)),
            node_count_weight=float(prep_cfg.get("node_count_weight", 1.0)),
            speed_beta=float(prep_cfg.get("speed_beta", 0.5)),
            speed_topk_type=str(prep_cfg.get("speed_topk_type", "degree")),
        )
        files = {
            "graph": out_dir / "graph.pt",
            "dist": out_dir / "dist.pt",
            "meta": out_dir / "meta.json",
        }
        for rank in range(len(result["ranks"])):
            files[f"rank_{rank:03d}"] = out_dir / f"rank_{rank:03d}.pt"
            if bool(prep_cfg.get("build_feature", True)):
                files[f"feature_{rank:03d}"] = out_dir / f"feature_{rank:03d}.pt"
            if bool(prep_cfg.get("build_partition_data", False)):
                files[f"partition_data_{rank:03d}"] = out_dir / f"partition_data_{rank:03d}.pt"
        return ArtifactBundle(root=out_dir, graph_mode="ctdg", files=files, meta=dict(result["meta"]))

    def build_runtime(self, ctx: RuntimeContext, artifacts: ArtifactBundle) -> None:
        if self._prepared_by == "new_pipeline" or (
            artifacts.files.get("graph") is not None
            and artifacts.files.get("dist") is not None
            and artifacts.files.get(f"rank_{int(ctx.rank):03d}") is not None
        ):
            self._runtime = _CTDGArtifactRuntime.from_artifacts(ctx, artifacts)
            self._prepared_by = "new_pipeline"
            return
        from starry_unigraph.backends.ctdg.runtime.session import CTDGSession
        from starry_unigraph.types import SessionContext

        session_ctx = SessionContext(
            config=dict(ctx.config),
            project_root=artifacts.root.parent,
            artifact_root=artifacts.root,
        )
        self._session = self._session or CTDGSession()
        self._session.build_runtime(session_ctx)

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if self._prepared_by == "new_pipeline":
            if self._runtime is None:
                raise RuntimeError("build_runtime() must run before iter_batches()")
            yield from self._runtime.iter_batches(split)
            return
        if self._session is None:
            raise RuntimeError("build_runtime() must run before iter_batches()")
        iterator = self._session.iter_train(self._session.online_runtime.ctx) if split == "train" else self._session.iter_eval(self._session.online_runtime.ctx, split)
        for old_batch in iterator:
            yield _coerce_ctdg_batch(old_batch, split)

    def reset_state(self) -> None:
        if self._prepared_by == "new_pipeline":
            if self._runtime is not None:
                self._runtime.reset_state()
            return
        runtime = getattr(self._session, "online_runtime", None)
        mailbox = getattr(runtime, "mailbox", None)
        if mailbox is not None and hasattr(mailbox, "reset"):
            mailbox.reset()

    def reset_profile_stats(self) -> None:
        if self._prepared_by == "new_pipeline" and self._runtime is not None:
            self._runtime.reset_profile_stats()

    def pop_profile_stats(self) -> dict[str, float]:
        if self._prepared_by == "new_pipeline" and self._runtime is not None:
            return self._runtime.pop_profile_stats()
        return {}


def _coerce_ctdg_batch(old_batch: object, split: str) -> Batch:
    pos_src = getattr(old_batch, "pos_src", None)
    pos_dst = getattr(old_batch, "pos_dst", None)
    ts = getattr(old_batch, "timestamps", None)
    if pos_src is None:
        pos_src = torch.empty(0, dtype=torch.long)
    roots = pos_src if pos_src is not None else torch.empty(0, dtype=torch.long)
    return Batch(
        split=split,
        roots=roots,
        timestamps=ts,
        graph=getattr(old_batch, "graph", old_batch),
        pos_src=pos_src,
        pos_dst=pos_dst,
        neg_src=getattr(old_batch, "neg_src", None),
        neg_dst=getattr(old_batch, "neg_dst", None),
        neg_weight=getattr(old_batch, "neg_weight", None),
        node_ids=getattr(old_batch, "node_ids", None),
    )


class _CTDGArtifactRuntime:
    def __init__(
        self,
        *,
        graph: dict[str, Any],
        dist: dict[str, Any],
        rank_artifact: dict[str, Any],
        feature_artifact: dict[str, Any] | None,
        device: torch.device,
        task_name: str,
        node_batch_size: int,
        sampler: Any = None,
        feature_runtime: Any = None,
        memory_runtime: Any = None,
        mailbox_runtime: Any = None,
        negative_sampler: NegativeSampler | None = None,
        negative_ratio: int = 0,
        negative_dst_pool: torch.Tensor | None = None,
        local_negative_dst_pool: torch.Tensor | None = None,
        remote_negative_dst_pool: torch.Tensor | None = None,
        prefetch_batches: bool = True,
    ) -> None:
        self.graph = graph
        self.dist = dist
        self.rank_artifact = rank_artifact
        self.feature_artifact = feature_artifact
        self.edge_feat_dim = _feature_dim(None if feature_artifact is None else feature_artifact.get("edge_feat"))
        self.device = device
        self.task_name = task_name
        self.node_batch_size = int(node_batch_size)
        self.sampler = sampler
        self.feature_runtime = feature_runtime
        self.memory_runtime = memory_runtime
        self.mailbox_runtime = mailbox_runtime
        self.negative_sampler = negative_sampler
        self.negative_ratio = int(negative_ratio)
        self.negative_dst_pool = None if negative_dst_pool is None else negative_dst_pool.long().cpu().contiguous()
        self.local_negative_dst_pool = None if local_negative_dst_pool is None else local_negative_dst_pool.long().cpu().contiguous()
        self.remote_negative_dst_pool = None if remote_negative_dst_pool is None else remote_negative_dst_pool.long().cpu().contiguous()
        self.prefetch_batches = bool(prefetch_batches)
        self.local_edge_ids = rank_artifact["local_edge_ids"].long().cpu().contiguous()
        self.graph_src = graph["src"].long().cpu().contiguous()
        self.graph_dst = graph["dst"].long().cpu().contiguous()
        self.graph_ts = torch.as_tensor(graph["ts"]).cpu().contiguous()
        self.graph_eids = graph.get("edge_ids", torch.arange(int(self.graph_src.numel()), dtype=torch.long)).long().cpu().contiguous()
        self.graph_edge_feat = None if graph.get("edge_feat") is None else torch.as_tensor(graph["edge_feat"]).cpu().contiguous()
        self.graph_edge_label = None if graph.get("edge_label") is None else torch.as_tensor(graph["edge_label"]).cpu().contiguous()
        self._profile_stats: dict[str, float] = {}
        self.reset_profile_stats()

    @classmethod
    def from_artifacts(cls, ctx: RuntimeContext, artifacts: ArtifactBundle) -> "_CTDGArtifactRuntime":
        rank = int(ctx.rank)
        feature_path = artifacts.files.get(f"feature_{rank:03d}")
        task_cfg = dict(ctx.config.get("task", {}))
        prep_cfg = dict(ctx.config.get("preprocess", {}))
        runtime_cfg = dict(ctx.config.get("runtime", {}))
        graph = _load_torch_artifact(artifacts.require("graph"))
        dist = _load_torch_artifact(artifacts.require("dist"))
        rank_artifact = _load_torch_artifact(artifacts.require(f"rank_{rank:03d}"))
        feature_artifact = _load_torch_artifact(feature_path) if feature_path is not None and feature_path.exists() else None
        sampler = runtime_cfg.get("sampler")
        if sampler is None and bool(runtime_cfg.get("build_sampler", False)):
            sampler = _build_native_sampler(
                graph=graph,
                dist=dist,
                ctx=ctx,
                runtime_cfg=runtime_cfg,
            )
        feature_runtime = runtime_cfg.get("feature_runtime")
        if feature_runtime is None and bool(runtime_cfg.get("build_feature_runtime", False)):
            feature_runtime = _build_feature_runtime(
                rank_artifact=rank_artifact,
                dist=dist,
                feature_artifact=feature_artifact,
                ctx=ctx,
            )
        memory_runtime = runtime_cfg.get("memory_runtime")
        if memory_runtime is None and bool(runtime_cfg.get("build_memory_runtime", False)):
            memory_runtime = _build_memory_runtime(
                rank_artifact=rank_artifact,
                dist=dist,
                ctx=ctx,
                runtime_cfg=runtime_cfg,
            )
        mailbox_runtime = runtime_cfg.get("mailbox_runtime")
        if mailbox_runtime is None and bool(runtime_cfg.get("build_mailbox_runtime", False)):
            mailbox_runtime = _build_mailbox_runtime(
                rank_artifact=rank_artifact,
                dist=dist,
                ctx=ctx,
                runtime_cfg=runtime_cfg,
            )
        negative_sampler = runtime_cfg.get("negative_sampler")
        negative_ratio = int(runtime_cfg.get("negative_ratio", 0))
        if negative_sampler is None and negative_ratio > 0:
            negative_sampler = _build_negative_sampler(runtime_cfg)
        negative_dst_pool = _negative_dst_pool(graph=graph, runtime_cfg=runtime_cfg)
        local_negative_dst_pool, remote_negative_dst_pool = _rank_negative_dst_pools(
            graph=graph,
            dist=dist,
            rank=int(ctx.rank),
            runtime_cfg=runtime_cfg,
        )
        return cls(
            graph=graph,
            dist=dist,
            rank_artifact=rank_artifact,
            feature_artifact=feature_artifact,
            device=torch.device(ctx.device),
            task_name=str(task_cfg.get("name", "edge_prediction")).strip().lower(),
            node_batch_size=int(task_cfg.get("batch_size", prep_cfg.get("node_batch_size", prep_cfg.get("batch_size", 1024)))),
            sampler=sampler,
            feature_runtime=feature_runtime,
            memory_runtime=memory_runtime,
            mailbox_runtime=mailbox_runtime,
            negative_sampler=negative_sampler,
            negative_ratio=negative_ratio,
            negative_dst_pool=negative_dst_pool,
            local_negative_dst_pool=local_negative_dst_pool,
            remote_negative_dst_pool=remote_negative_dst_pool,
            prefetch_batches=bool(runtime_cfg.get("prefetch_batches", True)),
        )

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if self.task_name.startswith("node_") or self.task_name in {"node_prediction", "node_regression"}:
            batches = self._iter_node_batches(str(split))
            yield from self._maybe_sample_batches(batches)
            return
        if self.sampler is not None:
            yield from self._maybe_sample_edge_batches(str(split))
            return
        yield from self._iter_edge_batches(str(split))

    def reset_state(self) -> None:
        if self.memory_runtime is not None and hasattr(self.memory_runtime, "reset_state"):
            self.memory_runtime.reset_state()
        if self.mailbox_runtime is not None and hasattr(self.mailbox_runtime, "reset_state"):
            self.mailbox_runtime.reset_state()
        if self.sampler is not None and hasattr(self.sampler, "reset"):
            self.sampler.reset()

    def reset_profile_stats(self) -> None:
        if self.sampler is not None and hasattr(self.sampler, "reset_profile_stats"):
            self.sampler.reset_profile_stats()
        self._profile_stats = {
            "backend_batch_build_seconds": 0.0,
            "backend_negative_attach_seconds": 0.0,
            "backend_sampling_seconds": 0.0,
            "backend_sampling_request_seconds": 0.0,
            "backend_sampling_native_seconds": 0.0,
            "backend_submit_reads_seconds": 0.0,
            "backend_root_remap_seconds": 0.0,
            "backend_submit_node_feature_seconds": 0.0,
            "backend_submit_edge_feature_seconds": 0.0,
            "backend_submit_memory_seconds": 0.0,
            "backend_submit_mailbox_seconds": 0.0,
            "backend_materialize_seconds": 0.0,
            "backend_wait_patch_seconds": 0.0,
            "backend_wait_node_feature_seconds": 0.0,
            "backend_wait_edge_feature_seconds": 0.0,
            "backend_wait_memory_seconds": 0.0,
            "backend_wait_mailbox_seconds": 0.0,
            "backend_patch_inputs_seconds": 0.0,
            "backend_batches": 0.0,
        }

    def pop_profile_stats(self) -> dict[str, float]:
        out = dict(self._profile_stats)
        if self.sampler is not None and hasattr(self.sampler, "pop_profile_stats"):
            out.update({f"backend_{k}": float(v) for k, v in self.sampler.pop_profile_stats().items()})
        self.reset_profile_stats()
        return out

    def _iter_edge_event_positions(self, split: str) -> Iterator[torch.Tensor]:
        split = str(split)
        packed = self.rank_artifact.get("split_event_pos", {}).get(split)
        if isinstance(packed, dict):
            data = torch.as_tensor(packed.get("data"), dtype=torch.long).cpu().contiguous()
            ptr = torch.as_tensor(packed.get("ptr"), dtype=torch.long).cpu().contiguous()
            for index in range(max(int(ptr.numel()) - 1, 0)):
                begin = int(ptr[index])
                end = int(ptr[index + 1])
                yield data[begin:end].long().contiguous()
            return
        global_windows = _split_windows(self.graph, split)
        local_windows = self.rank_artifact.get("split_time_ptr", {}).get(split)
        if local_windows is None:
            local_windows = global_windows
        for index in range(int(local_windows.size(0))):
            if index < int(global_windows.size(0)):
                begin, end = (int(v) for v in global_windows[index].tolist())
                event_pos = _window_local_event_positions(self.local_edge_ids, begin=begin, end=end)
            else:
                event_pos = torch.empty(0, dtype=torch.long)
            yield event_pos

    def _iter_edge_batches(self, split: str) -> Iterator[Batch]:
        split = str(split)
        for event_pos in self._iter_edge_event_positions(split):
            yield self._batch_from_event_positions(split=split, event_pos=event_pos)

    def _iter_node_batches(self, split: str) -> Iterator[Batch]:
        labels = self.graph.get("node_label")
        if labels is None:
            return
        labels = torch.as_tensor(labels).cpu().contiguous()
        nodes = self.graph.get("node_label_nodes")
        if nodes is None:
            nodes = torch.arange(int(labels.size(0)), dtype=torch.long)
        else:
            nodes = torch.as_tensor(nodes, dtype=torch.long).cpu().contiguous()
        ts = self.graph.get("node_label_ts")
        if ts is None:
            ts = torch.zeros((int(nodes.numel()),), dtype=torch.float32)
        else:
            ts = torch.as_tensor(ts).cpu().contiguous()
        split_ids = self.graph.get("node_label_split")
        if split_ids is None:
            split_ids = torch.zeros((int(nodes.numel()),), dtype=torch.uint8)
        else:
            split_ids = torch.as_tensor(split_ids, dtype=torch.uint8).cpu().contiguous()
        wanted = {"train": 0, "val": 1, "test": 2}[split]
        positions = (split_ids == wanted).nonzero(as_tuple=True)[0].long().cpu().contiguous()
        for begin in range(0, int(positions.numel()), self.node_batch_size):
            pos = positions[begin : begin + self.node_batch_size]
            batch_nodes = nodes.index_select(0, pos).long().contiguous()
            batch_ts = ts.index_select(0, pos).contiguous()
            batch_labels = labels.index_select(0, pos).contiguous()
            yield Batch(
                split=split,
                roots=batch_nodes.to(self.device),
                timestamps=batch_ts.to(self.device),
                graph=None,
                labels=batch_labels.to(self.device),
                node_ids=batch_nodes.to(self.device),
            )

    def _maybe_sample_batches(self, batches: Iterator[Batch]) -> Iterator[Batch]:
        if self.sampler is None:
            yield from batches
            return
        if not self.prefetch_batches:
            for batch in batches:
                yield self._sample_and_patch(self._attach_negative(batch))
            return
        with ThreadPoolExecutor(max_workers=1) as executor:
            iterator = iter(batches)
            try:
                first = next(iterator)
            except StopIteration:
                return
            future = executor.submit(self._sample_batch, self._attach_negative(first))
            for next_batch in iterator:
                current_future = future
                future = executor.submit(self._sample_batch, self._attach_negative(next_batch))
                yield self._patch_sampled_batch(current_future.result())
            yield self._patch_sampled_batch(future.result())

    def _maybe_sample_edge_batches(self, split: str) -> Iterator[Batch]:
        if self.sampler is None:
            yield from self._iter_edge_batches(split)
            return
        positions = self._iter_edge_event_positions(split)
        if not self.prefetch_batches:
            for event_pos in positions:
                yield self._patch_sampled_batch(self._prepare_sampled_edge_batch(split, event_pos))
            return
        with ThreadPoolExecutor(max_workers=1) as executor:
            iterator = iter(positions)
            try:
                first = next(iterator)
            except StopIteration:
                return
            future = executor.submit(self._prepare_sampled_edge_batch, split, first)
            for next_event_pos in iterator:
                current_future = future
                future = executor.submit(self._prepare_sampled_edge_batch, split, next_event_pos)
                yield self._patch_sampled_batch(current_future.result())
            yield self._patch_sampled_batch(future.result())

    def _prepare_sampled_edge_batch(self, split: str, event_pos: torch.Tensor) -> tuple[Batch, Any]:
        batch = self._batch_from_event_positions(split=split, event_pos=event_pos)
        batch = self._attach_negative(batch)
        return self._sample_batch(batch)

    def _attach_negative(self, batch: Batch) -> Batch:
        t0 = time.perf_counter()
        if self.negative_sampler is None or self.negative_ratio <= 0:
            self._profile_stats["backend_negative_attach_seconds"] += float(time.perf_counter() - t0)
            return batch
        if batch.src is None or batch.dst is None or batch.ts is None:
            self._profile_stats["backend_negative_attach_seconds"] += float(time.perf_counter() - t0)
            return batch
        result = self.negative_sampler.sample(
            NegativeSamplingRequest(
                pos_src=batch.src,
                pos_dst=batch.dst,
                num_nodes=int(self.graph["num_nodes"]),
                ratio=int(self.negative_ratio),
                split=batch.split,
                dst_pool=self.negative_dst_pool,
                local_dst_pool=self.local_negative_dst_pool,
                remote_dst_pool=self.remote_negative_dst_pool,
            )
        )
        neg_roots = result.neg_dst.to(batch.roots.device).long().contiguous()
        neg_begin = int(batch.roots.numel())
        batch.roots = torch.cat([batch.roots, neg_roots], dim=0).long().contiguous()
        batch.neg_dst = torch.arange(
            neg_begin,
            neg_begin + int(neg_roots.numel()),
            dtype=torch.long,
            device=batch.roots.device,
        )
        if result.weight is not None:
            batch.neg_weight = result.weight.to(batch.roots.device, dtype=torch.float32).contiguous()
        neg_ts = batch.ts.repeat_interleave(int(result.ratio)).to(batch.timestamps.device)
        batch.timestamps = torch.cat([batch.timestamps, neg_ts], dim=0).contiguous()
        self._profile_stats["backend_negative_attach_seconds"] += float(time.perf_counter() - t0)
        return batch

    def _sample_and_patch(self, batch: Batch) -> Batch:
        return self._patch_sampled_batch(self._sample_batch(batch))

    def _sample_batch(self, batch: Batch) -> tuple[Batch, Any]:
        t_total = time.perf_counter()
        t_req = time.perf_counter()
        request = _sampling_request_from_batch(batch)
        self._profile_stats["backend_sampling_request_seconds"] += float(time.perf_counter() - t_req)
        t_native = time.perf_counter()
        out = batch, self.sampler.sample(request)
        native_seconds = float(time.perf_counter() - t_native)
        self._profile_stats["backend_sampling_native_seconds"] += native_seconds
        self._profile_stats["backend_sampling_seconds"] += float(time.perf_counter() - t_total)
        return out

    def _patch_sampled_batch(self, sampled: tuple[Batch, Any]) -> Batch:
        batch, output = sampled
        t_submit = time.perf_counter()
        t_remap = time.perf_counter()
        _remap_batch_root_indices(batch, output)
        self._profile_stats["backend_root_remap_seconds"] += float(time.perf_counter() - t_remap)
        reads = self._submit_runtime_reads(output)
        self._profile_stats["backend_submit_reads_seconds"] += float(time.perf_counter() - t_submit)
        t_mat = time.perf_counter()
        batch.graph = _materialize_mfgs(output)
        batch.graph = _move_mfgs_to_device(batch.graph, self.device)
        self._profile_stats["backend_materialize_seconds"] += float(time.perf_counter() - t_mat)
        t_wait = time.perf_counter()
        self._wait_and_patch_runtime_reads(batch.graph, reads)
        self._profile_stats["backend_wait_patch_seconds"] += float(time.perf_counter() - t_wait)
        self._profile_stats["backend_batches"] += 1.0
        return batch

    def _submit_runtime_reads(self, output: Any) -> dict[str, tuple[Any, Any, Any]]:
        reads: dict[str, tuple[Any, Any, Any]] = {}
        if self.feature_runtime is not None:
            t0 = time.perf_counter()
            layout = self.feature_runtime.build_layout_from_sampling(output)
            self._profile_stats["backend_submit_node_feature_seconds"] += float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            edge_layout = self.feature_runtime.build_edge_layout_from_sampling(output)
            self._profile_stats["backend_submit_edge_feature_seconds"] += float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            reads["node_feature"] = (self.feature_runtime, layout, self.feature_runtime.submit_fetch(layout))
            self._profile_stats["backend_submit_node_feature_seconds"] += float(time.perf_counter() - t0)
            if edge_layout is not None:
                t0 = time.perf_counter()
                reads["edge_feature"] = (self.feature_runtime, edge_layout, self.feature_runtime.submit_edge_fetch(edge_layout))
                self._profile_stats["backend_submit_edge_feature_seconds"] += float(time.perf_counter() - t0)
        if self.memory_runtime is not None:
            t0 = time.perf_counter()
            layout = self.memory_runtime.build_read_layout_from_sampling(output)
            self._profile_stats["backend_submit_memory_seconds"] += float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            reads["memory"] = (self.memory_runtime, layout, self.memory_runtime.submit_read(layout))
            self._profile_stats["backend_submit_memory_seconds"] += float(time.perf_counter() - t0)
        if self.mailbox_runtime is not None:
            t0 = time.perf_counter()
            layout = self.mailbox_runtime.build_read_layout_from_sampling(output)
            self._profile_stats["backend_submit_mailbox_seconds"] += float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            reads["mailbox"] = (self.mailbox_runtime, layout, self.mailbox_runtime.submit_read(layout))
            self._profile_stats["backend_submit_mailbox_seconds"] += float(time.perf_counter() - t0)
        return reads

    def _wait_and_patch_runtime_reads(self, mfgs: Any, reads: dict[str, tuple[Any, Any, Any]]) -> None:
        feature = None
        edge_feature = None
        memory = None
        memory_ts = None
        mailbox = None
        mailbox_ts = None
        if "node_feature" in reads:
            runtime, layout, handle = reads["node_feature"]
            t0 = time.perf_counter()
            feature = runtime.wait_fetch(handle, layout)
            self._profile_stats["backend_wait_node_feature_seconds"] += float(time.perf_counter() - t0)
        if "edge_feature" in reads:
            runtime, layout, handle = reads["edge_feature"]
            t0 = time.perf_counter()
            edge_feature = runtime.wait_edge_fetch(handle, layout)
            self._profile_stats["backend_wait_edge_feature_seconds"] += float(time.perf_counter() - t0)
        if "memory" in reads:
            runtime, layout, handle = reads["memory"]
            t0 = time.perf_counter()
            memory, memory_ts = runtime.wait_read(handle, layout)
            self._profile_stats["backend_wait_memory_seconds"] += float(time.perf_counter() - t0)
        if "mailbox" in reads:
            runtime, layout, handle = reads["mailbox"]
            t0 = time.perf_counter()
            mailbox, mailbox_ts = runtime.wait_read(handle, layout)
            self._profile_stats["backend_wait_mailbox_seconds"] += float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _patch_first_layer_inputs(
            mfgs,
            node_feat=feature,
            edge_feat=edge_feature,
            edge_feat_dim=self.edge_feat_dim,
            memory=memory,
            memory_ts=memory_ts,
            mailbox=mailbox,
            mailbox_ts=mailbox_ts,
        )
        self._profile_stats["backend_patch_inputs_seconds"] += float(time.perf_counter() - t0)

    def _batch_from_event_positions(self, *, split: str, event_pos: torch.Tensor) -> Batch:
        t0 = time.perf_counter()
        src = self.graph_src.index_select(0, event_pos)
        dst = self.graph_dst.index_select(0, event_pos)
        ts = self.graph_ts.index_select(0, event_pos)
        eids = self.graph_eids.index_select(0, event_pos)
        edge_feat = _select_optional(self.graph_edge_feat, event_pos)
        num_edges = int(src.numel())
        roots = torch.cat([src, dst], dim=0).long().contiguous()
        root_ts = torch.cat([ts, ts], dim=0).contiguous()
        labels = _select_optional(self.graph_edge_label, event_pos)
        batch = Batch(
            split=split,
            roots=roots.to(self.device),
            timestamps=root_ts.to(self.device),
            graph=None,
            eids=eids.to(self.device),
            src=src.to(self.device),
            dst=dst.to(self.device),
            ts=ts.to(self.device),
            edge_feat=None if edge_feat is None else edge_feat.to(self.device),
            pos_src=torch.arange(num_edges, dtype=torch.long, device=self.device),
            pos_dst=torch.arange(num_edges, num_edges * 2, dtype=torch.long, device=self.device),
            labels=None if labels is None else labels.to(self.device),
        )
        self._profile_stats["backend_batch_build_seconds"] += float(time.perf_counter() - t0)
        return batch


def _load_torch_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("artifact path is required")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"artifact must be a dict: {path}")
    return obj


def _split_windows(graph: dict[str, Any], split: str) -> torch.Tensor:
    split_time_ptr = graph.get("split_time_ptr")
    if isinstance(split_time_ptr, dict) and split in split_time_ptr:
        return split_time_ptr[split].long().cpu().contiguous()
    if split == "train":
        return graph["time_ptr_2"].long().cpu().contiguous()
    return torch.zeros((0, 2), dtype=torch.long)


def _window_local_event_positions(local_edge_ids: torch.Tensor, *, begin: int, end: int) -> torch.Tensor:
    if end <= begin or local_edge_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    keep = (local_edge_ids >= int(begin)) & (local_edge_ids < int(end))
    return torch.sort(local_edge_ids[keep].long()).values.contiguous()


def _select_optional(tensor: Any, event_pos: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    return torch.as_tensor(tensor).cpu().index_select(0, event_pos).contiguous()


def _negative_dst_pool(*, graph: dict[str, Any], runtime_cfg: dict[str, Any]) -> torch.Tensor | None:
    policy = str(runtime_cfg.get("negative_dst_pool", runtime_cfg.get("negative_pool", "all_nodes"))).strip().lower()
    if policy in {"global_dst", "dst", "all_dst", "global-dst"}:
        return torch.unique(graph["dst"].long().cpu(), sorted=True).contiguous()
    if policy in {"none", "all_nodes", "all-nodes", "nodes"}:
        return None
    raw = runtime_cfg.get("dst_pool")
    if raw is not None:
        return torch.as_tensor(raw, dtype=torch.long).cpu().contiguous()
    raise ValueError(f"unsupported negative dst pool policy: {policy!r}")


def _build_negative_sampler(runtime_cfg: dict[str, Any]) -> NegativeSampler:
    policy = str(runtime_cfg.get("negative_sampler_policy", runtime_cfg.get("negative_policy", "uniform"))).strip().lower()
    if policy in {"local_remote", "rank_local_remote", "partition_local_remote"}:
        p = runtime_cfg.get("negative_train_local_dst_prob", runtime_cfg.get("train_local_dst_prob", runtime_cfg.get("negative_local_probability", None)))
        remote_p = runtime_cfg.get("negative_train_remote_dst_prob", runtime_cfg.get("train_remote_dst_prob", None))
        return PoolNegativeSampler(
            train_local_dst_prob=None if p is None else float(p),
            train_remote_dst_prob=0.0 if remote_p is None else float(remote_p),
            test_policy=str(runtime_cfg.get("negative_test_policy", "global")),
        )
    return RandomNegativeSampler()


def _rank_negative_dst_pools(
    *,
    graph: dict[str, Any],
    dist: dict[str, Any],
    rank: int,
    runtime_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    policy = str(runtime_cfg.get("negative_sampler_policy", runtime_cfg.get("negative_policy", "uniform"))).strip().lower()
    if policy not in {"local_remote", "rank_local_remote", "partition_local_remote"}:
        return None, None
    node_to_chunk = dist.get("node_to_chunk")
    chunk_owner = dist.get("chunk_owner")
    if node_to_chunk is not None and chunk_owner is not None:
        node_part = chunk_owner.long().cpu().index_select(0, node_to_chunk.long().cpu())
    else:
        node_part = dist.get("node_owner")
        if node_part is None:
            return None, None
        node_part = node_part.long().cpu()
    dst = torch.unique(graph["dst"].long().cpu(), sorted=True)
    dst_part = node_part.index_select(0, dst)
    local = dst[dst_part == int(rank)].long().contiguous()
    remote = dst[dst_part != int(rank)].long().contiguous()
    return local, remote


def _build_native_sampler(
    *,
    graph: dict[str, Any],
    dist: dict[str, Any],
    ctx: RuntimeContext,
    runtime_cfg: dict[str, Any],
) -> Any:
    edge_owner = dist.get("edge_owner")
    node_to_chunk = dist.get("node_to_chunk")
    chunk_owner = dist.get("chunk_owner")
    node_part = None
    if node_to_chunk is not None and chunk_owner is not None:
        node_part = chunk_owner.long().cpu().index_select(0, node_to_chunk.long().cpu()).to(torch.int32).contiguous()
    temporal_graph = TemporalGraphData(
        row=graph["src"].long().cpu().contiguous(),
        col=graph["dst"].long().cpu().contiguous(),
        edge_ids=graph.get("edge_ids", torch.arange(int(graph["src"].numel()), dtype=torch.long)).long().cpu().contiguous(),
        timestamps=graph.get("ts"),
        num_nodes=int(graph["num_nodes"]),
        node_part=node_part,
        edge_part=None if edge_owner is None else edge_owner.to(torch.int32).cpu().contiguous(),
    )
    config = NativeSamplerConfig(
        fanouts=tuple(int(v) for v in runtime_cfg.get("fanouts", (10,))),
        num_layers=int(runtime_cfg.get("num_layers", 1)),
        policy=_native_sampler_policy(str(runtime_cfg.get("policy", "recent"))),
        workers=int(runtime_cfg.get("sampler_workers", runtime_cfg.get("workers", 1))),
        local_part=int(ctx.rank),
    )
    probability = float(runtime_cfg.get("sample_probability", runtime_cfg.get("boundary_probability", 1.0)))
    graph_name = str(runtime_cfg.get("graph_name", "ctdg_events"))
    try:
        factory = MemShareNativeSamplerFactory(graph_name=graph_name, probability=probability)
    except TypeError:
        factory = MemShareNativeSamplerFactory(graph_name=graph_name)
    return factory.build(temporal_graph, config)


def _build_feature_runtime(
    *,
    rank_artifact: dict[str, Any],
    dist: dict[str, Any],
    feature_artifact: dict[str, Any] | None,
    ctx: RuntimeContext,
) -> CTDGFeatureRuntime:
    node_feat = None if feature_artifact is None else feature_artifact.get("node_feat")
    edge_feat = None if feature_artifact is None else feature_artifact.get("edge_feat")
    store = FeatureStore(node_features=node_feat, edge_features=edge_feat)
    index = DistIndexTables(
        master_dist_index=rank_artifact["read_dist_index"].long().cpu().contiguous(),
        read_dist_index=rank_artifact["read_dist_index"].long().cpu().contiguous(),
    )
    comm = DynamicFetchComm(torch.device(ctx.device))
    edge_dist_index = dist.get("edge_dist_index")
    if edge_dist_index is not None:
        edge_dist_index = edge_dist_index.long().cpu().contiguous()
    return CTDGFeatureRuntime(
        index=index,
        feature_store=store,
        comm=comm,
        world_size=int(ctx.world_size),
        edge_dist_index=edge_dist_index,
    )


def _native_sampler_policy(policy: str) -> str:
    policy = str(policy).strip().lower()
    if policy.startswith("boundary_"):
        return "boundery_" + policy[len("boundary_"):]
    return policy


def _build_dist_index_tables(*, rank_artifact: dict[str, Any], dist: dict[str, Any]) -> DistIndexTables:
    master = dist.get("master_dist_index")
    if master is None:
        master = rank_artifact["read_dist_index"]
    return DistIndexTables(
        master_dist_index=master.long().cpu().contiguous(),
        read_dist_index=rank_artifact["read_dist_index"].long().cpu().contiguous(),
    )


def _build_memory_runtime(
    *,
    rank_artifact: dict[str, Any],
    dist: dict[str, Any],
    ctx: RuntimeContext,
    runtime_cfg: dict[str, Any],
) -> MemoryRuntime:
    local_nodes = rank_artifact["local_node_ids"].long().cpu().contiguous()
    memory_cfg = dict(runtime_cfg.get("memory", {}))
    init_memory = runtime_cfg.get("memory_init", memory_cfg.get("init"))
    init_ts = runtime_cfg.get("memory_ts_init", memory_cfg.get("ts_init"))
    memory_dim = int(runtime_cfg.get("memory_dim", memory_cfg.get("dim", 0)))
    if init_memory is None:
        if memory_dim <= 0:
            raise ValueError("runtime.memory_dim (or runtime.memory.dim) is required to build memory runtime")
        memory = torch.zeros((int(local_nodes.numel()), memory_dim), dtype=torch.float32, device=torch.device(ctx.device))
    else:
        memory = torch.as_tensor(init_memory, dtype=torch.float32, device=torch.device(ctx.device)).contiguous()
    if init_ts is None:
        ts = torch.zeros((int(memory.size(0)),), dtype=torch.float32, device=memory.device)
    else:
        ts = torch.as_tensor(init_ts, dtype=torch.float32, device=memory.device).reshape(-1).contiguous()
    master = dist.get("master_dist_index")
    if master is None:
        master = rank_artifact["read_dist_index"]
    return MemoryRuntime(
        index=DistIndexTables(
            master_dist_index=master.long().cpu().contiguous(),
            read_dist_index=master.long().cpu().contiguous(),
        ),
        store=MemoryStore(memory, ts),
        fetch_comm=DynamicFetchComm(torch.device(ctx.device)),
        push_comm=DynamicPushComm(torch.device(ctx.device)),
        world_size=int(ctx.world_size),
    )


def _build_mailbox_runtime(
    *,
    rank_artifact: dict[str, Any],
    dist: dict[str, Any],
    ctx: RuntimeContext,
    runtime_cfg: dict[str, Any],
) -> MailboxRuntime:
    local_nodes = rank_artifact["local_node_ids"].long().cpu().contiguous()
    mailbox_cfg = dict(runtime_cfg.get("mailbox", {}))
    init_mailbox = runtime_cfg.get("mailbox_init", mailbox_cfg.get("init"))
    init_ts = runtime_cfg.get("mailbox_ts_init", mailbox_cfg.get("ts_init"))
    mailbox_size = int(runtime_cfg.get("mailbox_size", mailbox_cfg.get("size", 1)))
    msg_dim = int(runtime_cfg.get("mailbox_msg_dim", mailbox_cfg.get("msg_dim", 0)))
    if init_mailbox is None:
        if mailbox_size <= 0:
            raise ValueError("runtime.mailbox_size must be positive")
        if msg_dim <= 0:
            raise ValueError("runtime.mailbox_msg_dim (or runtime.mailbox.msg_dim) is required to build mailbox runtime")
        mailbox = torch.zeros(
            (int(local_nodes.numel()), mailbox_size, msg_dim),
            dtype=torch.float32,
            device=torch.device(ctx.device),
        )
    else:
        mailbox = torch.as_tensor(init_mailbox, dtype=torch.float32, device=torch.device(ctx.device)).contiguous()
    if init_ts is None:
        mailbox_ts = torch.zeros(mailbox.shape[:2], dtype=torch.float32, device=mailbox.device)
    else:
        mailbox_ts = torch.as_tensor(init_ts, dtype=torch.float32, device=mailbox.device).contiguous()
    next_pos = torch.zeros((int(mailbox.size(0)),), dtype=torch.long, device=mailbox.device)
    master = dist.get("master_dist_index")
    if master is None:
        master = rank_artifact["read_dist_index"]
    return MailboxRuntime(
        index=DistIndexTables(
            master_dist_index=master.long().cpu().contiguous(),
            read_dist_index=master.long().cpu().contiguous(),
        ),
        store=MailboxStore(mailbox, mailbox_ts, next_pos),
        fetch_comm=DynamicFetchComm(torch.device(ctx.device)),
        push_comm=DynamicPushComm(torch.device(ctx.device)),
        world_size=int(ctx.world_size),
    )


def _remap_batch_root_indices(batch: Batch, output: Any) -> None:
    root_lids = getattr(output.node_compute, "root_lids", None)
    groups = getattr(output.node_compute, "groups", {})
    if root_lids is None:
        return
    root_lids = root_lids.to(batch.roots.device).long().contiguous()
    if batch.pos_src is not None and "pos_src" in groups:
        begin, end = groups["pos_src"]
        batch.pos_src = root_lids[int(begin) : int(end)].contiguous()
    if batch.pos_dst is not None and "pos_dst" in groups:
        begin, end = groups["pos_dst"]
        batch.pos_dst = root_lids[int(begin) : int(end)].contiguous()
    if batch.neg_dst is not None and "neg_dst" in groups:
        begin, end = groups["neg_dst"]
        batch.neg_dst = root_lids[int(begin) : int(end)].contiguous()


def _materialize_mfgs(output: Any) -> Any:
    mfgs = getattr(output, "mfgs", output)
    if not mfgs or hasattr(mfgs[0], "srcdata"):
        return mfgs
    if not hasattr(mfgs[0], "csc_indptr"):
        return mfgs
    import dgl

    node_gids = output.node_compute.node_gids.long().cpu()
    node_ts = output.node_compute.node_ts
    if node_ts is not None:
        node_ts = node_ts.cpu()
    edge_gids = None if output.edge_compute is None else output.edge_compute.edge_gids.long().cpu()
    blocks = []
    for mfg in mfgs:
        indptr = mfg.csc_indptr.long().cpu().contiguous()
        indices = mfg.csc_indices.long().cpu().contiguous()
        old_src_lids = mfg.src_lids.long().cpu()
        dst_lids = mfg.dst_lids.long().cpu()
        src_lids, indices = _ensure_dst_prefix_src_lids(
            old_src_lids=old_src_lids,
            dst_lids=dst_lids,
            indices=indices,
        )
        dgl_eids = torch.arange(int(indices.numel()), dtype=torch.long)
        block = dgl.create_block(
            ("csc", (indptr, indices, dgl_eids)),
            num_src_nodes=int(src_lids.numel()),
            num_dst_nodes=int(dst_lids.numel()),
        )
        block.srcdata["__ID"] = src_lids
        block.dstdata["__ID"] = dst_lids
        block.srcdata["ID"] = node_gids.index_select(0, src_lids)
        block.dstdata["ID"] = node_gids.index_select(0, dst_lids)
        if node_ts is not None:
            block.srcdata["ts"] = node_ts.index_select(0, src_lids)
            block.dstdata["ts"] = node_ts.index_select(0, dst_lids)
        if mfg.delta_t is not None:
            block.edata["dt"] = mfg.delta_t.cpu()
        if edge_gids is not None and mfg.edge_lids.numel() > 0:
            edge_lids = mfg.edge_lids.long().cpu()
            block.edata["__ID"] = edge_lids
            block.edata["ID"] = edge_gids.index_select(0, edge_lids)
        blocks.append([block])
    return blocks


def _move_mfgs_to_device(mfgs: Any, device: torch.device) -> Any:
    if mfgs is None:
        return None
    if isinstance(mfgs, list):
        return [_move_mfgs_to_device(item, device) for item in mfgs]
    if isinstance(mfgs, tuple):
        return tuple(_move_mfgs_to_device(item, device) for item in mfgs)
    if hasattr(mfgs, "to"):
        return mfgs.to(device)
    return mfgs


def _ensure_dst_prefix_src_lids(*, old_src_lids: torch.Tensor, dst_lids: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(old_src_lids.numel()) >= int(dst_lids.numel()) and torch.equal(old_src_lids[: int(dst_lids.numel())], dst_lids):
        return old_src_lids, indices
    if old_src_lids.numel() == 0:
        return dst_lids.long().contiguous(), indices.long().contiguous()
    if dst_lids.numel() == 0:
        return old_src_lids.long().contiguous(), indices.long().contiguous()
    order = torch.argsort(dst_lids, stable=True)
    sorted_dst = dst_lids.index_select(0, order)
    lookup = torch.searchsorted(sorted_dst, old_src_lids.long())
    pos = lookup.clamp_max(max(int(sorted_dst.numel()) - 1, 0))
    in_dst = (lookup < int(sorted_dst.numel())) & (sorted_dst.index_select(0, pos) == old_src_lids.long())
    tail = old_src_lids[~in_dst].long().contiguous()
    src_lids = torch.cat([dst_lids, tail], dim=0).long().contiguous()
    old_to_new = torch.empty_like(old_src_lids, dtype=torch.long)
    old_to_new[in_dst] = order.index_select(0, lookup[in_dst]).long()
    old_to_new[~in_dst] = int(dst_lids.numel()) + torch.arange(int(tail.numel()), dtype=torch.long)
    return src_lids, old_to_new.index_select(0, indices.long()).long().contiguous()


def _patch_first_layer_inputs(
    mfgs: Any,
    *,
    node_feat: torch.Tensor | None,
    edge_feat: torch.Tensor | None,
    edge_feat_dim: int = 0,
    memory: torch.Tensor | None,
    memory_ts: torch.Tensor | None,
    mailbox: torch.Tensor | None,
    mailbox_ts: torch.Tensor | None,
) -> None:
    if not mfgs:
        return
    if isinstance(mfgs, (list, tuple)) and mfgs and hasattr(mfgs[0], "srcdata"):
        first_layer = mfgs
    else:
        first_layer = mfgs[0] if isinstance(mfgs, (list, tuple)) else [mfgs]
    for block in first_layer:
        srcdata = getattr(block, "srcdata", None)
        if srcdata is None or "__ID" not in srcdata:
            continue
        idx = srcdata["__ID"].long()
        device = idx.device
        if node_feat is not None:
            srcdata["h"] = node_feat.index_select(0, idx.to(node_feat.device)).to(device)
        if memory is not None:
            srcdata["mem"] = memory.index_select(0, idx.to(memory.device)).to(device)
        if memory_ts is not None:
            srcdata["mem_ts"] = memory_ts.index_select(0, idx.to(memory_ts.device)).to(device)
        if mailbox is not None:
            mail = mailbox.index_select(0, idx.to(mailbox.device)).to(device)
            srcdata["mem_input"] = mail.reshape(mail.size(0), -1)
        if mailbox_ts is not None:
            srcdata["mail_ts"] = mailbox_ts.index_select(0, idx.to(mailbox_ts.device)).to(device)
    if edge_feat is not None:
        for block in _flatten_mfg_blocks(mfgs):
            edata = getattr(block, "edata", None)
            if edata is None or "__ID" not in edata:
                continue
            idx = edata["__ID"].long()
            edata["f"] = edge_feat.index_select(0, idx.to(edge_feat.device)).to(idx.device)
    if edge_feat_dim > 0:
        for block in _flatten_mfg_blocks(mfgs):
            edata = getattr(block, "edata", None)
            if edata is not None and "f" not in edata:
                edata["f"] = torch.zeros(
                    (int(block.num_edges()), int(edge_feat_dim)),
                    dtype=torch.float32,
                    device=block.device,
                )


def _flatten_mfg_blocks(mfgs: Any) -> list[Any]:
    if mfgs is None:
        return []
    if isinstance(mfgs, (list, tuple)):
        out: list[Any] = []
        for item in mfgs:
            out.extend(_flatten_mfg_blocks(item))
        return out
    return [mfgs]


def _feature_dim(feature: Any) -> int:
    if feature is None:
        return 0
    tensor = torch.as_tensor(feature)
    if tensor.dim() == 0:
        return 0
    return int(tensor.size(-1)) if tensor.dim() > 1 else 1


def _sampling_request_from_batch(batch: Batch) -> TemporalSamplingRequest:
    groups: dict[str, tuple[int, int]] = {}
    if batch.pos_src is not None:
        groups["pos_src"] = _range_from_index(batch.pos_src)
    if batch.pos_dst is not None:
        groups["pos_dst"] = _range_from_index(batch.pos_dst)
    if batch.neg_dst is not None:
        groups["neg_dst"] = _range_from_index(batch.neg_dst)
    roots = RootSet(
        nodes=batch.roots.long().detach().cpu().contiguous(),
        ts=None if batch.timestamps is None else batch.timestamps.detach().cpu().contiguous(),
        groups=groups,
    )
    return TemporalSamplingRequest(
        roots=roots,
        fanouts=(),
        num_layers=0,
        policy="runtime",
        meta={"split": batch.split},
    )


def _range_from_index(index: torch.Tensor) -> tuple[int, int]:
    if int(index.numel()) == 0:
        return (0, 0)
    start = int(index[0].item())
    return (start, start + int(index.numel()))
