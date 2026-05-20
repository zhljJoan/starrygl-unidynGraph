from __future__ import annotations

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
from atc_starrygl_lib.sampling.negative import NegativeSampler, NegativeSamplingRequest, RandomNegativeSampler
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
        if self._prepared_by == "new_pipeline":
            self._runtime = _CTDGArtifactRuntime.from_artifacts(ctx, artifacts)
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
        prefetch_batches: bool = True,
    ) -> None:
        self.graph = graph
        self.dist = dist
        self.rank_artifact = rank_artifact
        self.feature_artifact = feature_artifact
        self.device = device
        self.task_name = task_name
        self.node_batch_size = int(node_batch_size)
        self.sampler = sampler
        self.feature_runtime = feature_runtime
        self.memory_runtime = memory_runtime
        self.mailbox_runtime = mailbox_runtime
        self.negative_sampler = negative_sampler
        self.negative_ratio = int(negative_ratio)
        self.prefetch_batches = bool(prefetch_batches)
        self.local_edge_ids = rank_artifact["local_edge_ids"].long().cpu().contiguous()

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
            negative_sampler = RandomNegativeSampler()
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
            prefetch_batches=bool(runtime_cfg.get("prefetch_batches", True)),
        )

    def iter_batches(self, split: str) -> Iterator[Batch]:
        if self.task_name.startswith("node_") or self.task_name in {"node_prediction", "node_regression"}:
            batches = self._iter_node_batches(str(split))
            yield from self._maybe_sample_batches(batches)
            return
        yield from self._maybe_sample_batches(self._iter_edge_batches(str(split)))

    def _iter_edge_batches(self, split: str) -> Iterator[Batch]:
        split = str(split)
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

    def _attach_negative(self, batch: Batch) -> Batch:
        if self.negative_sampler is None or self.negative_ratio <= 0:
            return batch
        if batch.src is None or batch.dst is None or batch.ts is None:
            return batch
        result = self.negative_sampler.sample(
            NegativeSamplingRequest(
                pos_src=batch.src,
                pos_dst=batch.dst,
                num_nodes=int(self.graph["num_nodes"]),
                ratio=int(self.negative_ratio),
                split=batch.split,
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
        neg_ts = batch.ts.repeat_interleave(int(result.ratio)).to(batch.timestamps.device)
        batch.timestamps = torch.cat([batch.timestamps, neg_ts], dim=0).contiguous()
        return batch

    def _sample_and_patch(self, batch: Batch) -> Batch:
        return self._patch_sampled_batch(self._sample_batch(batch))

    def _sample_batch(self, batch: Batch) -> tuple[Batch, Any]:
        request = _sampling_request_from_batch(batch)
        return batch, self.sampler.sample(request)

    def _patch_sampled_batch(self, sampled: tuple[Batch, Any]) -> Batch:
        batch, output = sampled
        _remap_batch_root_indices(batch, output)
        reads = self._submit_runtime_reads(output)
        batch.graph = _materialize_mfgs(output)
        self._wait_and_patch_runtime_reads(batch.graph, reads)
        return batch

    def _submit_runtime_reads(self, output: Any) -> dict[str, tuple[Any, Any, Any]]:
        reads: dict[str, tuple[Any, Any, Any]] = {}
        if self.feature_runtime is not None:
            layout = self.feature_runtime.build_layout_from_sampling(output)
            edge_layout = self.feature_runtime.build_edge_layout_from_sampling(output)
            reads["node_feature"] = (self.feature_runtime, layout, self.feature_runtime.submit_fetch(layout))
            if edge_layout is not None:
                reads["edge_feature"] = (self.feature_runtime, edge_layout, self.feature_runtime.submit_edge_fetch(edge_layout))
        if self.memory_runtime is not None:
            layout = self.memory_runtime.build_read_layout_from_sampling(output)
            reads["memory"] = (self.memory_runtime, layout, self.memory_runtime.submit_read(layout))
        if self.mailbox_runtime is not None:
            layout = self.mailbox_runtime.build_read_layout_from_sampling(output)
            reads["mailbox"] = (self.mailbox_runtime, layout, self.mailbox_runtime.submit_read(layout))
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
            feature = runtime.wait_fetch(handle, layout)
        if "edge_feature" in reads:
            runtime, layout, handle = reads["edge_feature"]
            edge_feature = runtime.wait_edge_fetch(handle, layout)
        if "memory" in reads:
            runtime, layout, handle = reads["memory"]
            memory, memory_ts = runtime.wait_read(handle, layout)
        if "mailbox" in reads:
            runtime, layout, handle = reads["mailbox"]
            mailbox, mailbox_ts = runtime.wait_read(handle, layout)
        _patch_first_layer_inputs(
            mfgs,
            node_feat=feature,
            edge_feat=edge_feature,
            memory=memory,
            memory_ts=memory_ts,
            mailbox=mailbox,
            mailbox_ts=mailbox_ts,
        )

    def _batch_from_event_positions(self, *, split: str, event_pos: torch.Tensor) -> Batch:
        graph = self.graph
        src = graph["src"].long().cpu().index_select(0, event_pos)
        dst = graph["dst"].long().cpu().index_select(0, event_pos)
        ts = graph["ts"].cpu().index_select(0, event_pos)
        eids = graph.get("edge_ids", torch.arange(int(graph["src"].numel()), dtype=torch.long)).long().cpu().index_select(0, event_pos)
        num_edges = int(src.numel())
        roots = torch.cat([src, dst], dim=0).long().contiguous()
        root_ts = torch.cat([ts, ts], dim=0).contiguous()
        labels = _select_optional(graph.get("edge_label"), event_pos)
        return Batch(
            split=split,
            roots=roots.to(self.device),
            timestamps=root_ts.to(self.device),
            graph=None,
            eids=eids.to(self.device),
            src=src.to(self.device),
            dst=dst.to(self.device),
            ts=ts.to(self.device),
            pos_src=torch.arange(num_edges, dtype=torch.long, device=self.device),
            pos_dst=torch.arange(num_edges, num_edges * 2, dtype=torch.long, device=self.device),
            labels=None if labels is None else labels.to(self.device),
        )


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
        policy=str(runtime_cfg.get("policy", "recent")),
        workers=int(runtime_cfg.get("sampler_workers", runtime_cfg.get("workers", 1))),
        local_part=int(ctx.rank),
    )
    return MemShareNativeSamplerFactory(graph_name=str(runtime_cfg.get("graph_name", "ctdg_events"))).build(temporal_graph, config)


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
    return MemoryRuntime(
        index=_build_dist_index_tables(rank_artifact=rank_artifact, dist=dist),
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
    return MailboxRuntime(
        index=_build_dist_index_tables(rank_artifact=rank_artifact, dist=dist),
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


def _ensure_dst_prefix_src_lids(*, old_src_lids: torch.Tensor, dst_lids: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(old_src_lids.numel()) >= int(dst_lids.numel()) and torch.equal(old_src_lids[: int(dst_lids.numel())], dst_lids):
        return old_src_lids, indices
    dst_set = set(int(v) for v in dst_lids.tolist())
    tail = torch.tensor([int(v) for v in old_src_lids.tolist() if int(v) not in dst_set], dtype=torch.long)
    src_lids = torch.cat([dst_lids, tail], dim=0).long().contiguous()
    row_map = {int(v): i for i, v in enumerate(src_lids.tolist())}
    old_to_new = torch.tensor([row_map[int(v)] for v in old_src_lids.tolist()], dtype=torch.long)
    return src_lids, old_to_new.index_select(0, indices.long()).long().contiguous()


def _patch_first_layer_inputs(
    mfgs: Any,
    *,
    node_feat: torch.Tensor | None,
    edge_feat: torch.Tensor | None,
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
            if not isinstance(edata, dict) or "__ID" not in edata:
                continue
            idx = edata["__ID"].long()
            edata["f"] = edge_feat.index_select(0, idx.to(edge_feat.device)).to(idx.device)


def _flatten_mfg_blocks(mfgs: Any) -> list[Any]:
    if mfgs is None:
        return []
    if isinstance(mfgs, (list, tuple)):
        out: list[Any] = []
        for item in mfgs:
            out.extend(_flatten_mfg_blocks(item))
        return out
    return [mfgs]


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
