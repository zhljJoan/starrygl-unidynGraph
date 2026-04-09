from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from starry_unigraph.backends.ctdg import CTDGFeatureRoute, CTDGLinkPredictor, CTDGMemoryBank, NativeTemporalSampler, TGTemporalDataset
from starry_unigraph.backends.ctdg.historical_cache import AdaParameter, CTDGHistoricalCache
from starry_unigraph.backends.ctdg.model import CTDGMemoryUpdater
from starry_unigraph.backends.ctdg.runtime import CTDGOnlineRuntime
from starry_unigraph.preprocess import ArtifactOutput, ArtifactPayload, GraphPreprocessor
from starry_unigraph.providers.common import ARTIFACT_VERSION, BaseProvider
from starry_unigraph.registry import ProviderRegistry
from starry_unigraph.types import PreparedArtifacts, RuntimeBundle, SessionContext


class CTDGPreprocessor(GraphPreprocessor):
    graph_mode = "ctdg"

    def prepare_raw(self, session_ctx: SessionContext) -> None:
        dataset_root = session_ctx.dataset_path or Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        dataset = TGTemporalDataset(
            dataset_root,
            session_ctx.config["data"]["name"],
            split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
            config=session_ctx.config,
        )
        session_ctx.provider_state["ctdg_dataset_stats"] = dataset.describe()

    def build_partitions(self, session_ctx: SessionContext) -> None:
        stats = session_ctx.provider_state["ctdg_dataset_stats"]
        session_ctx.provider_state["partition_manifest"] = {
            "graph_mode": "ctdg",
            "num_parts": int(session_ctx.config["dist"]["world_size"]),
            "partition_algo": str(session_ctx.config["graph"]["partition"]),
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
        }

    def build_runtime_artifacts(self, session_ctx: SessionContext) -> PreparedArtifacts:
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
        }
        return self.emit_artifacts(
            session_ctx,
            ArtifactPayload(
                provider_meta=provider_meta,
                outputs=[
                    ArtifactOutput("partitions/manifest.json", session_ctx.provider_state["partition_manifest"]),
                    ArtifactOutput("routes/manifest.json", feature_route.describe()),
                    ArtifactOutput("sampling/index.json", {"dataset": session_ctx.config["data"]["name"], **stats}),
                ],
            ),
        )


class CTDGProvider(BaseProvider):
    graph_mode = "ctdg"
    provider_key = "ctdg"

    def __init__(self, task_adapter: Any):
        super().__init__(task_adapter=task_adapter)
        self.preprocessor = CTDGPreprocessor()
        self.online_runtime: CTDGOnlineRuntime | None = None
        self._memory_updater: CTDGMemoryUpdater | None = None

    def prepare_data(self, session_ctx: SessionContext) -> PreparedArtifacts:
        self.prepared = self.preprocessor.run(session_ctx)
        session_ctx.provider_state["prepared_meta"] = self.prepared.provider_meta
        return self.prepared

    def build_runtime(self, session_ctx: SessionContext) -> RuntimeBundle:
        runtime = self._build_runtime_common(session_ctx)
        dataset_root = session_ctx.dataset_path or Path(session_ctx.config["data"]["root"]).expanduser().resolve()
        dataset = TGTemporalDataset(
            dataset_root,
            session_ctx.config["data"]["name"],
            split_ratio=session_ctx.config.get("data", {}).get("split_ratio"),
            config=session_ctx.config,
        )
        device = str(session_ctx.config["runtime"]["device"])
        hidden_dim = int(session_ctx.config["model"]["hidden_dim"])
        dist_ctx = session_ctx.dist

        # Read CTDG-specific config
        ctdg_cfg = session_ctx.config.get("ctdg", {})
        mailbox_slots = int(ctdg_cfg.get("mailbox_slots", 4))
        historical_alpha = float(ctdg_cfg.get("historical_alpha", 0.5))
        async_sync = bool(ctdg_cfg.get("async_sync", True))
        ada_param_enabled = bool(ctdg_cfg.get("ada_param_enabled", True))
        dim_time = int(ctdg_cfg.get("dim_time", 100))
        num_head = int(ctdg_cfg.get("num_head", 2))
        dropout = float(ctdg_cfg.get("dropout", 0.1))
        att_dropout = float(ctdg_cfg.get("att_dropout", 0.1))

        sampler = NativeTemporalSampler(
            dataset=dataset,
            fanout=list(session_ctx.config["sampling"]["neighbor_limit"]),
            history=int(session_ctx.config["sampling"]["history"]),
            strategy=str(session_ctx.config["sampling"]["strategy"]),
            workers=max(1, min(4, dist_ctx.local_world_size or 1)),
        )

        edge_feat_dim = max(1, dataset.edge_feat_dim)
        memory = CTDGMemoryBank(
            num_nodes=dataset.num_nodes,
            hidden_dim=hidden_dim,
            device=device,
            mailbox_slots=mailbox_slots,
            edge_feat_dim=edge_feat_dim,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            async_sync=async_sync,
        )

        model = CTDGLinkPredictor(
            num_nodes=dataset.num_nodes,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            dim_time=dim_time,
            num_head=num_head,
            dropout=dropout,
            att_dropout=att_dropout,
        ).to(device)

        # Build CTDGMemoryUpdater
        mailbox_slot_dim = 2 * hidden_dim + edge_feat_dim
        memory_updater = CTDGMemoryUpdater(
            hidden_dim=hidden_dim,
            mailbox_slot_dim=mailbox_slot_dim,
            mailbox_slots=mailbox_slots,
        ).to(device)
        self._memory_updater = memory_updater

        # DDP wrap model (not memory_updater — synced manually via broadcast)
        try:
            import torch.distributed as dist
            if dist_ctx.is_distributed and dist.is_initialized():
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, device_ids=[dist_ctx.local_rank], find_unused_parameters=True)
        except Exception:
            pass

        # Unified optimizer (model + memory_updater)
        all_params = list(
            (model.module if hasattr(model, "module") else model).parameters()
        ) + list(memory_updater.parameters())
        optimizer = torch.optim.Adam(
            all_params,
            lr=float(session_ctx.config.get("train", {}).get("lr", 1e-3)),
        )

        route = CTDGFeatureRoute(
            route_type=str(session_ctx.config["graph"]["route"]),
            world_size=dist_ctx.world_size,
        )

        # Initialize historical cache for distributed shared-node tracking
        try:
            import torch.distributed as dist
            if dist_ctx.is_distributed and dist.is_initialized():
                node_part = torch.arange(dataset.num_nodes, dtype=torch.long) % dist_ctx.world_size
                shared_nodes = torch.where(node_part != dist_ctx.rank)[0].to(device)
                ada = AdaParameter(alpha=historical_alpha) if ada_param_enabled else None
                memory.historical_cache = CTDGHistoricalCache(
                    num_shared=shared_nodes.numel(),
                    hidden_dim=hidden_dim,
                    device=device,
                    num_nodes=dataset.num_nodes,
                    shared_node_ids=shared_nodes,
                    ada_param=ada,
                )
        except Exception:
            pass

        self.online_runtime = CTDGOnlineRuntime(
            dataset=dataset,
            sampler=sampler,
            memory=memory,
            model=model,
            optimizer=optimizer,
            route=route,
            device=device,
            dist_ctx=dist_ctx,
            memory_updater=memory_updater,
        )
        runtime.model = model
        runtime.optimizer = optimizer
        runtime.state.update(
            {
                "dataset": dataset.describe(),
                "memory_state": memory.describe(),
                "route_state": route.describe(),
                "pipeline": "ctdg_online_native",
            }
        )
        return runtime

    def build_train_iterator(self, session_ctx: SessionContext, split: str = "train") -> Iterable[Any]:
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(split=split, batch_size=int(session_ctx.config["train"]["batch_size"]))

    def build_eval_iterator(self, session_ctx: SessionContext, split: str = "val") -> Iterable[Any]:
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(split=split, batch_size=int(session_ctx.config["train"]["batch_size"]))

    def build_predict_iterator(self, session_ctx: SessionContext, split: str = "test") -> Iterable[Any]:
        assert self.online_runtime is not None
        yield from self.online_runtime.iter_batches(split=split, batch_size=int(session_ctx.config["train"]["batch_size"]))

    def run_train_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        output = self.online_runtime.train_step(batch)
        self.runtime.state["memory_state"] = self.online_runtime.memory.describe()
        return output

    def run_eval_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        return self.online_runtime.eval_step(batch, split=batch.split)

    def run_predict_step(self, batch: Any) -> dict[str, Any]:
        assert self.online_runtime is not None
        return self.online_runtime.predict_step(batch)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model + memory_updater state dicts."""
        assert self.online_runtime is not None
        model = self.online_runtime.model
        model_state = (model.module if hasattr(model, "module") else model).state_dict()
        ckpt: dict[str, Any] = {"model": model_state}
        if self._memory_updater is not None:
            ckpt["memory_updater"] = self._memory_updater.state_dict()
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model + memory_updater state dicts."""
        assert self.online_runtime is not None
        ckpt = torch.load(str(path), map_location="cpu")
        model = self.online_runtime.model
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        if self._memory_updater is not None and "memory_updater" in ckpt:
            self._memory_updater.load_state_dict(ckpt["memory_updater"])


ProviderRegistry.register("ctdg", CTDGProvider)
