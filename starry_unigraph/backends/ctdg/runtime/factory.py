"""Factory for building a CTDGOnlineRuntime from a session config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from starry_unigraph.types import RuntimeBundle, SessionContext
from .cache import AdaParameter, CTDGHistoricalCache
from .data import TGTemporalDataset
from .memory import CTDGMemoryBank
from .models import CTDGLinkPredictor, CTDGMemoryUpdater
from .route import CTDGFeatureRoute
from .runtime import CTDGOnlineRuntime
from .sampler import NativeTemporalSampler


def build_ctdg_runtime(session_ctx: SessionContext) -> tuple[CTDGOnlineRuntime, RuntimeBundle]:
    """Build a :class:`CTDGOnlineRuntime` and a :class:`RuntimeBundle` from config.

    Returns:
        ``(online_runtime, runtime_bundle)`` — the runtime contains
        ``model`` and ``optimizer`` fields; ``online_runtime`` exposes
        ``iter_batches``, ``train_step``, ``eval_step``, ``predict_step``.
    """
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

    mailbox_slot_dim = 2 * hidden_dim + edge_feat_dim
    memory_updater = CTDGMemoryUpdater(
        hidden_dim=hidden_dim,
        mailbox_slot_dim=mailbox_slot_dim,
        mailbox_slots=mailbox_slots,
    ).to(device)

    try:
        import torch.distributed as dist
        if dist_ctx.is_distributed and dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[dist_ctx.local_rank], find_unused_parameters=True)
    except Exception:
        pass

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

    online_runtime = CTDGOnlineRuntime(
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
    runtime = RuntimeBundle(
        model=model,
        optimizer=optimizer,
        state={
            "pipeline": "ctdg_online_native",
            "dataset": dataset.describe(),
            "memory_state": memory.describe(),
            "route_state": route.describe(),
        },
    )
    return online_runtime, runtime
