from __future__ import annotations

from typing import Any


MEMORY_SYNC_MODE_LEGACY = "legacy"
MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_EXACT = "memshare_public_exact"
MEMORY_SYNC_MODE_MEMSHARE_HISTORICAL = "memshare_historical"
MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_HISTORICAL = "memshare_public_historical"
DEFAULT_MEMORY_SYNC_MODE = MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_EXACT


def resolve_memory_sync_mode(runtime_cfg: dict[str, Any] | None) -> dict[str, Any]:
    runtime_cfg = dict(runtime_cfg or {})
    historical_cfg = dict(runtime_cfg.get("historical", {})) if isinstance(runtime_cfg.get("historical", {}), dict) else {}
    async_memory_cfg = (
        dict(runtime_cfg.get("async_memory", {}))
        if isinstance(runtime_cfg.get("async_memory", {}), dict)
        else {}
    )

    raw_mode = runtime_cfg.get("memory_sync_mode", runtime_cfg.get("sync_mode", DEFAULT_MEMORY_SYNC_MODE))
    mode = str(raw_mode).strip().lower()
    if not mode:
        mode = MEMORY_SYNC_MODE_LEGACY

    is_memshare_public_exact = mode in {
        MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_EXACT,
        "memshare-exact",
        "memshare-public",
    }
    is_memshare_historical = mode in {MEMORY_SYNC_MODE_MEMSHARE_HISTORICAL, "historical", "memshare-historical"}
    is_memshare_public_historical = mode in {
        MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_HISTORICAL,
        "memshare-public-historical",
        "memshare-exact-historical",
        "memshare_public_exact_historical",
    }

    shared_filter = bool(async_memory_cfg.get("shared_filter", historical_cfg.get("enabled", False)))
    staged_commit = bool(async_memory_cfg.get("staged_commit", False))
    delta_compensation = bool(async_memory_cfg.get("delta_compensation", False))
    preload_candidate_delta = bool(async_memory_cfg.get("preload_candidate_delta", False))
    historical_blend = bool(historical_cfg.get("enabled", False)) or bool(async_memory_cfg.get("historical_blend", False))
    wait_mode = str(runtime_cfg.get("wait_mode", async_memory_cfg.get("commit_order", MEMORY_SYNC_MODE_LEGACY))).strip().lower()
    if not wait_mode:
        wait_mode = MEMORY_SYNC_MODE_LEGACY
    schedule_async_commit = runtime_cfg.get("schedule_async_commit")
    historical_enabled = bool(historical_cfg.get("enabled", False)) or bool(shared_filter)
    preserve_replica_history = historical_enabled or delta_compensation
    memory_replica_push = bool(runtime_cfg.get("memory_replica_push", False))
    mailbox_replica_push = bool(runtime_cfg.get("mailbox_replica_push", memory_replica_push))

    if is_memshare_public_historical:
        historical_enabled = True
        shared_filter = True
        staged_commit = True
        delta_compensation = True
        historical_blend = True
        preload_candidate_delta = bool(async_memory_cfg.get("preload_candidate_delta", True))
        wait_mode = "memshare"
        preserve_replica_history = True
        memory_replica_push = bool(runtime_cfg.get("memory_replica_push", True))
        mailbox_replica_push = bool(runtime_cfg.get("mailbox_replica_push", True))
        if schedule_async_commit is None:
            schedule_async_commit = True
        mode = MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_HISTORICAL
    elif is_memshare_public_exact:
        historical_enabled = True
        shared_filter = True
        historical_blend = False
        preload_candidate_delta = bool(async_memory_cfg.get("preload_candidate_delta", False))
        preserve_replica_history = True
        memory_replica_push = bool(runtime_cfg.get("memory_replica_push", True))
        mailbox_replica_push = bool(runtime_cfg.get("mailbox_replica_push", True))
        if schedule_async_commit is None:
            schedule_async_commit = True
        mode = MEMORY_SYNC_MODE_MEMSHARE_PUBLIC_EXACT
    elif is_memshare_historical:
        historical_enabled = True
        shared_filter = True
        staged_commit = True
        delta_compensation = True
        historical_blend = True
        preload_candidate_delta = bool(async_memory_cfg.get("preload_candidate_delta", True))
        wait_mode = "memshare"
        preserve_replica_history = True
        memory_replica_push = bool(runtime_cfg.get("memory_replica_push", True))
        mailbox_replica_push = bool(runtime_cfg.get("mailbox_replica_push", True))
        if schedule_async_commit is None:
            schedule_async_commit = True
        mode = MEMORY_SYNC_MODE_MEMSHARE_HISTORICAL
    elif schedule_async_commit is None:
        schedule_async_commit = False

    return {
        "mode": mode,
        "historical_enabled": historical_enabled,
        "shared_filter": shared_filter,
        "staged_commit": staged_commit,
        "delta_compensation": delta_compensation,
        "preload_candidate_delta": preload_candidate_delta,
        "historical_blend": historical_blend,
        "wait_mode": wait_mode,
        "schedule_async_commit": bool(schedule_async_commit),
        "preserve_replica_history": preserve_replica_history,
        "memory_replica_push": memory_replica_push,
        "mailbox_replica_push": mailbox_replica_push,
    }


def normalize_memory_sync_config(runtime_cfg: dict[str, Any] | None) -> dict[str, Any]:
    runtime_cfg = dict(runtime_cfg or {})
    historical_cfg = dict(runtime_cfg.get("historical", {})) if isinstance(runtime_cfg.get("historical", {}), dict) else {}
    async_memory_cfg = (
        dict(runtime_cfg.get("async_memory", {}))
        if isinstance(runtime_cfg.get("async_memory", {}), dict)
        else {}
    )
    sync_cfg = resolve_memory_sync_mode(runtime_cfg)

    historical_cfg["enabled"] = bool(sync_cfg["historical_enabled"])
    async_memory_cfg["shared_filter"] = bool(sync_cfg["shared_filter"])
    async_memory_cfg["staged_commit"] = bool(sync_cfg["staged_commit"])
    async_memory_cfg["delta_compensation"] = bool(sync_cfg["delta_compensation"])
    async_memory_cfg["preload_candidate_delta"] = bool(sync_cfg["preload_candidate_delta"])
    async_memory_cfg["historical_blend"] = bool(sync_cfg["historical_blend"])
    async_memory_cfg["commit_order"] = str(sync_cfg["wait_mode"])

    runtime_cfg["memory_sync_mode"] = str(sync_cfg["mode"])
    runtime_cfg["wait_mode"] = str(sync_cfg["wait_mode"])
    if "schedule_async_commit" not in runtime_cfg:
        runtime_cfg["schedule_async_commit"] = bool(sync_cfg["schedule_async_commit"])
    runtime_cfg["preserve_replica_history"] = bool(sync_cfg["preserve_replica_history"])
    runtime_cfg["memory_replica_push"] = bool(sync_cfg["memory_replica_push"])
    runtime_cfg["mailbox_replica_push"] = bool(sync_cfg["mailbox_replica_push"])
    runtime_cfg["historical"] = historical_cfg
    runtime_cfg["async_memory"] = async_memory_cfg
    return runtime_cfg
