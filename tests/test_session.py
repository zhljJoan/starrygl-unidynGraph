from __future__ import annotations

import json
from pathlib import Path

import torch

from starry_unigraph import SchedulerSession
from starry_unigraph.native import is_bts_sampler_available


def make_config(tmp_path: Path, family: str, task: str, backend: str = "mock") -> dict:
    graph_mode = "ctdg" if family in {"tgn", "dyrep"} else "dtdg"
    storage = "events" if graph_mode == "ctdg" else "snapshots"
    return {
        "model": {
            "name": family,
            "family": family,
            "task": task,
            "hidden_dim": 32,
            "memory": {"type": "gru"},
            "window": {"size": 4},
        },
        "data": {
            "root": str(tmp_path / "data"),
            "name": f"{family}_dataset",
            "format": "mock",
            "graph_mode": graph_mode,
            "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "snaps": 4,
            "eval_interval": 1,
        },
        "runtime": {
            "backend": backend,
            "device": "cpu",
            "cache": "gpu_local",
            "state_sync": "versioned",
            "checkpoint": str(tmp_path / "ckpt.pkl"),
        },
        "dtdg": {
            "pipeline": "flare_native",
        },
        "ctdg": {
            "pipeline": "online",
        },
        "preprocess": {
            "cluster": {
                "enabled": True,
                "num_per_partition": 2,
                "max_nodes": 128,
                "max_edges": 256,
            },
            "chunk": {
                "window_multiple": 2,
                "max_events_per_chunk": 64,
                "max_time_span": 32,
            },
        },
        "sampling": {
            "neighbor_limit": [10],
            "strategy": "recent",
            "history": 1,
            "neg_sampling": "random",
        },
        "graph": {
            "storage": storage,
            "partition": "metis",
            "route": "all2all",
        },
        "dist": {
            "backend": "nccl",
            "world_size": 1,
            "master_addr": "127.0.0.1",
            "master_port": 29500,
        },
    }


def test_from_config_binds_ctdg_provider(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    assert session.provider.graph_mode == "ctdg"


def test_from_config_binds_dtdg_provider(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    assert session.provider.graph_mode == "dtdg"


def test_prepare_data_creates_ctdg_artifacts(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    prepared = session.prepare_data()
    assert prepared.meta_path.exists()
    payload = json.loads(prepared.meta_path.read_text(encoding="utf-8"))
    assert payload["graph_mode"] == "ctdg"
    assert "feature_route_plan" in payload


def test_prepare_data_creates_dtdg_artifacts(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    prepared = session.prepare_data()
    assert prepared.meta_path.exists()
    payload = json.loads(prepared.meta_path.read_text(encoding="utf-8"))
    assert payload["graph_mode"] == "dtdg"
    assert "snapshot_route_plan" in payload
    assert payload["pipeline"] == "flare_native"
    assert (prepared.directories["flare"] / "manifest.json").exists()


def test_prepare_data_creates_chunked_dtdg_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    prepared = session.prepare_data()
    payload = json.loads(prepared.meta_path.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "chunked"
    assert (prepared.directories["clusters"] / "part_000" / "cluster_manifest.json").exists()
    assert (prepared.directories["snapshots"] / "manifest.json").exists()


def test_build_runtime_ctdg_and_dtdg(tmp_path: Path) -> None:
    ctdg = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    ctdg.prepare_data()
    ctdg_runtime = ctdg.build_runtime()
    assert "memory_state" in ctdg_runtime.state

    dtdg = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    dtdg.prepare_data()
    dtdg_runtime = dtdg.build_runtime()
    assert "window_state" in dtdg_runtime.state
    assert dtdg_runtime.state["dtdg_pipeline"] == "flare_native"


def test_build_runtime_chunked_dtdg(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    session.prepare_data()
    runtime = session.build_runtime()
    assert runtime.state["dtdg_pipeline"] == "chunked"
    assert runtime.state["chunk_manifest"]


def test_chunked_node_task_writes_target_store(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    prepared = session.prepare_data()
    payload = json.loads(prepared.meta_path.read_text(encoding="utf-8"))
    target_path = session.ctx.artifact_root / "targets" / "node_targets.pth"
    assert target_path.exists()
    targets = torch.load(target_path, weights_only=False)
    assert targets["format"] == "chunk_node_targets_v1"
    num_slices = int(payload.get("num_slices", payload["num_snapshots"]))
    assert targets["target_ptr"].numel() == num_slices + 1
    assert targets["target_node"].numel() == targets["target_owner"].numel()
    assert payload["target_assignment_policy"] == "hot_target_master_else_node_owner"


def test_chunked_stgraph_supports_edge_predict_head(tmp_path: Path) -> None:
    config = make_config(tmp_path, "tgn", "temporal_link_prediction")
    config["data"]["graph_mode"] = "chunk"
    config["model"]["name"] = "mpnn_lstm"
    config["model"]["family"] = "mpnn_lstm"
    config["graph"]["partition"] = "balanced"
    config.setdefault("chunk", {})["partition_strategy"] = "balanced"
    config["chunk"]["runtime"] = "stgraph"

    session = SchedulerSession.from_config(config)
    session.prepare_data()
    session.build_runtime()
    summary = session.run_epoch(split="train")

    assert summary["steps"] > 0
    assert summary["outputs"][0]["meta"]["stgraph"] is True
    assert summary["outputs"][0]["meta"]["stage_payloads"]["edge_predict_head"] is True
    output = summary["outputs"][0]["output"]
    assert "pos_score" in output
    assert "neg_score" in output


def test_chunked_prepare_writes_canonical_artifact_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    session.prepare_data()
    root = session.ctx.artifact_root

    manifest_path = root / "artifact_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["placement_version"] == 0
    assert manifest["schema_version"] == 1
    required = set(manifest["required_files"])
    for rel_path in {
        "placement.pth",
        "events/canonical_events.pth",
        "targets/node_targets.pth",
        "indices/master_dist_index.pth",
        "indices/read_dist_index_000.pth",
        "chunks/chunk_assignment.pth",
        "routes/manifest.json",
    }:
        assert rel_path in required
        assert (root / rel_path).exists()

    events = torch.load(root / "events" / "canonical_events.pth", weights_only=False)
    chunks = torch.load(root / "chunks" / "chunk_assignment.pth", weights_only=False)
    assert events["format"] == "chunk_canonical_events_v1"
    assert events["schema_version"] == 1
    assert chunks["format"] == "chunk_assignment_v1"
    assert chunks["schema_version"] == 1


def test_missing_required_chunk_artifact_fails_early(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    session.prepare_data()
    (session.ctx.artifact_root / "events" / "canonical_events.pth").unlink()
    try:
        session.build_runtime()
    except FileNotFoundError as exc:
        assert "Missing required chunk artifact" in str(exc)
    else:
        raise AssertionError("Expected missing required chunk artifact failure")


def test_chunk_event_schema_mismatch_fails_early(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dtdg"]["pipeline"] = "chunked"
    session = SchedulerSession.from_config(config)
    session.prepare_data()
    event_path = session.ctx.artifact_root / "events" / "canonical_events.pth"
    payload = torch.load(event_path, weights_only=False)
    payload["format"] = "bad_schema"
    torch.save(payload, event_path)
    try:
        session.build_runtime()
    except RuntimeError as exc:
        assert "Canonical events schema mismatch" in str(exc)
    else:
        raise AssertionError("Expected canonical event schema mismatch failure")


def test_run_epoch_executes_both_chains(tmp_path: Path) -> None:
    ctdg = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    ctdg.prepare_data()
    ctdg.build_runtime()
    train_summary = ctdg.run_epoch(split="train")
    assert train_summary["steps"] > 0
    assert "sample->feature_fetch->state_fetch->memory_updater->neighbor_attention_aggregate->message_generate->state_writeback" in train_summary["outputs"][0]["meta"]["chain"]
    assert "state_transition" in train_summary["outputs"][0]["meta"]["stage_payloads"]
    assert "neighbor_attention_aggregate" in train_summary["outputs"][0]["meta"]["stage_payloads"]
    assert train_summary["outputs"][0]["meta"]["async_ops"]

    dtdg = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    dtdg.prepare_data()
    dtdg.build_runtime()
    train_summary = dtdg.run_epoch(split="train")
    assert train_summary["steps"] == 2
    assert "load_snapshot->route_apply->state_fetch->state_transition->state_writeback" in train_summary["outputs"][0]["meta"]["chain"]
    assert "state_transition" in train_summary["outputs"][0]["meta"]["stage_payloads"]


def test_checkpoint_roundtrip_restores_runtime_state(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    session.prepare_data()
    session.build_runtime()
    session.run_epoch(split="train")
    checkpoint_path = tmp_path / "scheduler.pkl"
    session.save_checkpoint(checkpoint_path)

    restored = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    restored.prepare_data()
    payload = restored.load_checkpoint(checkpoint_path)
    assert payload["global_step"] == session.global_step
    assert restored.provider.runtime.state["cursor"] == session.provider.runtime.state["cursor"]


def test_predict_unifies_output_shape(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    session.prepare_data()
    session.build_runtime()
    result = session.predict(split="test")
    assert len(result.predictions) > 0
    assert result.meta["graph_mode"] == "dtdg"


def test_inactive_fields_warn_but_do_not_fail(tmp_path: Path) -> None:
    config = make_config(tmp_path, "tgn", "temporal_link_prediction")
    config["graph"]["storage"] = "snapshots"
    session = SchedulerSession.from_config(config)
    assert session.ctx.warnings


def test_missing_required_path_uses_default_value(tmp_path: Path) -> None:
    config = make_config(tmp_path, "tgn", "temporal_link_prediction")
    del config["runtime"]["device"]
    session = SchedulerSession.from_config(config)
    assert session.ctx.config["runtime"]["device"] == "cpu"


def test_session_uses_single_process_dist_defaults(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    assert session.ctx.dist.world_size == 1
    assert session.ctx.dist.rank == 0
    assert session.ctx.dist.local_rank == 0
    assert session.ctx.dist.launcher == "single_process"


def test_torchrun_env_overrides_dist_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.2")
    monkeypatch.setenv("MASTER_PORT", "23456")

    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["runtime"]["device"] = "cuda"
    session = SchedulerSession.from_config(config)

    assert session.ctx.dist.world_size == 4
    assert session.ctx.dist.rank == 2
    assert session.ctx.dist.local_rank == 1
    assert session.ctx.dist.master_addr == "10.0.0.2"
    assert session.ctx.dist.master_port == 23456
    assert session.ctx.dist.launcher == "torchrun"
    assert session.ctx.config["runtime"]["device"] == "cuda:1"


def test_native_backend_uses_internal_cores(tmp_path: Path) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression", backend="native")
    session = SchedulerSession.from_config(config)
    prepared = session.prepare_data()
    assert prepared.provider_meta["graph_mode"] == "dtdg"
    runtime = session.build_runtime()
    assert "native_backend" not in runtime.state
    assert "snapshot_state" in runtime.state


def test_missing_route_manifest_fails_early(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "tgn", "temporal_link_prediction"))
    prepared = session.prepare_data()
    (prepared.directories["routes"] / "manifest.json").unlink()
    try:
        session.build_runtime()
    except FileNotFoundError as exc:
        assert "route manifest" in str(exc)
    else:
        raise AssertionError("Expected missing route manifest failure")


def test_artifact_num_parts_must_match_runtime_world_size(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path, "evolvegcn", "snapshot_node_regression")
    config["dist"]["world_size"] = 2

    prepare_session = SchedulerSession.from_config(config)
    prepare_session.prepare_data()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    train_session = SchedulerSession.from_config(config)
    train_session.provider.prepared = prepare_session.provider.prepared
    try:
        train_session.build_runtime()
    except RuntimeError as exc:
        assert "num_parts mismatch" in str(exc)
    else:
        raise AssertionError("Expected artifact world size mismatch")


def test_artifact_version_mismatch_fails_early(tmp_path: Path) -> None:
    session = SchedulerSession.from_config(make_config(tmp_path, "evolvegcn", "snapshot_node_regression"))
    prepared = session.prepare_data()
    prepared.meta_path.write_text(
        json.dumps({**prepared.provider_meta, "artifact_version": 999}),
        encoding="utf-8",
    )
    try:
        session.build_runtime()
    except RuntimeError as exc:
        assert "Artifact version mismatch" in str(exc)
    else:
        raise AssertionError("Expected artifact version mismatch")


def test_bts_native_sampler_module_is_merged() -> None:
    assert is_bts_sampler_available()
