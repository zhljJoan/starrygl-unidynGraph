import torch

from atc_starrygl_lib.comm.dist_index import encode_dist_index
from atc_starrygl_lib.core.config import normalize_config
from atc_starrygl_lib.ctdg.runtime.backend import _native_sampler_policy
from atc_starrygl_lib.memory.sync_mode import normalize_memory_sync_config, resolve_memory_sync_mode
from atc_starrygl_lib.runtime.unified import SampledBlockGCNEncoder
from atc_starrygl_lib.runtime.unified import build_model_and_head
from atc_starrygl_lib.core.types import RuntimeContext
from pathlib import Path


def test_config_infers_snapshot_full_graph_for_dtdg_model_without_sampling() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "tgcn"},
        "task": {"name": "node_regression"},
    })

    assert cfg["graph"]["mode"] == "dtdg"
    assert cfg["runtime"]["execution_plan"] == "snapshot_full_graph"


def test_config_routes_dtdg_model_with_sampling_to_temporal_sampling() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "gcn"},
        "gnn": {"sampling": {"fanouts": [5, 5], "policy": "uniform"}},
        "task": {"name": "edge_prediction"},
    })

    assert cfg["graph"]["mode"] == "ctdg"
    assert cfg["runtime"]["execution_plan"] == "temporal_sampling"
    assert cfg["runtime"]["fanouts"] == [5, 5]
    assert cfg["runtime"]["num_layers"] == 2
    assert cfg["runtime"]["policy"] == "uniform"


def test_config_maps_full_graph_slice_config_and_history() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "tgcn"},
        "gnn": {
            "history": 2,
            "slice_config": {"chunk_decay": [1, 2]},
            "full_graph": {"chunk_order": "identity"},
        },
        "task": {"name": "node_regression"},
    })

    assert cfg["graph"]["mode"] == "dtdg"
    assert cfg["model"]["history"] == 2
    assert cfg["runtime"]["num_full_snapshots"] == 2
    assert cfg["runtime"]["chunk_decay"] == [1, 2]
    assert cfg["runtime"]["chunk_order"] == "identity"


def test_boundary_sampling_config_maps_probability_and_memory_history() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "general"},
        "gnn": {
            "history": 3,
            "memory_update": "rnn",
            "memory_history": 4,
            "sampling": {
                "fanouts": [10],
                "policy": "boundary_recent_sample",
                "probability": 0.1,
            },
        },
        "task": {"name": "edge_prediction"},
    })

    assert cfg["graph"]["mode"] == "ctdg"
    assert cfg["model"]["memory_update"] == "rnn"
    assert cfg["model"]["memory_history"] == 4
    assert cfg["runtime"]["policy"] == "boundary_recent_uniform"
    assert cfg["runtime"]["sample_probability"] == 0.1
    assert cfg["runtime"]["mailbox_size"] == 4
    assert _native_sampler_policy(cfg["runtime"]["policy"]) == "boundery_recent_uniform"


def test_runtime_component_sections_map_to_flat_runtime_controls() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "general"},
        "gnn": {"sampling": {"fanouts": [5], "policy": "uniform"}},
        "task": {"name": "edge_prediction"},
        "runtime": {
            "sampling": {"workers": 7, "prefetch_lookahead": 4},
            "communication": {
                "gradient_sync": "ddp",
                "schedule_async_commit": True,
                "profile_sync_timing": True,
            },
            "training": {
                "feature_device": "cpu",
                "train_compute_metrics": False,
                "commit_memory": False,
            },
        },
    })

    runtime = cfg["runtime"]
    assert runtime["sampler_workers"] == 7
    assert runtime["prefetch_sample_lookahead"] == 4
    assert runtime["gradient_sync"] == "ddp"
    assert runtime["schedule_async_commit"] is True
    assert runtime["profile_sync_timing"] is True
    assert runtime["feature_device"] == "cpu"
    assert runtime["train_compute_metrics"] is False
    assert runtime["commit_memory"] is False


def test_historical_runtime_builds_blend_enabled_updater() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "memory_sync_mode": "memshare_historical",
            "historical": {"enabled": True, "alpha": 0.1, "times_threshold": 10},
            "async_memory": {"shared_filter": True, "staged_commit": True, "delta_compensation": True},
        },
    })

    class DummyBackend:
        pass

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = object()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_cache is not None
    assert updater.historical_blend is not None


def test_memshare_historical_sync_mode_builds_memshare_style_updater() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_historical",
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    class DummyBackend:
        pass

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = object()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_cache is not None
    assert updater.historical_filter is not None
    assert float(updater.historical_cache.alpha) == 0.1
    assert updater.historical_blend is not None
    assert updater.use_staged_commit is True
    assert updater.use_shared_filter is True
    assert updater.enable_delta_compensation is True


def test_shared_memory_ssim_overrides_historical_alpha() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_historical",
            "shared_memory_ssim": 2.0,
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    class DummyBackend:
        pass

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = object()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_cache is not None
    assert updater.historical_filter is not None
    assert float(updater.historical_cache.alpha) == 2.0


def test_memshare_historical_can_use_shared_local_filter_index_scope() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_historical",
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {
                "alpha": 0.1,
                "times_threshold": 10,
                "filter_index_scope": "shared_local",
            },
        },
    })

    class DummyBackend:
        pass

    class DummyIndex:
        def __init__(self) -> None:
            self.read_dist_index = encode_dist_index(
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.tensor([0, 0, 0], dtype=torch.long),
                shared=torch.tensor([True, False, True], dtype=torch.bool),
            )

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = type("MemRt", (), {"index": DummyIndex()})()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_filter is not None
    assert int(updater.historical_filter.historical_memory.size(0)) == 3
    assert updater.historical_filter.node_id_to_index.tolist() == [0, -1, 2]


def test_memshare_historical_defaults_to_shared_local_filter_index_scope() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_historical",
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {
                "alpha": 0.1,
                "times_threshold": 10,
            },
        },
    })

    class DummyBackend:
        pass

    class DummyIndex:
        def __init__(self) -> None:
            self.read_dist_index = encode_dist_index(
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.tensor([0, 0, 0], dtype=torch.long),
                shared=torch.tensor([True, False, True], dtype=torch.bool),
            )

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = type("MemRt", (), {"index": DummyIndex()})()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_filter is not None
    assert int(updater.historical_filter.historical_memory.size(0)) == 3
    assert updater.historical_filter.node_id_to_index.tolist() == [0, -1, 2]


def test_memshare_public_exact_sync_mode_builds_filter_only_updater() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_public_exact",
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    class DummyBackend:
        pass

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = object()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_cache is not None
    assert updater.historical_blend is None
    assert updater.use_staged_commit is False
    assert updater.use_shared_filter is True
    assert updater.enable_delta_compensation is False


def test_memshare_public_historical_sync_mode_builds_memshare_closest_updater() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {
            "name": "general",
            "hidden_dim": 8,
            "dim_time": 8,
            "gnn_arch": "identity",
            "layers": 1,
            "memory_update": "gru",
            "memory_history": 1,
        },
        "gnn": {
            "history": 1,
            "sampling": {"fanouts": [5], "policy": "boundary_recent_decay", "probability": 0.1},
        },
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_public_historical",
            "build_memory_runtime": True,
            "build_mailbox_runtime": True,
            "memory_dim": 8,
            "mailbox_size": 1,
            "mailbox_msg_dim": 16,
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    class DummyBackend:
        pass

    class DummyRuntime:
        pass

    backend = DummyBackend()
    runtime = DummyRuntime()
    runtime.memory_runtime = object()
    runtime.mailbox_runtime = object()
    runtime.edge_feat_dim = 0
    backend._runtime = runtime

    ctx = RuntimeContext(config=cfg, artifact_root=Path("/tmp"), rank=0, world_size=1, device="cpu")
    model, _ = build_model_and_head(ctx, backend)
    updater = model.memory_updater

    assert updater.historical_cache is not None
    assert updater.historical_filter is not None
    assert updater.historical_blend is not None
    assert updater.use_staged_commit is True
    assert updater.use_shared_filter is True
    assert updater.enable_delta_compensation is True


def test_normalize_config_materializes_memshare_historical_runtime_contract() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "general"},
        "gnn": {"sampling": {"fanouts": [5], "policy": "uniform"}},
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_historical",
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    runtime = cfg["runtime"]
    assert runtime["memory_sync_mode"] == "memshare_historical"
    assert runtime["wait_mode"] == "memshare"
    assert runtime["preserve_replica_history"] is True
    assert runtime["schedule_async_commit"] is True
    assert runtime["memory_replica_push"] is True
    assert runtime["mailbox_replica_push"] is True
    assert runtime["historical"]["enabled"] is True
    assert runtime["async_memory"]["shared_filter"] is True
    assert runtime["async_memory"]["staged_commit"] is True
    assert runtime["async_memory"]["delta_compensation"] is True
    assert runtime["async_memory"]["preload_candidate_delta"] is True
    assert runtime["async_memory"]["historical_blend"] is True
    assert runtime["async_memory"]["commit_order"] == "memshare"


def test_normalize_config_materializes_memshare_public_exact_runtime_contract() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "general"},
        "gnn": {"sampling": {"fanouts": [5], "policy": "uniform"}},
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_public_exact",
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    runtime = cfg["runtime"]
    assert runtime["memory_sync_mode"] == "memshare_public_exact"
    assert runtime["wait_mode"] == "legacy"
    assert runtime["preserve_replica_history"] is True
    assert runtime["schedule_async_commit"] is True
    assert runtime["memory_replica_push"] is True
    assert runtime["mailbox_replica_push"] is True
    assert runtime["historical"]["enabled"] is True
    assert runtime["async_memory"]["shared_filter"] is True
    assert runtime["async_memory"]["staged_commit"] is False
    assert runtime["async_memory"]["delta_compensation"] is False
    assert runtime["async_memory"]["preload_candidate_delta"] is False
    assert runtime["async_memory"]["historical_blend"] is False


def test_normalize_config_materializes_memshare_public_historical_runtime_contract() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "general"},
        "gnn": {"sampling": {"fanouts": [5], "policy": "uniform"}},
        "task": {"name": "edge_prediction"},
        "runtime": {
            "memory_sync_mode": "memshare_public_historical",
            "historical": {"alpha": 0.1, "times_threshold": 10},
        },
    })

    runtime = cfg["runtime"]
    assert runtime["memory_sync_mode"] == "memshare_public_historical"
    assert runtime["wait_mode"] == "memshare"
    assert runtime["preserve_replica_history"] is True
    assert runtime["schedule_async_commit"] is True
    assert runtime["memory_replica_push"] is True
    assert runtime["mailbox_replica_push"] is True
    assert runtime["historical"]["enabled"] is True
    assert runtime["async_memory"]["shared_filter"] is True
    assert runtime["async_memory"]["staged_commit"] is True
    assert runtime["async_memory"]["delta_compensation"] is True
    assert runtime["async_memory"]["preload_candidate_delta"] is True
    assert runtime["async_memory"]["historical_blend"] is True
    assert runtime["async_memory"]["commit_order"] == "memshare"


def test_resolve_memory_sync_mode_prefers_explicit_wait_mode() -> None:
    sync_cfg = resolve_memory_sync_mode({
        "wait_mode": "memshare",
        "async_memory": {"commit_order": "legacy"},
    })

    assert sync_cfg["wait_mode"] == "memshare"


def test_resolve_memory_sync_mode_defaults_to_memshare_public_exact() -> None:
    sync_cfg = resolve_memory_sync_mode({})

    assert sync_cfg["mode"] == "memshare_public_exact"
    assert sync_cfg["schedule_async_commit"] is True
    assert sync_cfg["memory_replica_push"] is True
    assert sync_cfg["mailbox_replica_push"] is True


def test_memshare_historical_defaults_to_async_overlap() -> None:
    sync_cfg = resolve_memory_sync_mode({
        "memory_sync_mode": "memshare_historical",
        "historical": {"enabled": True},
    })

    assert sync_cfg["schedule_async_commit"] is True


def test_normalize_memory_sync_config_preserves_user_replica_push_override() -> None:
    runtime = normalize_memory_sync_config({
        "memory_sync_mode": "memshare_historical",
        "memory_replica_push": False,
        "mailbox_replica_push": False,
    })

    assert runtime["memory_replica_push"] is False
    assert runtime["mailbox_replica_push"] is False


def test_sampled_block_gcn_encoder_uses_sampled_block_features() -> None:
    class Block:
        is_block = True

        def __init__(self) -> None:
            self.srcdata = {"h": torch.ones(3, 2)}
            self.dstdata = {}
            self.edata = {"gcn_norm": torch.ones(2)}

        def num_src_nodes(self):
            return 3

        def num_dst_nodes(self):
            return 2

        def local_scope(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update_all(self, _msg, _reduce):
            self.dstdata["x"] = self.srcdata["x"][:2]

    encoder = SampledBlockGCNEncoder(2, 4, num_layers=1)
    out = encoder.encode([[Block()]])

    assert out.shape == (2, 4)
