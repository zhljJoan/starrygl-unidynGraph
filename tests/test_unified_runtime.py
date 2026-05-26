import torch

from atc_starrygl_lib.core.config import normalize_config
from atc_starrygl_lib.ctdg.runtime.backend import _native_sampler_policy
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
