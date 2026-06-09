from pathlib import Path

import torch

from atc_starrygl_lib.comm.dist_index import dist_index_is_shared, dist_index_loc, dist_index_part, encode_dist_index
from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.ctdg.runtime.backend import (
    MemShareTemporalSamplingBackend,
    _attach_precomputed_commit_roots,
    _remap_batch_root_indices,
    _build_sampler_temporal_graph,
    _populate_commit_rows,
    build_memory_replica_index,
)
from atc_starrygl_lib.memory import SharedHistoricalCache
from atc_starrygl_lib.sampling import MemShareNativeSamplerFactory, NativeSamplerConfig, RootSet, TemporalSamplingRequest


def test_new_pipeline_runtime_reader_iterates_rank_local_event_batches(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "dst": torch.tensor([5, 6, 7, 8, 9], dtype=torch.long),
        "ts": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        "edge_ids": torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 2], [2, 4]], dtype=torch.long),
            "val": torch.tensor([[4, 5]], dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
        "time_ptr_2": torch.tensor([[0, 2], [2, 4], [4, 5]], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0, 2], dtype=torch.long),
        "split_event_pos": {
            "train": {
                "data": torch.tensor([0, 2], dtype=torch.long),
                "ptr": torch.tensor([0, 1, 2], dtype=torch.long),
            },
            "val": {
                "data": torch.empty(0, dtype=torch.long),
                "ptr": torch.tensor([0, 0], dtype=torch.long),
            },
            "test": {
                "data": torch.empty(0, dtype=torch.long),
                "ptr": torch.tensor([0], dtype=torch.long),
            },
        },
        "split_time_ptr": {
            "train": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "val": torch.tensor([[0, 0]], dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(config={}, artifact_root=tmp_path, rank=0, world_size=1, device="cpu"),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    train = list(backend.iter_batches("train"))
    assert len(train) == 2
    assert train[0].eids.tolist() == [10]
    assert train[0].src.tolist() == [0]
    assert train[0].dst.tolist() == [5]
    assert train[0].ts.tolist() == [1.0]
    assert train[0].roots.tolist() == [0, 5]
    assert train[0].timestamps.tolist() == [1.0, 1.0]
    assert train[0].pos_src.tolist() == [0]
    assert train[0].pos_dst.tolist() == [1]
    assert train[0].node_ids is None
    assert train[1].eids.tolist() == [12]
    assert train[1].src.tolist() == [2]
    assert train[1].dst.tolist() == [7]
    assert train[1].timestamps.tolist() == [3.0, 3.0]

    val = list(backend.iter_batches("val"))
    assert len(val) == 1
    assert val[0].eids.numel() == 0
    assert val[0].roots.numel() == 0
    assert val[0].pos_src.numel() == 0
    assert val[0].pos_dst.numel() == 0


def test_new_pipeline_runtime_reader_can_drop_last_train_window(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "dst": torch.tensor([5, 6, 7, 8, 9], dtype=torch.long),
        "ts": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
        "edge_ids": torch.arange(5, dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 2], [2, 4], [4, 5]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
        "time_ptr_2": torch.tensor([[0, 2], [2, 4], [4, 5]], dtype=torch.long),
        "num_nodes": 10,
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.arange(5, dtype=torch.long),
        "split_event_pos": {
            "train": {
                "data": torch.arange(5, dtype=torch.long),
                "ptr": torch.tensor([0, 2, 4, 5], dtype=torch.long),
            },
            "val": {"data": torch.empty(0, dtype=torch.long), "ptr": torch.tensor([0], dtype=torch.long)},
            "test": {"data": torch.empty(0, dtype=torch.long), "ptr": torch.tensor([0], dtype=torch.long)},
        },
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {"train_drop_last": True, "train_drop_last_batch_size": 2},
                "preprocess": {"batch_size": 2},
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    train = list(backend.iter_batches("train"))
    assert len(train) == 2
    assert [batch.eids.tolist() for batch in train] == [[0, 1], [2, 3]]


def test_new_pipeline_runtime_reader_iterates_node_label_batches(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([1], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([0], dtype=torch.long),
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
        "node_label_nodes": torch.tensor([3, 4, 5, 6], dtype=torch.long),
        "node_label_ts": torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float32),
        "node_label": torch.tensor([1, 0, 2, 1], dtype=torch.long),
        "node_label_split": torch.tensor([0, 0, 1, 2], dtype=torch.uint8),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(config={"task": {"name": "node_prediction", "batch_size": 2}}, artifact_root=tmp_path, rank=0, world_size=1, device="cpu"),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    train = list(backend.iter_batches("train"))
    assert len(train) == 1
    assert train[0].roots.tolist() == [3, 4]
    assert train[0].timestamps.tolist() == [10.0, 11.0]
    assert train[0].node_ids.tolist() == [3, 4]
    assert train[0].labels.tolist() == [1, 0]
    assert train[0].pos_src is None
    assert train[0].pos_dst is None

    val = list(backend.iter_batches("val"))
    assert len(val) == 1
    assert val[0].roots.tolist() == [5]
    assert val[0].labels.tolist() == [2]


def test_new_pipeline_runtime_prefetches_sampling_and_patches_features(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0, 1], dtype=torch.long),
        "dst": torch.tensor([2, 3], dtype=torch.long),
        "ts": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "edge_ids": torch.tensor([10, 11], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
        "time_ptr_2": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0, 1], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    sampler = _FakeSampler()
    feature_runtime = _FakeFeatureRuntime()
    memory_runtime = _FakeMemoryRuntime()
    mailbox_runtime = _FakeMailboxRuntime()
    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "sampler": sampler,
                    "feature_runtime": feature_runtime,
                    "memory_runtime": memory_runtime,
                    "mailbox_runtime": mailbox_runtime,
                }
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    batches = list(backend.iter_batches("train"))
    assert [b.roots.tolist() for b in batches] == [[0, 2], [1, 3]]
    assert [b.graph[0].srcdata["h"].tolist() for b in batches] == [[[1.0]], [[2.0]]]
    assert [b.graph[0].edata["f"].tolist() for b in batches] == [[[30.0]], [[31.0]]]
    assert [b.graph[0].srcdata["mem"].tolist() for b in batches] == [[[10.0]], [[11.0]]]
    assert [b.graph[0].srcdata["mem_ts"].tolist() for b in batches] == [[100.0], [101.0]]
    assert [b.graph[0].srcdata["mem_input"].tolist() for b in batches] == [[[20.0, 21.0]], [[21.0, 22.0]]]
    assert [b.graph[0].srcdata["mail_ts"].tolist() for b in batches] == [[[200.0]], [[201.0]]]
    assert [b.pos_src.tolist() for b in batches] == [[0], [0]]
    assert [b.pos_dst.tolist() for b in batches] == [[1], [1]]
    assert [call["roots"] for call in sampler.calls] == [[0, 2], [1, 3]]
    assert [call["groups"] for call in sampler.calls] == [
        {"pos_src": (0, 1), "pos_dst": (1, 2)},
        {"pos_src": (0, 1), "pos_dst": (1, 2)},
    ]
    assert feature_runtime.submitted == [0, 1]
    assert memory_runtime.submitted == [0, 1]
    assert mailbox_runtime.submitted == [0, 1]


def test_new_pipeline_runtime_patches_historical_inputs_for_shared_nodes(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([2], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([10], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "split_time_ptr": {
            "train": torch.tensor([[0, 1]], dtype=torch.long),
            "val": torch.zeros((0, 2), dtype=torch.long),
            "test": torch.zeros((0, 2), dtype=torch.long),
        },
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    sampler = _FakeSampler()
    memory_runtime = _FakeMemoryRuntime(shared_nodes={0})
    mailbox_runtime = _FakeMailboxRuntime()
    historical_cache = SharedHistoricalCache(memory_dim=1, num_nodes=4, alpha=0.1, times_threshold=10)
    historical_cache.historical_memory[0] = torch.tensor([99.0], dtype=torch.float32)
    historical_cache.historical_ts[0] = torch.tensor(777.0, dtype=torch.float32)

    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "sampler": sampler,
                    "memory_runtime": memory_runtime,
                    "mailbox_runtime": mailbox_runtime,
                }
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )
    backend._runtime.historical_cache_provider = historical_cache

    batch = next(iter(backend.iter_batches("train")))
    srcdata = batch.graph[0].srcdata
    assert srcdata["shared_mask"].tolist() == [True]
    assert srcdata["mem"].tolist() == [[10.0]]
    assert srcdata["mem_ts"].tolist() == [100.0]
    assert srcdata["mail_ts"].tolist() == [[200.0]]
    assert srcdata["his_mem"].tolist() == [[99.0]]
    assert srcdata["his_ts"].tolist() == [[777.0]]


def test_new_pipeline_runtime_builds_default_sampler_feature_memory_and_mailbox_runtime(monkeypatch, tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([1], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([0], dtype=torch.long),
        "num_nodes": 2,
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
    }
    dist = {
        "world_size": 1,
        "edge_owner": torch.tensor([0], dtype=torch.long),
        "node_to_chunk": torch.tensor([0, 0], dtype=torch.long),
        "chunk_owner": torch.tensor([0], dtype=torch.long),
        "master_dist_index": torch.tensor([10, 11], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_node_ids": torch.tensor([0, 1], dtype=torch.long),
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "read_dist_index": torch.tensor([0, 1], dtype=torch.long),
    }
    feature = {
        "node_feat": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "edge_feat": None,
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save(dist, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")
    torch.save(feature, tmp_path / "feature_000.pt")
    built = {}

    class FakeFactory:
        def __init__(self, graph_name):
            built["graph_name"] = graph_name

        def build(self, graph_data, config):
            built["graph_data"] = graph_data
            built["config"] = config
            return _FakeSampler()

    monkeypatch.setattr("atc_starrygl_lib.ctdg.runtime.backend.MemShareNativeSamplerFactory", FakeFactory)
    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "build_sampler": True,
                    "build_feature_runtime": True,
                    "build_memory_runtime": True,
                    "build_mailbox_runtime": True,
                    "memory_use_shared_reads": True,
                    "mailbox_use_shared_reads": True,
                    "memory_dim": 4,
                    "mailbox_size": 2,
                    "mailbox_msg_dim": 8,
                    "fanouts": [2, 3],
                    "num_layers": 2,
                    "policy": "recent",
                    "sampler_workers": 4,
                    "graph_name": "wiki",
                }
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
                "feature_000": tmp_path / "feature_000.pt",
            },
        ),
    )

    assert built["graph_name"] == "wiki"
    assert built["graph_data"].row.tolist() == [0, 1]
    assert built["graph_data"].col.tolist() == [1, 0]
    assert built["graph_data"].edge_ids.tolist() == [0, 0]
    assert built["graph_data"].timestamps.tolist() == [1.0, 1.0]
    assert built["graph_data"].edge_part.tolist() == [0, 0]
    assert built["graph_data"].node_part.tolist() == [0, 0]
    assert built["config"].fanouts == (2, 3)
    assert built["config"].num_layers == 2
    assert backend._runtime.feature_runtime is not None
    assert backend._runtime.memory_runtime is not None
    assert backend._runtime.mailbox_runtime is not None
    assert torch.equal(backend._runtime.memory_runtime.index.read_dist_index, rank["read_dist_index"])
    assert torch.equal(backend._runtime.mailbox_runtime.index.read_dist_index, rank["read_dist_index"])
    assert backend._runtime.memory_runtime.store.memory.shape == (2, 4)
    assert backend._runtime.mailbox_runtime.store.mailbox.shape == (2, 2, 8)


def test_build_sampler_temporal_graph_can_disable_reverse_edges() -> None:
    graph = {
        "src": torch.tensor([0, 2], dtype=torch.long),
        "dst": torch.tensor([1, 3], dtype=torch.long),
        "ts": torch.tensor([10, 20], dtype=torch.long),
        "edge_ids": torch.tensor([100, 101], dtype=torch.long),
    }

    out = _build_sampler_temporal_graph(
        graph=graph,
        num_nodes=4,
        node_part=None,
        edge_owner=torch.tensor([0, 1], dtype=torch.long),
        add_reverse_edges=False,
    )

    assert out.row.tolist() == [0, 2]
    assert out.col.tolist() == [1, 3]
    assert out.edge_ids.tolist() == [100, 101]
    assert out.timestamps.tolist() == [10, 20]
    assert out.edge_part.tolist() == [0, 1]


def test_native_compact_sampling_uses_unique_head_rows_with_root_inverse() -> None:
    graph_data = _build_sampler_temporal_graph(
        graph={
            "src": torch.tensor([0, 0], dtype=torch.long),
            "dst": torch.tensor([1, 2], dtype=torch.long),
            "ts": torch.tensor([1, 1], dtype=torch.long),
            "edge_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        num_nodes=3,
        node_part=None,
        edge_owner=torch.tensor([0, 0], dtype=torch.long),
        add_reverse_edges=False,
    )
    sampler = MemShareNativeSamplerFactory(graph_name="unit").build(
        graph_data,
        NativeSamplerConfig(
            fanouts=(2,),
            num_layers=1,
            policy="recent",
            workers=1,
        ),
    )

    out = sampler.sample(
        TemporalSamplingRequest(
            roots=RootSet(
                nodes=torch.tensor([0, 0, 1], dtype=torch.long),
                ts=torch.tensor([2, 2, 2], dtype=torch.long),
                groups={"pos_src": (0, 1), "pos_dst": (1, 2), "neg_dst": (2, 3)},
            ),
            fanouts=(),
            num_layers=0,
            policy="runtime",
        )
    )

    assert out.node_compute.root_lids.tolist() == [0, 0, 1]
    assert out.mfgs[0].dst_lids.tolist() == [0, 1]


def test_native_compact_sampling_exports_edge_feature_read_layout() -> None:
    graph_data = _build_sampler_temporal_graph(
        graph={
            "src": torch.tensor([0, 0], dtype=torch.long),
            "dst": torch.tensor([1, 2], dtype=torch.long),
            "ts": torch.tensor([1, 1], dtype=torch.long),
            "edge_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        num_nodes=3,
        node_part=None,
        edge_owner=torch.tensor([1, 0], dtype=torch.long),
        add_reverse_edges=False,
        edge_read_dist_index=encode_dist_index(
            torch.tensor([3, 1], dtype=torch.long),
            torch.tensor([1, 0], dtype=torch.long),
        ),
    )
    sampler = MemShareNativeSamplerFactory(graph_name="unit_edge_read_layout").build(
        graph_data,
        NativeSamplerConfig(
            fanouts=(2,),
            num_layers=1,
            policy="recent",
            workers=1,
            world_size=2,
        ),
    )

    out = sampler.sample(
        TemporalSamplingRequest(
            roots=RootSet(
                nodes=torch.tensor([1, 2], dtype=torch.long),
                ts=torch.tensor([2, 2], dtype=torch.long),
                groups={},
            ),
            fanouts=(),
            num_layers=0,
            policy="runtime",
        )
    )

    assert out.edge_compute.edge_gids.tolist() == [0, 1]
    assert out.edge_comm.read_ptr.tolist() == [0, 1, 2]
    assert dist_index_part(out.edge_comm.read_index).tolist() == [0, 1]
    assert dist_index_loc(out.edge_comm.read_index).tolist() == [1, 3]
    assert out.edge_comm.compute_to_feature.tolist() == [1, 0]


def test_build_memory_replica_index_targets_only_remote_shared_rows() -> None:
    dist = {
        "master_dist_index": torch.cat(
            [
                encode_dist_index(torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long), shared=True),
                encode_dist_index(torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long), shared=True),
            ],
            dim=0,
        ),
        "replica_node_ids_by_part": [
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ],
    }

    replica_index = build_memory_replica_index(dist=dist)

    assert replica_index is not None
    assert replica_index.replica_ptr.tolist() == [0, 1, 1]
    assert dist_index_part(replica_index.replica_target_index).tolist() == [1]
    assert dist_index_loc(replica_index.replica_target_index).tolist() == [0]
    assert dist_index_is_shared(replica_index.replica_target_index).tolist() == [True]


def test_new_pipeline_runtime_attaches_negative_roots_before_sampling(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([2], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([10], dtype=torch.long),
        "num_nodes": 4,
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    sampler = _FakeSampler()
    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "sampler": sampler,
                    "negative_sampler": _FixedNegativeSampler(torch.tensor([3], dtype=torch.long)),
                    "negative_ratio": 1,
                }
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    batch = next(backend.iter_batches("train"))
    assert sampler.calls[0]["roots"] == [0, 2, 3]
    assert sampler.calls[0]["groups"] == {"pos_src": (0, 1), "pos_dst": (1, 2), "neg_dst": (2, 3)}
    assert batch.neg_dst.tolist() == [2]
    assert batch.neg_weight is None


def test_new_pipeline_runtime_attaches_negative_weights(tmp_path: Path) -> None:
    graph = {
        "src": torch.tensor([0], dtype=torch.long),
        "dst": torch.tensor([2], dtype=torch.long),
        "ts": torch.tensor([1.0], dtype=torch.float32),
        "edge_ids": torch.tensor([10], dtype=torch.long),
        "num_nodes": 4,
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
        "time_ptr_2": torch.tensor([[0, 1]], dtype=torch.long),
    }
    rank = {
        "rank": 0,
        "local_edge_ids": torch.tensor([0], dtype=torch.long),
        "split_time_ptr": {"train": torch.tensor([[0, 1]], dtype=torch.long)},
    }
    torch.save(graph, tmp_path / "graph.pt")
    torch.save({"world_size": 1}, tmp_path / "dist.pt")
    torch.save(rank, tmp_path / "rank_000.pt")

    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    backend.build_runtime(
        RuntimeContext(
            config={
                "runtime": {
                    "sampler": _FakeSampler(),
                    "negative_sampler": _FixedNegativeSampler(
                        torch.tensor([3], dtype=torch.long),
                        weight=torch.tensor([4.0], dtype=torch.float32),
                    ),
                    "negative_ratio": 1,
                }
            },
            artifact_root=tmp_path,
            rank=0,
            world_size=1,
            device="cpu",
        ),
        ArtifactBundle(
            root=tmp_path,
            graph_mode="ctdg",
            files={
                "graph": tmp_path / "graph.pt",
                "dist": tmp_path / "dist.pt",
                "rank_000": tmp_path / "rank_000.pt",
            },
        ),
    )

    batch = next(backend.iter_batches("train"))
    assert batch.neg_weight.tolist() == [4.0]


def test_populate_commit_rows_uses_first_block_node_time_mapping() -> None:
    block = _FakeBlock()
    block.srcdata["ID"] = torch.tensor([10, 11, 10, 12], dtype=torch.long)
    block.srcdata["ts"] = torch.tensor([1.0, 2.0, 2.0, 2.0], dtype=torch.float32)
    from atc_starrygl_lib.core.types import Batch
    batch = Batch(
        split="train",
        roots=torch.tensor([10, 11, 10, 12], dtype=torch.long),
        graph=[[block]],
        src=torch.tensor([10, 11], dtype=torch.long),
        dst=torch.tensor([10, 12], dtype=torch.long),
        ts=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )

    _populate_commit_rows(batch)

    assert batch.commit_src_rows is not None and batch.commit_src_rows.tolist() == [0, 1]
    assert batch.commit_dst_rows is not None and batch.commit_dst_rows.tolist() == [0, 3]


def test_populate_commit_rows_prefers_batch_root_row_space() -> None:
    from atc_starrygl_lib.core.types import Batch

    batch = Batch(
        split="train",
        roots=torch.tensor([10, 10, 12], dtype=torch.long),
        graph=[[_FakeBlock()]],
        src=torch.tensor([10], dtype=torch.long),
        dst=torch.tensor([12], dtype=torch.long),
        ts=torch.tensor([1.0], dtype=torch.float32),
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
    )

    _populate_commit_rows(batch)

    assert batch.commit_src_rows is not None and batch.commit_src_rows.tolist() == [0]
    assert batch.commit_dst_rows is not None and batch.commit_dst_rows.tolist() == [1]


def test_remap_batch_root_indices_uses_sampling_inverse_mapping() -> None:
    from atc_starrygl_lib.core.types import Batch

    batch = Batch(
        split="train",
        roots=torch.tensor([10, 10, 12], dtype=torch.long),
        timestamps=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
        graph=[[_FakeBlock()]],
        pos_src=torch.tensor([0], dtype=torch.long),
        pos_dst=torch.tensor([1], dtype=torch.long),
        neg_dst=torch.tensor([2], dtype=torch.long),
    )
    output = _FakeSamplingOutput(
        idx=0,
        groups={"pos_src": (0, 1), "pos_dst": (1, 2), "neg_dst": (2, 3)},
        root_lids=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    _remap_batch_root_indices(batch, output)

    assert batch.pos_src is not None and batch.pos_src.tolist() == [0]
    assert batch.pos_dst is not None and batch.pos_dst.tolist() == [0]
    assert batch.neg_dst is not None and batch.neg_dst.tolist() == [1]


def test_remap_batch_root_indices_maps_precomputed_commit_roots() -> None:
    from atc_starrygl_lib.core.types import Batch

    batch = Batch(
        split="train",
        roots=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        timestamps=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32),
        graph=[[_FakeBlock()]],
        commit_memory_nodes=torch.tensor([11, 13], dtype=torch.long),
        commit_memory_root_pos=torch.tensor([1, 3], dtype=torch.long),
        commit_mailbox_nodes=torch.tensor([11, 13], dtype=torch.long),
        commit_mailbox_self_root_pos=torch.tensor([1, 3], dtype=torch.long),
        commit_mailbox_peer_root_pos=torch.tensor([3, 1], dtype=torch.long),
        commit_mailbox_edge_pos=torch.tensor([1, 1], dtype=torch.long),
        commit_mailbox_ts=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    output = _FakeSamplingOutput(
        idx=0,
        groups={},
        root_lids=torch.tensor([7, 8, 9, 10], dtype=torch.long),
    )

    _remap_batch_root_indices(batch, output)
    _populate_commit_rows(batch)

    assert batch.commit_memory_rows is not None and batch.commit_memory_rows.tolist() == [8, 10]
    assert batch.commit_mailbox_self_rows is not None and batch.commit_mailbox_self_rows.tolist() == [8, 10]
    assert batch.commit_mailbox_peer_rows is not None and batch.commit_mailbox_peer_rows.tolist() == [10, 8]
    assert batch.commit_src_rows is None
    assert batch.commit_dst_rows is None


def test_attach_precomputed_commit_roots_adds_memory_write_layout() -> None:
    from atc_starrygl_lib.core.types import Batch

    batch = Batch(
        split="train",
        roots=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        timestamps=torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32),
    )
    rank = {
        "update_node_ptr": torch.tensor([0, 2], dtype=torch.long),
        "update_node_ids": torch.tensor([10, 13], dtype=torch.long),
        "update_node_ts": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "update_event_pos": torch.tensor([0, 1], dtype=torch.long),
        "update_endpoint": torch.tensor([0, 1], dtype=torch.long),
        "memory_write_route": {
            "ptr": torch.tensor([0, 2], dtype=torch.long),
            "target_ptr": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "target_index": encode_dist_index(
                torch.tensor([5, 7], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            ),
            "source_pos": torch.tensor([0, 1], dtype=torch.long),
        },
    }

    _attach_precomputed_commit_roots(
        batch,
        rank_artifact=rank,
        window_index=0,
        event_pos=torch.tensor([0, 1], dtype=torch.long),
        num_edges=2,
        device=torch.device("cpu"),
    )

    assert batch.commit_memory_nodes is not None and batch.commit_memory_nodes.tolist() == [10, 13]
    assert batch.commit_memory_source_pos is not None and batch.commit_memory_source_pos.tolist() == [0, 1]
    assert batch.commit_memory_target_ptr is not None and batch.commit_memory_target_ptr.tolist() == [0, 1, 2]
    assert batch.commit_memory_target_index is not None
    assert dist_index_part(batch.commit_memory_target_index).tolist() == [0, 1]
    assert dist_index_loc(batch.commit_memory_target_index).tolist() == [5, 7]


class _FakeBlock:
    def __init__(self) -> None:
        self.srcdata = {"__ID": torch.tensor([0], dtype=torch.long), "ID": torch.tensor([0], dtype=torch.long)}
        self.edata = {"__ID": torch.tensor([0], dtype=torch.long), "ID": torch.tensor([0], dtype=torch.long)}


class _FakeSamplingOutput:
    def __init__(self, idx: int, groups: dict[str, tuple[int, int]], root_lids: torch.Tensor | None = None) -> None:
        self.idx = idx
        self.mfgs = [_FakeBlock()]
        self.node_compute = _FakeNodeCompute(groups, root_lids=root_lids)
        self.edge_compute = _FakeEdgeCompute()
        self.edge_comm = _FakeEdgeComm()


class _FakeNodeCompute:
    def __init__(self, groups: dict[str, tuple[int, int]], root_lids: torch.Tensor | None = None) -> None:
        self.root_lids = torch.arange(0, max((end for _, end in groups.values()), default=0), dtype=torch.long) if root_lids is None else root_lids
        self.groups = groups


class _FakeEdgeCompute:
    def __init__(self) -> None:
        self.edge_gids = torch.tensor([0], dtype=torch.long)
        self.edge_ts = None


class _FakeEdgeComm:
    def __init__(self) -> None:
        self.edge_gids = torch.tensor([0], dtype=torch.long)
        self.compute_to_comm = torch.tensor([0], dtype=torch.long)
        self.time_slices = None


class _FakeSampler:
    def __init__(self) -> None:
        self.calls = []

    def sample(self, request):
        idx = len(self.calls)
        groups = dict(request.roots.groups)
        self.calls.append({"roots": request.roots.nodes.tolist(), "groups": groups})
        return _FakeSamplingOutput(idx, groups)


class _FixedNegativeSampler:
    def __init__(self, neg_dst: torch.Tensor, weight: torch.Tensor | None = None) -> None:
        self.neg_dst = neg_dst
        self.weight = weight

    def sample(self, request):
        from atc_starrygl_lib.sampling.negative import NegativeSamplingResult

        return NegativeSamplingResult(
            neg_src=request.pos_src.repeat_interleave(int(request.ratio)),
            neg_dst=self.neg_dst.to(request.pos_src.device),
            ratio=int(request.ratio),
            weight=None if self.weight is None else self.weight.to(request.pos_src.device),
        )


class _FakeHandle:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class _FakeFeatureRuntime:
    def __init__(self) -> None:
        self.submitted = []
        self.patched = []

    def build_layout_from_sampling(self, output):
        return output.idx

    def submit_fetch(self, layout):
        self.submitted.append(layout)
        return _FakeHandle(layout)

    def wait_fetch(self, handle, layout):
        return torch.tensor([[float(handle.idx + 1)]])

    def build_edge_layout_from_sampling(self, output):
        return output.idx

    def submit_edge_fetch(self, layout):
        return _FakeHandle(layout)

    def wait_edge_fetch(self, handle, layout):
        return torch.tensor([[float(handle.idx + 30)]])


class _FakeMemoryRuntime:
    def __init__(self, *, shared_nodes: set[int] | None = None) -> None:
        self.submitted = []
        self.index = _FakeMemoryIndex(shared_nodes=shared_nodes or set())

    def build_read_layout_from_sampling(self, output):
        return output.idx

    def submit_read(self, layout):
        self.submitted.append(layout)
        return _FakeHandle(layout)

    def wait_read(self, handle, layout):
        idx = float(handle.idx)
        return torch.tensor([[10.0 + idx]]), torch.tensor([100.0 + idx])


class _FakeMailboxRuntime:
    def __init__(self) -> None:
        self.submitted = []

    def build_read_layout_from_sampling(self, output):
        return output.idx

    def submit_read(self, layout):
        self.submitted.append(layout)
        return _FakeHandle(layout)

    def wait_read(self, handle, layout):
        idx = float(handle.idx)
        return torch.tensor([[[20.0 + idx, 21.0 + idx]]]), torch.tensor([[200.0 + idx]])


class _FakeMemoryIndex:
    def __init__(self, *, shared_nodes: set[int]) -> None:
        self.master_dist_index = encode_dist_index(
            torch.arange(16, dtype=torch.long),
            torch.zeros(16, dtype=torch.long),
            shared=torch.tensor([node in shared_nodes for node in range(16)], dtype=torch.bool),
        )

    def master_for(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.master_dist_index.index_select(0, node_ids.long())
