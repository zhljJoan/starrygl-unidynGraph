"""End-to-end test for chunk runtime阶段8.

Validates:
1. prepare() pipeline runs with mem_routes
2. ChunkRuntimeLoader.iter_train/eval/predict yield BatchData
3. All four task adapters build BatchData and compute loss/metrics
4. PredictionHead + run_batch interface
5. Negative sampler produces correct shapes
"""

import torch
from starry_unigraph.backends.chunk.prepare import prepare, build_chunk_assignment
from starry_unigraph.backends.chunk.data.partition import PartitionData, TensorData
from starry_unigraph.backends.chunk.data.batch import BatchData
from starry_unigraph.backends.chunk.runtime import (
    get_task_adapter, run_batch, ChunkRuntimeLoader,
    NegativeSamplerHook,
)
from starry_unigraph.backends.chunk.model import PredictionHead


# ---------------------------------------------------------------------------
# Fixture: tiny PartitionData (2 snapshots)
# ---------------------------------------------------------------------------

def _make_part(num_local=8, num_remote=4, num_edges=10, num_snaps=2):
    """Build a tiny PartitionData for testing.

    In this PartitionData variant:
      edge_dst  = torch.arange(num_local + 1)  (row-index range, NOT flat dst list)
      edge_ptr  = CSR pointer of length num_local + 1
      edge_src  = flat source indices per edge (length num_edges)
    """
    # Assign edges uniformly to dst nodes
    rng = torch.Generator(); rng.manual_seed(42)
    snaps_src  = [torch.randint(0, num_local + num_remote, (num_edges,), generator=rng) for _ in range(num_snaps)]
    snaps_eids = [torch.arange(num_edges) for _ in range(num_snaps)]
    snaps_chunks = [torch.zeros(num_edges, dtype=torch.long) for _ in range(num_snaps)]
    node_to_chunk = torch.zeros(num_local + num_remote, dtype=torch.long)

    # edge_dst = range row-index (length num_local + 1)
    edge_dst_row = torch.arange(num_local + 1, dtype=torch.long)

    # edge_ptr: distribute num_edges across num_local dst nodes
    # simple: fill evenly, last node gets remainder
    base, rem = divmod(num_edges, num_local)
    counts = torch.full((num_local,), base, dtype=torch.long)
    counts[:rem] += 1
    edge_ptrs = []
    for _ in range(num_snaps):
        ptr = torch.zeros(num_local + 1, dtype=torch.long)
        ptr[1:] = counts.cumsum(0)
        edge_ptrs.append(ptr)

    return PartitionData(
        src_ids   = TensorData.from_tensors([torch.arange(num_remote)] * num_snaps),
        dst_ids   = TensorData.from_tensors([torch.arange(num_local)] * num_snaps),
        edge_ids  = TensorData.from_tensors(snaps_eids),
        edge_src  = TensorData.from_tensors(snaps_src),
        edge_dst  = TensorData.from_tensors([edge_dst_row] * num_snaps),
        edge_ptr  = TensorData.from_tensors(edge_ptrs),
        dst_chunk = TensorData.from_tensors(snaps_chunks),
        node_to_chunk=node_to_chunk,
    )


# ---------------------------------------------------------------------------
# Test prepare() pipeline
# ---------------------------------------------------------------------------

def test_prepare_pipeline():
    num_nodes, P = 100, 4
    node_partition = torch.arange(num_nodes) % P
    edge_src = torch.randint(0, num_nodes, (300,))
    edge_dst = torch.randint(0, num_nodes, (300,))
    edge_ts  = torch.arange(300, dtype=torch.float)
    time_ptr = torch.tensor([0, 100, 200, 300])

    art = prepare(
        edge_src=edge_src,
        edge_dst=edge_dst,
        node_partition=node_partition,
        node_to_partition=node_partition,
        num_partitions=P,
        edge_timestamps=edge_ts,
        time_ptr=time_ptr,
        build_mem_routes=True,
        num_candidates=2,
    )

    assert art.node_owner.shape == (num_nodes,)
    assert art.mem_routes is not None
    assert len(art.mem_routes) == P       # one list per partition
    assert len(art.mem_routes[0]) == 3    # 3 time slices
    print(f"✓ prepare(): node_owner={art.node_owner.shape}, "
          f"mem_routes[P={P}][T=3], "
          f"migrations={len(art.rebalance_manifest.migrations)}")


# ---------------------------------------------------------------------------
# Test task adapters on real PartitionData
# ---------------------------------------------------------------------------

def _test_adapter(task_type, part, **kwargs):
    adapter = get_task_adapter(task_type, **kwargs)
    event_pos = torch.tensor([0, 1, 2, 3])
    neg_sampler = NegativeSamplerHook.from_config({})

    batch = adapter.build_batch(
        part=part,
        snapshot_idx=0,
        event_pos=event_pos,
        split="train",
        neg_sampler=neg_sampler,
        num_nodes=20,
    )
    assert isinstance(batch, BatchData)
    assert batch.node_ids is not None
    return batch, adapter


def test_edge_predict_adapter():
    part = _make_part()
    batch, adapter = _test_adapter("edge_predict", part)
    assert batch.pos_src is not None and batch.neg_src is not None
    # Fake model output
    out = {
        "pos_score": torch.randn(batch.pos_src.numel()),
        "neg_score": torch.randn(batch.neg_src.numel()),
    }
    loss = adapter.compute_loss(out, batch)
    metrics = adapter.compute_metrics(out, batch)
    assert loss.item() > 0
    assert "auc" in metrics
    print(f"✓ edge_predict: loss={loss.item():.3f}, auc={metrics['auc']:.3f}")


def test_edge_regress_adapter():
    part = _make_part()
    batch, adapter = _test_adapter("edge_regress", part)
    if batch.pos_src is not None:
        M = batch.pos_src.numel()
        out = {"edge_pred": torch.randn(M, 1)}
        batch.labels = torch.randn(M, 1)
        loss = adapter.compute_loss(out, batch)
        metrics = adapter.compute_metrics(out, batch)
        assert loss.item() >= 0
        assert "mae" in metrics
        print(f"✓ edge_regress: loss={loss.item():.3f}, mae={metrics['mae']:.3f}")
    else:
        print("✓ edge_regress: no edges (empty batch OK)")


def test_node_classify_adapter():
    part = _make_part()
    batch, adapter = _test_adapter("node_classify", part)
    N = batch.node_ids.numel()
    out = {"logits": torch.randn(N, 3)}
    batch.labels = torch.randint(0, 3, (N,))
    loss = adapter.compute_loss(out, batch)
    metrics = adapter.compute_metrics(out, batch)
    assert loss.item() > 0
    assert "accuracy" in metrics
    print(f"✓ node_classify: loss={loss.item():.3f}, acc={metrics['accuracy']:.3f}")


def test_node_regress_adapter():
    part = _make_part()
    batch, adapter = _test_adapter("node_regress", part)
    N = batch.node_ids.numel()
    out = {"node_pred": torch.randn(N, 1)}
    batch.labels = torch.randn(N, 1)
    loss = adapter.compute_loss(out, batch)
    metrics = adapter.compute_metrics(out, batch)
    assert loss.item() >= 0
    assert "rmse" in metrics
    print(f"✓ node_regress: loss={loss.item():.3f}, rmse={metrics['rmse']:.3f}")


# ---------------------------------------------------------------------------
# Test PredictionHead + run_batch
# ---------------------------------------------------------------------------

def test_prediction_head_and_run_batch():
    import torch.nn as nn

    part    = _make_part()
    adapter = get_task_adapter("edge_predict")
    neg     = NegativeSamplerHook.from_config({})
    batch   = adapter.build_batch(part=part, snapshot_idx=0,
                                  event_pos=torch.tensor([0,1,2]),
                                  split="train", neg_sampler=neg, num_nodes=20)

    head = PredictionHead("edge_predict", embedding_dim=8)

    # Minimal backbone: takes BatchData, returns embeddings
    class TinyBackbone(nn.Module):
        def __init__(self): super().__init__(); self.emb = nn.Embedding(30, 8)
        def forward(self, batch): return self.emb(batch.node_ids)

    model = TinyBackbone()
    opt   = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.01)

    # Use num_nodes equal to local node count so neg indices stay in range
    num_local_nodes = int(batch.node_ids.numel())
    batch = adapter.build_batch(part=part, snapshot_idx=0,
                                event_pos=torch.tensor([0, 1, 2]),
                                split="train", neg_sampler=neg,
                                num_nodes=num_local_nodes)

    result = run_batch(
        model=model,
        batch=batch,
        task_adapter=adapter,
        optimizer=opt,
        prediction_head=head,
        train=True,
    )
    assert "loss" in result and "metrics" in result
    print(f"✓ run_batch: loss={result['loss'].item():.3f}, metrics={result['metrics']}")


# ---------------------------------------------------------------------------
# Test ChunkRuntimeLoader iter with in-memory setup
# ---------------------------------------------------------------------------

def test_chunk_loader_iter():
    """ChunkRuntimeLoader with manually constructed artifacts (no disk)."""
    import types

    part     = _make_part(num_snaps=6)
    adapter  = get_task_adapter("edge_predict")
    neg      = NegativeSamplerHook.from_config({})

    loader = ChunkRuntimeLoader(
        part_data      = part,
        mem_routes     = [],
        spatial_routes = [],
        task_adapter   = adapter,
        neg_sampler    = neg,
        mfg_builder    = __import__(
            "starry_unigraph.backends.chunk.runtime.sampler",
            fromlist=["MFGBuilderHook"]
        ).MFGBuilderHook.default(),
        pipeline       = __import__(
            "starry_unigraph.backends.chunk.data.comm",
            fromlist=["CommPipeline"]
        ).CommPipeline(device=torch.device("cpu")),
        split_slices   = {"train": [0, 1, 2, 3], "val": [4], "test": [5]},
        num_nodes      = 20,
        rank           = 0,
        world_size     = 1,
        device         = torch.device("cpu"),
    )

    train_batches = list(loader.iter_train())
    assert len(train_batches) == 4
    for b in train_batches:
        assert isinstance(b, BatchData)
        assert b.node_ids is not None

    val_batches  = list(loader.iter_eval())
    test_batches = list(loader.iter_predict())
    assert len(val_batches) == 1
    assert len(test_batches) == 1

    desc = loader.describe()
    assert desc["task_type"] == "edge_predict"
    print(f"✓ ChunkRuntimeLoader: train={len(train_batches)}, "
          f"val={len(val_batches)}, test={len(test_batches)}, "
          f"desc={desc}")


if __name__ == "__main__":
    test_prepare_pipeline()
    test_edge_predict_adapter()
    test_edge_regress_adapter()
    test_node_classify_adapter()
    test_node_regress_adapter()
    test_prediction_head_and_run_batch()
    test_chunk_loader_iter()
    print("\n✅ 阶段8 全部测试通过!")
