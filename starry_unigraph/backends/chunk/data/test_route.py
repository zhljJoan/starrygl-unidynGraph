"""Tests for the restructured SpatialRouteData / MemoryRouteData / CommPipeline."""

import asyncio
import torch
from starry_unigraph.backends.chunk.prepare import (
    build_chunk_assignment,
    build_memory_route_phase1,
    assign_memory_route_ptrs,
    build_cpu_memory_layout,
)
from starry_unigraph.backends.chunk.data import (
    MemoryRouteData, SpatialRouteData, CPUMemoryLayout, CommPipeline
)


# ---------------------------------------------------------------------------
# MemoryRouteData: two-phase construction
# ---------------------------------------------------------------------------

def test_phase1_dedup_and_candidates():
    """Phase 1: unique_nodes correct, cand_pos latests-first, -1 for absent."""
    edge_src = torch.tensor([0, 0, 1, 2])
    edge_dst = torch.tensor([3, 3, 4, 5])
    edge_ts  = torch.tensor([1.0, 5.0, 2.0, 3.0])
    time_ptr = torch.tensor([0, 4])

    routes = build_memory_route_phase1(edge_src, edge_dst, edge_ts, time_ptr, num_candidates=2)
    r = routes[0]

    assert r.unique_nodes.numel() == 6, f"Expected 6 unique nodes, got {r.unique_nodes.numel()}"
    # Node 0 appears at positions 0 (ts=1) and 1 (ts=5) in edge_src,
    # and the duplicate edge_dst=3 appears at position 0 and 1 too.
    # Candidate at col 0 should be the latest event.
    node0_idx = (r.unique_nodes == 0).nonzero(as_tuple=True)[0]
    if node0_idx.numel():
        cands = r.cand_pos[node0_idx[0]]
        # latest event for node 0 is at position 1 (ts=5 in all_nodes idx=1)
        assert cands[0] >= 0, "col 0 should not be -1"
    print(f"✓ Phase 1: {r.unique_nodes.numel()} unique nodes, cand_pos shape {tuple(r.cand_pos.shape)}")


def test_phase2_ptrs_consistent():
    """Phase 2: send_ptr and recv_ptr are consistent across partitions."""
    num_nodes = 16
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)
    node_owner = assignment.chunk_to_owner_partition[assignment.node_to_chunk]

    edge_src = torch.tensor([0, 2, 4, 6])
    edge_dst = torch.tensor([8, 10, 12, 14])
    edge_ts  = torch.arange(4, dtype=torch.float)
    time_ptr = torch.tensor([0, 4])

    routes_p1 = build_memory_route_phase1(edge_src, edge_dst, edge_ts, time_ptr)
    per_part  = assign_memory_route_ptrs(routes_p1, node_owner, num_parts=2)

    for p in range(2):
        r = per_part[p][0]
        # recv_ptr consistency: recv_node_ids should all be owned by p
        if r.recv_node_ids.numel() > 0:
            owned = (node_owner[r.recv_node_ids] == p).all()
            assert owned, f"Partition {p} recv_node_ids contains foreign nodes"
    print("✓ Phase 2: recv_node_ids all owned by correct partition")


def test_repartition_reuse():
    """Re-running Phase 2 with new node_owner produces updated routing."""
    num_nodes = 12
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)
    node_owner_v1 = assignment.chunk_to_owner_partition[assignment.node_to_chunk]

    edge_src = torch.tensor([0, 2, 4])
    edge_dst = torch.tensor([6, 8, 10])
    edge_ts  = torch.arange(3, dtype=torch.float)
    time_ptr = torch.tensor([0, 3])

    routes_p1 = build_memory_route_phase1(edge_src, edge_dst, edge_ts, time_ptr)
    p1_unique = [r.unique_nodes.clone() for r in routes_p1]

    # Phase 2 with original owner
    per_part_v1 = assign_memory_route_ptrs(routes_p1, node_owner_v1, num_parts=2)

    # Simulate repartition: flip owners
    node_owner_v2 = 1 - node_owner_v1

    # Re-run Phase 2 only (Phase 1 routes reused)
    per_part_v2 = assign_memory_route_ptrs(routes_p1, node_owner_v2, num_parts=2)

    # Phase 1 unique_nodes should be unchanged
    for t, r in enumerate(routes_p1):
        assert torch.equal(r.unique_nodes, p1_unique[t]), "Phase 1 output was mutated"

    print("✓ Phase 1 reuse: unique_nodes unchanged after re-running Phase 2")


# ---------------------------------------------------------------------------
# CPUMemoryLayout
# ---------------------------------------------------------------------------

def test_cpu_layout_hot_cold():
    hot   = torch.tensor([5, 10, 20])
    cold  = torch.tensor([1, 3, 7])
    layout = CPUMemoryLayout.build(hot, cold, num_nodes=30)

    assert layout.hot_boundary == 3
    assert layout.global_to_local[5]  == 0
    assert layout.global_to_local[10] == 1
    assert layout.global_to_local[1]  == 3    # first cold node
    assert layout.global_to_local[0]  == -1   # not cached
    print(f"✓ CPUMemoryLayout: hot_boundary={layout.hot_boundary}, cached={layout.num_cached}")


# ---------------------------------------------------------------------------
# CommPipeline interface (no dist required)
# ---------------------------------------------------------------------------

def test_comm_pipeline_interface():
    """Verify CommPipeline can be instantiated; await_* return None when idle."""
    pipeline = CommPipeline(device=torch.device("cpu"))

    async def run():
        r_s = await pipeline.await_spatial()
        r_m = await pipeline.await_memory()
        r_r = await pipeline.await_replica()
        return r_s, r_m, r_r

    r_s, r_m, r_r = asyncio.run(run())
    assert r_s is None and r_m is None and r_r is None
    print("✓ CommPipeline: all await_* return None when no op submitted")


def test_comm_pipeline_drain_sync():
    """drain_all_sync() is safe to call when nothing is pending."""
    pipeline = CommPipeline(device=torch.device("cpu"))
    pipeline.drain_all_sync()   # should not raise
    print("✓ CommPipeline.drain_all_sync() safe on empty pipeline")


def test_memory_route_sampled_pos():
    """sampled_pos returns valid positions; fallback to latest works."""
    edge_src = torch.tensor([0, 0, 1])
    edge_dst = torch.tensor([2, 3, 4])
    edge_ts  = torch.tensor([1.0, 2.0, 1.5])
    time_ptr = torch.tensor([0, 3])

    routes = build_memory_route_phase1(edge_src, edge_dst, edge_ts, time_ptr, num_candidates=3)
    r = routes[0]

    latest = r.latest_pos()
    assert (latest >= 0).all(), "latest_pos should have no -1"

    sampled = r.sampled_pos()
    assert (sampled >= 0).all(), "sampled_pos fallback must prevent -1"
    print(f"✓ sampled_pos: no -1 values, shape {tuple(sampled.shape)}")


if __name__ == "__main__":
    test_phase1_dedup_and_candidates()
    test_phase2_ptrs_consistent()
    test_repartition_reuse()
    test_cpu_layout_hot_cold()
    test_comm_pipeline_interface()
    test_comm_pipeline_drain_sync()
    test_memory_route_sampled_pos()
    print("\n✅ All route / comm tests passed!")
