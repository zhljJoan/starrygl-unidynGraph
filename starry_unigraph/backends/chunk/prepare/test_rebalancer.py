"""Unit tests for Phase 4: chunk load stats and rebalancing."""

import torch
from starry_unigraph.backends.chunk.prepare import (
    build_chunk_assignment,
    compute_chunk_load_by_slice,
    compute_chunk_load_stats,
    compute_chunk_load_stats_from_windows,
    rebalance_chunks,
    derive_node_owner,
    ChunkLoadStats,
)


def test_compute_load_stats():
    """Load stats accumulate edges and remote-edge counts correctly."""
    # 12 nodes, 2 partitions (even → part0, odd → part1), 2 chunks each
    num_nodes = 12
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)

    # Edges: 0→1 (cross-partition), 2→3 (cross), 0→2 (same), 1→3 (same)
    edge_src = torch.tensor([0, 2, 0, 1])
    edge_dst = torch.tensor([1, 3, 2, 3])

    stats = compute_chunk_load_stats(
        edge_src, edge_dst,
        assignment.node_to_chunk,
        assignment.chunk_to_owner_partition,
        node_partition,
    )

    total_edges = sum(s.edge_count for s in stats.values())
    assert total_edges == 4, f"Expected 4, got {total_edges}"

    total_remote = sum(s.remote_edge_count for s in stats.values())
    assert total_remote == 2, f"Expected 2 remote edges (0→1, 2→3), got {total_remote}"

    print(f"✓ compute_load_stats: {total_edges} edges, {total_remote} remote")


def test_rebalance_chunks_reduces_imbalance():
    """Greedy rebalancer moves heavy chunks to lighter partitions."""
    num_nodes = 40
    num_partitions = 4
    node_partition = torch.arange(num_nodes) % num_partitions
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    # Artificially inflate load on partition-0 chunks
    load_stats = {}
    for cid in range(assignment.total_chunks):
        owner = int(assignment.chunk_to_initial_partition[cid])
        load = 1000.0 if owner == 0 else 100.0
        stat = ChunkLoadStats(chunk_id=cid, edge_count=int(load))
        stat.total_load = load
        load_stats[cid] = stat

    initial_imbalance = max(load_stats[c].total_load for c in load_stats) / (
        min(load_stats[c].total_load for c in load_stats) + 1e-9
    )

    updated, node_owner, manifest = rebalance_chunks(
        assignment, load_stats, num_partitions=num_partitions
    )

    assert len(manifest.migrations) > 0, "Expected migrations to occur"
    assert manifest.imbalance_after < manifest.imbalance_before, (
        f"Expected reduced imbalance: {manifest.imbalance_before:.2f} → {manifest.imbalance_after:.2f}"
    )

    print(f"✓ rebalance: {len(manifest.migrations)} migrations, "
          f"imbalance {manifest.imbalance_before:.2f} → {manifest.imbalance_after:.2f}")


def test_chunk_load_by_slice_and_vector_rebalance():
    """Per-slice load matrix drives vector-aware chunk ownership."""
    num_nodes = 8
    num_partitions = 2
    node_partition = torch.arange(num_nodes) % num_partitions
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)

    window_src = [torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7])]
    window_dst = [torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7])]
    load = compute_chunk_load_by_slice(
        window_src,
        window_dst,
        assignment.node_to_chunk,
        assignment.chunk_to_owner_partition,
        node_partition,
    )
    assert load.shape == (2, assignment.total_chunks)

    load_stats = {cid: ChunkLoadStats(chunk_id=cid, edge_count=1, total_load=1.0) for cid in range(assignment.total_chunks)}
    updated, node_owner, manifest = rebalance_chunks(
        assignment,
        load_stats,
        num_partitions=num_partitions,
        max_imbalance_ratio=1.0,
        chunk_load_by_slice=load,
    )
    assert node_owner.shape == (num_nodes,)
    assert manifest.num_chunks == assignment.total_chunks
    print(f"✓ vector rebalance: chunk_load_by_slice={tuple(load.shape)}, migrations={len(manifest.migrations)}")


def test_derive_node_owner():
    """node_owner correctly follows chunk_to_owner_partition."""
    num_nodes = 20
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)

    # Before rebalance: node_owner == initial partition
    node_owner = derive_node_owner(assignment)
    assert node_owner.shape == (num_nodes,)

    for nid in range(num_nodes):
        expected = int(node_partition[nid])
        actual = int(node_owner[nid])
        assert actual == expected, f"Node {nid}: expected {expected}, got {actual}"

    print(f"✓ derive_node_owner: all {num_nodes} nodes correct before rebalance")

    # Manually migrate chunk 0 to partition 1
    assignment.chunk_to_owner_partition[0] = 1
    node_owner_new = derive_node_owner(assignment)

    for nid in assignment.chunk_to_nodes[0]:
        assert int(node_owner_new[nid]) == 1, f"Node {nid} should now be in partition 1"

    print(f"✓ derive_node_owner: migration of chunk 0 reflected correctly")


def test_rebalance_chunk_assignment_dict_interface():
    """rebalance_chunk_assignment accepts plain dicts too."""
    from starry_unigraph.backends.chunk.prepare import rebalance_chunk_assignment

    num_nodes = 20
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=2)

    # Plain-dict load stats (legacy-compatible)
    load_stats = {
        cid: {"edges": 500 if cid < 2 else 50, "active_nodes": 10}
        for cid in range(assignment.total_chunks)
    }

    result = rebalance_chunk_assignment(assignment, load_stats)
    assert result is assignment  # mutated in-place
    print("✓ rebalance_chunk_assignment dict interface works")


if __name__ == "__main__":
    test_compute_load_stats()
    test_rebalance_chunks_reduces_imbalance()
    test_chunk_load_by_slice_and_vector_rebalance()
    test_derive_node_owner()
    test_rebalance_chunk_assignment_dict_interface()
    print("\n✅ All Phase 4 tests passed!")
