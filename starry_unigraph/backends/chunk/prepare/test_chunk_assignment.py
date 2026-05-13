"""Unit tests for chunk assignment logic.

Tests:
- build_chunk_assignment: Initialize chunk maps from node partitions
- get_node_owner: Retrieve partition owning a node
- get_nodes_in_partition: Retrieve all nodes in a partition
"""

import torch
from starry_unigraph.backends.chunk.prepare import ChunkAssignment, build_chunk_assignment


def test_build_chunk_assignment():
    """Test building chunk assignment from node partition."""
    # Create 100 nodes, 4 partitions, 8 chunks per partition
    num_nodes = 100
    num_partitions = 4
    chunks_per_partition = 8

    # Simple round-robin node partition
    node_partition = torch.arange(num_nodes) % num_partitions

    # Build assignment
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=chunks_per_partition)

    # Verify totals
    assert assignment.total_nodes == num_nodes
    assert assignment.total_chunks == num_partitions * chunks_per_partition

    print(f"✓ ChunkAssignment created: {num_nodes} nodes, {assignment.total_chunks} chunks")

    # Verify node_to_chunk is complete
    assert int(assignment.node_to_chunk.max()) < assignment.total_chunks
    assert int(assignment.node_to_chunk.min()) >= 0

    print(f"✓ node_to_chunk range: [0, {int(assignment.node_to_chunk.max())}]")

    # Verify chunk_to_nodes covers all nodes
    all_nodes_in_chunks = []
    for chunk_id in range(assignment.total_chunks):
        all_nodes_in_chunks.extend(assignment.chunk_to_nodes[chunk_id])
    all_nodes_in_chunks = sorted(all_nodes_in_chunks)
    expected_nodes = list(range(num_nodes))
    assert all_nodes_in_chunks == expected_nodes, f"Mismatch: {len(all_nodes_in_chunks)} vs {len(expected_nodes)}"

    print(f"✓ All {num_nodes} nodes assigned to chunks")

    # Verify chunk_to_initial_partition consistency
    for chunk_id in range(assignment.total_chunks):
        partition_id = int(assignment.chunk_to_initial_partition[chunk_id])
        expected_partition = chunk_id // chunks_per_partition
        assert partition_id == expected_partition, f"Chunk {chunk_id}: expected partition {expected_partition}, got {partition_id}"

    print(f"✓ chunk_to_initial_partition is consistent")

    # Verify get_node_owner
    for node_id in range(min(10, num_nodes)):
        expected_partition = int(node_partition[node_id])
        owner = assignment.get_node_owner(node_id)
        assert owner == expected_partition, f"Node {node_id}: expected owner {expected_partition}, got {owner}"

    print(f"✓ get_node_owner returns correct partition")


def test_chunk_rebalancing_placeholder():
    """Test that load stats can be stored and normalised to ChunkLoadStats."""
    num_nodes = 50
    num_partitions = 2
    node_partition = torch.arange(num_nodes) % num_partitions

    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    # Simulate plain-dict load stats (legacy-compatible interface)
    load_stats = {}
    for chunk_id in range(assignment.total_chunks):
        load_stats[chunk_id] = {"edges": 100 + chunk_id * 10, "active_nodes": 5 + chunk_id}

    from starry_unigraph.backends.chunk.prepare import rebalance_chunk_assignment
    from starry_unigraph.backends.chunk.prepare import ChunkLoadStats

    rebalanced = rebalance_chunk_assignment(assignment, load_stats)

    # Stats are normalised to ChunkLoadStats objects
    assert len(rebalanced.chunk_load_stats) == len(load_stats)
    for cid, stat in rebalanced.chunk_load_stats.items():
        assert isinstance(stat, ChunkLoadStats)

    print(f"✓ Load stats stored and normalised: {len(rebalanced.chunk_load_stats)} chunks")


if __name__ == "__main__":
    test_build_chunk_assignment()
    test_chunk_rebalancing_placeholder()
    print("\n✅ All tests passed!")
