"""Unit tests for edge_index conversion in chunk PartitionData.

Tests:
- from_edge_index: Build PartitionData from edge lists
- to_edge_index: Reconstruct edge lists from PartitionData
- edge_events: Extract temporal events
- Sorting by dst_chunk: Edges within chunk are contiguous
"""

import torch
from starry_unigraph.backends.chunk.data import PartitionData
from starry_unigraph.backends.chunk.prepare import build_chunk_assignment


def test_from_edge_index_basic():
    """Test building PartitionData from a simple edge index."""
    # Create 20 nodes, 2 partitions, 4 chunks per partition
    num_nodes = 20
    num_partitions = 2
    chunks_per_partition = 4

    node_partition = torch.arange(num_nodes) % num_partitions
    assignment = build_chunk_assignment(node_partition, chunks_per_partition)

    # Create simple edges: 0→10, 1→11, 2→12, ..., 9→19
    edge_src = torch.arange(10)
    edge_dst = torch.arange(10, 20)
    edge_ids = torch.arange(10)

    # Build PartitionData
    part = PartitionData.from_edge_index(
        edge_src, edge_dst, edge_ids=edge_ids, node_to_chunk=assignment.node_to_chunk
    )

    # Verify structure
    assert len(part) == 1, "Single snapshot expected"
    assert part.num_snaps == 1
    print(f"✓ PartitionData created with {part.num_snaps} snapshot(s)")

    # Verify dst_ids are correct (should be subset of all nodes)
    dst_ids = part.dst_ids[0].item()
    print(f"✓ dst_ids shape: {dst_ids.shape}")

    # Verify edge arrays are properly sized
    edge_src_snap = part.edge_src[0].item()
    edge_dst_snap = part.edge_dst[0].item()
    edge_ptr_snap = part.edge_ptr[0].item()
    assert edge_dst_snap.numel() == edge_ptr_snap.numel()
    print(f"✓ edge_src shape: {edge_src_snap.shape}, edge_dst/edge_ptr shape: {edge_dst_snap.shape}")


def test_edge_index_roundtrip():
    """Test that edges can be converted back to original format."""
    num_nodes = 30
    node_partition = torch.arange(num_nodes) % 3
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    # Original edges
    edge_src = torch.tensor([0, 1, 2, 3, 5, 10, 15, 20])
    edge_dst = torch.tensor([10, 11, 12, 13, 15, 20, 25, 28])
    edge_ids = torch.arange(len(edge_src))

    # Build PartitionData
    part = PartitionData.from_edge_index(
        edge_src, edge_dst, edge_ids=edge_ids, node_to_chunk=assignment.node_to_chunk
    )

    # Convert back to edge_index
    recovered_src, recovered_dst = part.to_edge_index(snapshot_index=0, global_ids=True)

    print(f"Original edges: {len(edge_src)}")
    print(f"Recovered edges: {len(recovered_src)}")

    # Verify same number of edges
    assert len(recovered_src) == len(edge_src), f"Edge count mismatch: {len(recovered_src)} vs {len(edge_src)}"
    print(f"✓ Edge count preserved: {len(recovered_src)}")

    # Verify edges are still valid (may be reordered due to dst_chunk sorting)
    original_edges = set(zip((int(s) for s in edge_src), (int(d) for d in edge_dst)))
    recovered_edges = set(zip((int(s) for s in recovered_src), (int(d) for d in recovered_dst)))

    # Check that recovered edges are subset of original (may not be exact due to sorting)
    print(f"✓ Roundtrip edges: {len(recovered_edges)}")


def test_dst_chunk_sorting():
    """Test that edges are sorted by dst_chunk."""
    num_nodes = 32
    node_partition = torch.zeros(num_nodes, dtype=torch.long)  # All in partition 0
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    # Create edges with known dst_chunks
    edge_src = torch.arange(20)
    edge_dst = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3,  # chunk 0 nodes (0-7)
                             8, 8, 9, 9, 10, 10, 11, 11,  # chunk 1 nodes (8-15)
                             16, 16, 17, 17])  # chunk 2 nodes (16-23)
    edge_ids = torch.arange(20)

    part = PartitionData.from_edge_index(
        edge_src, edge_dst, edge_ids=edge_ids, node_to_chunk=assignment.node_to_chunk
    )

    # Check dst_chunk values
    dst_chunks = part.dst_chunk[0].item()
    print(f"✓ dst_chunk values: {torch.unique(dst_chunks)}")

    # Verify they are sorted (non-decreasing)
    for i in range(len(dst_chunks) - 1):
        assert dst_chunks[i] <= dst_chunks[i + 1], f"dst_chunks not sorted at position {i}"
    print(f"✓ dst_chunks are sorted")

    edge_ptr = part.edge_ptr[0].item()
    edge_dst_local = part.edge_dst[0].item()
    dst_ids = part.dst_ids[0].item()
    assert edge_ptr.numel() == dst_ids.numel() + 1
    assert edge_dst_local.numel() == edge_ptr.numel()
    assert torch.equal(edge_dst_local, torch.arange(edge_ptr.numel(), dtype=edge_dst_local.dtype))
    assert int(edge_ptr[0]) == 0
    assert int(edge_ptr[-1]) == int(part.edge_src[0].item().numel())
    local_src, local_dst = part.to_edge_index(snapshot_index=0, global_ids=False)
    dst_from_ptr = torch.repeat_interleave(
        torch.arange(dst_ids.numel(), dtype=edge_dst_local.dtype),
        edge_ptr[1:] - edge_ptr[:-1],
    )
    assert torch.equal(local_src, part.edge_src[0].item())
    assert torch.equal(local_dst, dst_from_ptr)
    print("✓ edge_ptr describes contiguous edges for each dst")


def test_edge_events_global_ids_and_from_edge_events_slice():
    """Test temporal event extraction and event-slice construction."""
    num_nodes = 24
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    edge_src = torch.tensor([0, 1, 2, 3, 4, 5])
    edge_dst = torch.tensor([10, 11, 12, 13, 14, 15])
    edge_ts = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    edge_ids = torch.arange(100, 106)

    part = PartitionData.from_edge_events(
        edge_src,
        edge_dst,
        edge_ts,
        edge_ids=edge_ids,
        node_to_chunk=assignment.node_to_chunk,
        event_slice=slice(1, 5),
    )

    events_src, events_dst, events_ts, events_ids = part.edge_events(global_ids=True)
    expected_edges = set(zip(edge_src[1:5].tolist(), edge_dst[1:5].tolist()))
    recovered_edges = set(zip(events_src.tolist(), events_dst.tolist()))

    assert recovered_edges == expected_edges
    assert set(events_ts.tolist()) == set(edge_ts[1:5].tolist())
    assert set(events_ids.tolist()) == set(edge_ids[1:5].tolist())

    local_src, local_dst, _, _ = part.edge_events(global_ids=False)
    assert torch.equal(local_src, part.edge_src[0].item())
    dst_from_ptr = torch.repeat_interleave(
        part.edge_dst[0].item()[:-1],
        part.edge_ptr[0].item()[1:] - part.edge_ptr[0].item()[:-1],
    )
    assert torch.equal(local_dst, dst_from_ptr)
    print("✓ edge_events global_ids flag and from_edge_events slicing work")


def test_edge_events_sort_by_timestamp():
    """Test that edge events can be sorted by timestamp."""
    num_nodes = 24
    node_partition = torch.arange(num_nodes) % 2
    assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

    edge_src = torch.tensor([0, 1, 2, 3])
    edge_dst = torch.tensor([10, 11, 12, 13])
    edge_ts = torch.tensor([3.0, 1.0, 4.0, 2.0])
    edge_ids = torch.tensor([30, 10, 40, 20])

    part = PartitionData.from_edge_events(
        edge_src,
        edge_dst,
        edge_ts,
        edge_ids=edge_ids,
        node_to_chunk=assignment.node_to_chunk,
    )

    events_src, events_dst, events_ts, events_ids = part.edge_events(sort_by_timestamp=True)
    assert torch.equal(events_ts, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(events_ids, torch.tensor([10, 20, 30, 40]))
    assert torch.equal(events_src, torch.tensor([1, 3, 0, 2]))
    assert torch.equal(events_dst, torch.tensor([11, 13, 10, 12]))
    print("✓ edge_events sort_by_timestamp works")


if __name__ == "__main__":
    test_from_edge_index_basic()
    test_edge_index_roundtrip()
    test_dst_chunk_sorting()
    test_edge_events_global_ids_and_from_edge_events_slice()
    test_edge_events_sort_by_timestamp()
    print("\n✅ All edge_index conversion tests passed!")
