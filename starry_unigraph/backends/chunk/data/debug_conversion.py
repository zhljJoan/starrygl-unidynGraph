"""Debug test for edge_index conversion."""

import torch
from starry_unigraph.backends.chunk.data.edge_index_utils import (
    partition_nodes_by_locality,
    sort_edges_by_dst_chunk,
    snapshot_to_partitiondata_tensors,
)
from starry_unigraph.backends.chunk.prepare import build_chunk_assignment

# Simple test case
num_nodes = 20
node_partition = torch.arange(num_nodes) % 2
assignment = build_chunk_assignment(node_partition, num_chunks_per_partition=4)

edge_src = torch.arange(10)
edge_dst = torch.arange(10, 20)
edge_ids = torch.arange(10)

print(f"Original edges: src={edge_src}, dst={edge_dst}")
print(f"node_to_chunk: {assignment.node_to_chunk}")

# Step 1: Sort by dst_chunk
sorted_src, sorted_dst, sorted_ts, sorted_ids, dst_chunks = sort_edges_by_dst_chunk(
    edge_src, edge_dst, None, edge_ids, assignment.node_to_chunk
)
print(f"\nAfter sorting by dst_chunk:")
print(f"  sorted_src: {sorted_src}")
print(f"  sorted_dst: {sorted_dst}")
print(f"  dst_chunks: {dst_chunks}")

# Step 2: Partition nodes
src_ids, dst_ids = partition_nodes_by_locality(sorted_src, sorted_dst)
print(f"\nPartitioning nodes:")
print(f"  src_ids (combined): {src_ids}")
print(f"  dst_ids (local): {dst_ids}")

# Verify all sorted_src are in src_ids
all_in_src = all(int(s) in [int(x) for x in src_ids] for s in sorted_src)
print(f"\nAll sorted_src in src_ids? {all_in_src}")
if not all_in_src:
    missing = [int(s) for s in sorted_src if int(s) not in [int(x) for x in src_ids]]
    print(f"  Missing nodes: {missing}")
