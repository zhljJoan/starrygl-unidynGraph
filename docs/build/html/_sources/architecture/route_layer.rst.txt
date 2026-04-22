Route Layer Reference
=====================

The route layer abstracts distributed feature exchange across partitions
in multi-GPU/multi-machine training. It decouples batch materialization
(how we gather data) from physical data movement (how we move it in a cluster).

Overview
--------

Routes handle the communication pattern for gathering temporal neighbors:

- In **DTDG** (snapshots): Routes specify which nodes need features from remote GPUs
  per snapshot, enabling all-to-all collectives
- In **CTDG** (online): Routes embed replica/partition info into memory banks,
  avoiding explicit communication
- In **Chunk**: Routes manage cross-cluster edge forwarding

All modes use ``RouteData`` as the unified schema, but differ in how they
interpret and use it.

Core Concepts
-------------

**Feature vs. Structure**

In temporal graphs, node features can be distributed differently from the graph
structure:

- **Feature Distribution**: Which GPU holds node feature vectors (replica or partition)
- **Structure Distribution**: Which GPU holds the outgoing edges for a node

Routes connect these by specifying: "To compute embeddings for node X on GPU G,
fetch features for nodes Y₁, Y₂, ... from remote GPUs."

**Communication Patterns**

.. list-table:: Route Communication Patterns
   :header-rows: 1

   * - Mode
     - Pattern
     - Materialization
     - Use Case

   * - **DTDG**
     - Per-snapshot all-to-all
     - Blocking (sync all GPUs)
     - Snapshot-based sampling

   * - **CTDG**
     - Online embedding lookup
     - Asynchronous (event-driven)
     - Online event streams

   * - **Chunk**
     - Cross-cluster async
     - Event-driven or batch
     - Time-windowed processing

``RouteData`` — Unified Schema
-------------------------------

All modes use this dataclass to represent routing requirements:

.. code-block:: python

    @dataclass
    class RouteData:
        """Distributed feature exchange metadata."""

        # Which nodes to fetch from remote GPUs
        send_index: Tensor | None      # [dst_count] → ranks
        recv_index: Tensor | None      # [src_count] → local IDs

        # How many nodes per remote rank
        send_count: List[int]          # nodes sent to each rank
        recv_count: List[int]          # nodes received from each rank

        # DGL Block integration
        route: "Route" | None          # Route instance (DTDG only)

Construction:

.. code-block:: python

    routes = RouteData(
        send_index=torch.tensor([0, 0, 1, 1]),  # Send to rank 0,0,1,1
        recv_index=torch.tensor([15, 42, 7]),   # Recv from nodes 15,42,7
        send_count=[2, 2],                       # 2 nodes to each rank
        recv_count=[3],                          # 3 nodes from remote
        route=None                               # CTDG doesn't use this
    )

Mode-Specific Usage
-------------------

**DTDG: Per-Snapshot Routes**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTDG creates one ``Route`` per snapshot, describing how features move:

.. code-block:: python

    class Route:
        """One snapshot's feature routing (DTDG)."""

        @property
        def send_index(self) -> Tensor | None:
            """Which remote rank needs each local node's features."""
            ...

        @property
        def recv_index(self) -> Tensor | None:
            """Which remote nodes to fetch features for (by local ID)."""
            ...

        def forward(self, features: Tensor) -> Tensor:
            """Execute all-to-all exchange. [N_local, F] → [N_fetched, F]"""
            ...

Integration with DGL blocks:

.. code-block:: python

    block = dgl.create_block(...)
    block.route = route_for_this_snapshot

    # Inside GCN layer
    def _gcn_message_pass(graph, x):
        if graph.is_block and hasattr(graph, 'route'):
            src_x = graph.route.forward(x)  # Fetch remote features
        else:
            src_x = x  # Local only
        # ... continue message passing with src_x

**CTDG: Distributed Memory Routing with NCCL**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CTDG supports both **single-machine multi-GPU** and **multi-machine distributed** training via node partitioning and NCCL-based memory synchronization:

.. code-block:: python

    class CTDGMemoryBank:
        """Per-rank partitioned node memory with NCCL all-to-all sync."""

        def __init__(self, num_nodes: int, node_partition_map: Tensor, rank: int, world_size: int):
            self.node_partition_map = node_partition_map  # node_id → rank/GPU
            self.rank = rank
            self.world_size = world_size
            # Local storage for nodes assigned to this rank
            self.num_local_nodes = (node_partition_map == rank).sum()
            self.local_memory = Memory[num_local_nodes, dim]  # local node embeddings

        def fetch(self, node_ids: Tensor):
            """Get memory for nodes, syncing remote if needed via NCCL all-to-all."""
            local_mask = self.node_partition_map[node_ids] == self.rank
            result = zeros_like(node_ids)
            result[local_mask] = self.local_memory[global2local[local_mask]]
            if (~local_mask).any():
                result[~local_mask] = self._sync_remote_nccl(node_ids[~local_mask])
            return result

        def _sync_remote_nccl(self, remote_node_ids: Tensor):
            """Synchronize remote node features via NCCL all-to-all_single.

            Each rank sends its local embeddings to remote ranks via collective:
            - build send_counts/recv_counts based on node ownership
            - all ranks exchange: dist.all_to_all_single(recv_buf, send_buf, ...)
            - gather results from all ranks
            """
            send_counts, recv_counts = self._exchange_counts(remote_node_ids)
            recv_buffer = torch.zeros(sum(recv_counts), dim, device=device)
            send_buffer = self._pack_local_embeddings(send_counts)

            # All-to-all on all ranks (works single-machine and multi-machine)
            dist.all_to_all_single(recv_buffer, send_buffer,
                                  recv_split_sizes=recv_counts,
                                  input_split_sizes=send_counts)
            return self._unpack_received_buffer(recv_buffer)

**Communication Pattern**:

Both **single-machine (GPU-to-GPU)** and **multi-machine (machine-to-machine)** use the same **NCCL collective all-to-all_single**:

- Single-machine: All ranks (GPUs) on same PCIe/NVLink fabric → ~100-500 µs
- Multi-machine: All ranks across network → ~10-50 ms (backend: nccl-over-tcp)

**Key Feature**: The communication pattern is **identical** in both cases. Only the network layer differs.

**Chunk: Async Cross-Cluster Routing**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chunk mode uses async messaging to forward cross-cluster edges:

.. code-block:: python

    @dataclass
    class ChunkRoute:
        """Async messaging for cross-cluster edges."""
        remote_cluster_ids: Tensor  # Which clusters own neighbors
        remote_node_ids: Tensor     # Local IDs in remote clusters
        message_handlers: Dict[int, Callable]  # Cluster → handler

Implementation:

.. code-block:: python

    def materialize_chunk(chunk: ChunkAtomic):
        local_edges = chunk.local_tcsr
        for remote_edge in chunk.remote_edges:
            cluster_id, node_id = decode_remote_edge(remote_edge)
            routes[cluster_id].send_message(
                "fetch_features",
                payload={"node_ids": node_id}
            )
        # Wait for responses (async or blocking)

Route Construction Pipeline
----------------------------

Routes are built during preprocessing:

1. **DTDG**: ``dtdg_prepare.py:build_flare_partition_data_list()``

   .. code-block:: python

       for partition_id, partition in enumerate(partitions):
           routes_for_partition = []
           for snapshot in partition.snapshots:
               # Determine which nodes need features from other GPUs
               route = build_snapshot_route(
                   snapshot.edges,
                   partition.node_map,
                   global_node_map
               )
               routes_for_partition.append(route)

2. **CTDG**: ``preprocess.py:build_partitions()``

   .. code-block:: python

       node_partition_map = torch.zeros(num_nodes)
       for partition_id, nodes in enumerate(partitions):
           node_partition_map[nodes] = partition_id
       # No explicit RouteData; partition map is the route

3. **Chunk**: ``chunk_builder.py:ChunkBuilder.build()``

   .. code-block:: python

       for chunk in chunks:
           # Extract cross-cluster edges
           for (u, v) in chunk.remote_edges:
               cluster_owner = get_cluster(v)
               chunk.routes[cluster_owner].add(v)

Implementation Details: All-to-All Exchange
--------------------------------------------

DTDG's all-to-all uses torch.distributed collectives:

.. code-block:: python

    def route_forward(features: Tensor, send_index: Tensor):
        """All-to-all single for distributed training.

        send_index tells each GPU which features to send to which ranks.
        Returns: stacked features from all ranks.
        """
        # Typical usage:
        # GPU 0: features[0,1] → rank 1; features[2] → rank 2
        # GPU 1: features[3,4] → rank 0; features[5,6] → rank 2
        # GPU 2: features[7] → rank 0; features[8] → rank 1

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Gather: each GPU sends its features to remote GPUs
        dist.all_to_all_single(
            output=output_buffer,  # Pre-allocated for recv_count total features
            input=input_buffer,    # Features indexed by send_index
            output_split_sizes=send_count,
            input_split_sizes=recv_count,
        )
        return output_buffer

Comparing DTDG vs CTDG Routing
-------------------------------

**Key Insight**: Both CTDG and DTDG use NCCL **all-to-all_single** for multi-GPU/multi-machine communication.
The difference is what data they exchange and when.

.. list-table:: Routing Strategy Comparison
   :header-rows: 1

   * - Aspect
     - DTDG
     - CTDG

   * - **What**
     - Graph features (per-snapshot edges)
     - Node embeddings (temporal memory)

   * - **When**
     - Per snapshot
     - On-demand per batch (when node needed)

   * - **Exchange Method**
     - Route.forward() → NCCL all-to-all_single
     - Memory.fetch() → NCCL all-to-all_single

   * - **Communication**
     - Per-snapshot all-to-all (blocking/async)
     - Per-request all-to-all (async)

   * - **Single-Machine Latency**
     - ~100-500 µs (collective)
     - ~100-500 µs (collective)

   * - **Multi-Machine Latency**
     - ~10-50 ms (NCCL over TCP)
     - ~10-50 ms (NCCL over TCP)

   * - **Memory Cost**
     - Temporary recv buffer per snapshot
     - Partition storage (static, larger)

   * - **Flexibility**
     - Different routes per snapshot (time-varying)
     - Static partition map (time-invariant)

   * - **Optimal Use Case**
     - Discrete-time graphs, sparse sampling
     - Continuous-time graphs, large memory models

See Also
--------

- :doc:`data_layer` — RouteData schema and FeatureStore access
- :doc:`artifact_format` — Serialization of routes to disk
- :doc:`unified_pipeline` — How backends integrate routes into training
- Source: ``backends/dtdg/runtime/route.py``, ``backends/ctdg/runtime/route.py``
