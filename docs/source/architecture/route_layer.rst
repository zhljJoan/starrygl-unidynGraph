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

**CTDG: Distributed Memory Routing with Partitioning**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CTDG supports both **single-machine multi-GPU** and **multi-machine distributed** training via node partitioning and synchronized memory banks:

.. code-block:: python

    class CTDGMemoryBank:
        """Per-rank partitioned node memory with distributed sync."""

        def __init__(self, num_nodes: int, node_partition_map: Tensor, rank: int, world_size: int):
            self.node_partition_map = node_partition_map  # node_id → rank/GPU
            self.rank = rank
            self.world_size = world_size
            # Local storage for nodes assigned to this rank
            self.num_local_nodes = (node_partition_map == rank).sum()
            self.local_memory = Memory[num_local_nodes, dim]  # GPU/CPU storage
            self.remote_memory = Memory[num_remote_nodes, dim]  # Buffer for remote nodes

        def fetch(self, node_ids: Tensor):
            """Get memory for nodes, syncing remote if needed."""
            local_mask = self.node_partition_map[node_ids] == self.rank
            result = zeros_like(node_ids)
            result[local_mask] = self.local_memory[global2local[local_mask]]
            result[~local_mask] = self._sync_remote(node_ids[~local_mask])
            return result

        def _sync_remote(self, remote_node_ids: Tensor):
            """Synchronize remote node features via all-to-all (multi-GPU) or RPC (multi-machine)."""
            if self.world_size == 1:
                return self.local_memory  # Single rank
            elif self.is_same_machine():
                # Single-machine multi-GPU: use all-to-all collective
                return self._exchange_via_nccl(remote_node_ids)
            else:
                # Multi-machine: use RPC for cross-rank access
                return self._exchange_via_rpc(remote_node_ids)

**Single-Machine Multi-GPU Setup**:

.. code-block:: python

    # During preprocessing:
    node_partition_map = torch.zeros(num_nodes)
    for partition_id, nodes in enumerate(partitions):  # nodes per GPU
        node_partition_map[nodes] = partition_id

    # During training (each rank):
    memory_bank = CTDGMemoryBank(num_nodes, node_partition_map, rank, world_size)
    # Each GPU synchronizes features via NCCL all-to-all

**Single-Machine Characteristics**:
- ✅ Multi-GPU support via NCCL all-to-all collective
- ✅ Low latency (same PCIe/NVLink fabric)
- ✅ All-to-all overhead manageable (typically 2-8 GPUs)
- ✅ Node partitioning (SPEED or round-robin) for load balancing

**Multi-Machine Distributed Setup**:

.. code-block:: python

    # Initialize distributed context with RPC
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    rpc.init_rpc(f"worker_{rank}", rank=rank, world_size=world_size)

    # During training (each rank):
    memory_bank = CTDGMemoryBank(num_nodes, node_partition_map, rank, world_size)
    # Remote node access via RPC (non-blocking, overlapped with compute)

**Multi-Machine Characteristics**:
- ✅ Supports arbitrary number of machines
- ✅ RPC enables asynchronous remote memory access
- ✅ Latency-tolerant (overlaps communication with computation)
- ⚠️ Network bandwidth is bottleneck (similar to message passing systems)
- ⚠️ Not suitable for fully-connected graphs (use DTDG for sparse sampling instead)

**Communication Methods Comparison**:

.. list-table:: CTDG Communication Patterns
   :header-rows: 1

   * - Setup
     - Communication
     - Latency
     - Suitable For

   * - Single-GPU
     - Local memory access
     - ~1-5 µs
     - Baseline

   * - Single-Machine Multi-GPU
     - NCCL all-to-all
     - ~100-500 µs
     - Dense neighborhoods (e.g., social networks)

   * - Multi-Machine
     - RPC with async batching
     - ~1-10 ms
     - Sparse neighborhoods or partial memory replication

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

.. list-table:: Routing Strategy Comparison
   :header-rows: 1

   * - Aspect
     - DTDG
     - CTDG

   * - **Communication**
     - Explicit all-to-all per snapshot
     - Implicit memory bank partition + sync

   * - **Scalability**
     - Multi-machine (snap shot-based boundaries)
     - Multi-machine (RPC) + Single-machine multi-GPU (all-to-all)

   * - **Memory Cost**
     - Temporary recv buffer (features fetched per snapshot)
     - Permanent partition (nodes stored on assigned ranks)

   * - **Latency (Local)**
     - All-to-all collective ~100-500 µs
     - Memory lookup ~1-10 µs

   * - **Latency (Multi-Machine)**
     - All-to-all ~10-50 ms (blocking)
     - RPC ~1-10 ms per node (async, overlappable)

   * - **Flexibility**
     - Different routes per snapshot
     - Static partition map (time-invariant)

   * - **Optimal Use Case**
     - Dense sampling, multi-machine, sparse edges
     - Large neighborhoods, temporal state updates, arbitrary topology

See Also
--------

- :doc:`data_layer` — RouteData schema and FeatureStore access
- :doc:`artifact_format` — Serialization of routes to disk
- :doc:`unified_pipeline` — How backends integrate routes into training
- Source: ``backends/dtdg/runtime/route.py``, ``backends/ctdg/runtime/route.py``
