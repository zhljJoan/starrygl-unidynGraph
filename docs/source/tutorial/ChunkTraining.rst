Chunk Training Guide
====================

Chunk training is StarryUniGraph's flexible time-windowed training pathway,
using **ChunkAtomic** as the underlying data unit and unified scheduling
through **PipelineEngine** for both DTDG and CTDG modes.

Overview
--------

The Chunk pathway provides:

- **Time-Windowed Processing**: Divide the temporal graph into fixed or adaptive time windows
- **Node Clustering**: Within each time window, partition nodes into clusters for load balancing
- **Unified Scheduling**: Single ``PipelineEngine`` orchestrates both DTDG and CTDG via the same protocol
- **State Management**: Pluggable StateManager for temporal state tracking

Comparison with Other Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - CTDG
     - DTDG
     - Chunk

   * - Data Unit
     - Event batch
     - Snapshot
     - ChunkAtomic (time slice × cluster)

   * - State Management
     - CTDGMemoryBank
     - RNNStateManager
     - StateManager Protocol

   * - Scheduler
     - CTDGOnlineRuntime
     - FlareRuntimeLoader
     - PipelineEngine (unified)

   * - Communication
     - On-demand fetch
     - all_to_all
     - CommEngine (unified primitives)

Quick Start
-----------

**Preprocessing (partition graph into ChunkAtomic pool):**

.. code-block:: bash

    python -m starry_unigraph --config configs/chunk_default.yaml --phase prepare

**Training with distributed launcher:**

.. code-block:: bash

    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/chunk_default.yaml \
        --phase train

**Configuration example:**

.. code-block:: yaml

    data:
      graph_mode: chunk
      source: data/my_events.csv
      num_time_slices: 100    # Partition time axis into 100 slices
      num_clusters: 8         # Cluster nodes per slice (load balancing)
      cluster_method: metis   # Node clustering algorithm

    training:
      batch_size: 32
      num_epochs: 100
      learning_rate: 0.0001

ChunkAtomic Data Structure
--------------------------

``ChunkAtomic`` is the minimal scheduling and data unit in Chunk mode,
representing one time slice × one node cluster:

.. code-block:: python

    from starry_unigraph.data.chunk_atomic import ChunkAtomic
    import torch

    chunk = ChunkAtomic(
        chunk_id=(0, 2),                       # (time_slice_id=0, cluster_id=2)
        time_range=(1000.0, 2000.0),           # Time range for this chunk
        node_set=torch.tensor([10, 20, 30]),   # Master nodes in this cluster
        tcsr_rowptr=torch.tensor([0, 2, 4, 5]),
        tcsr_col=torch.tensor([20, 30, 10, 30, 10]),
        tcsr_ts=torch.tensor([1100., 1200., 1300., 1400., 1500.]),
        tcsr_edge_id=torch.tensor([0, 1, 2, 3, 4]),
        cross_node_ids=torch.tensor([50, 60]),  # Neighbors from other clusters
        cross_node_home=torch.tensor([3, 3]),   # Owner cluster for each neighbor
        cross_edge_count=torch.zeros(8),
        load_estimate=5.0,
    )

    # Core interface: materialize chunk data for sampling/inference
    local_mfg, remote_manifest = chunk.materialize(sampler_config)

ChunkBuilder: Constructing Chunk Pool
--------------------------------------

.. code-block:: python

    from starry_unigraph.data.chunk_builder import ChunkBuilder, ChunkBuildConfig

    cfg = ChunkBuildConfig(
        num_time_slices=100,    # Partition time axis
        num_clusters=8,         # Clusters per time slice
        cluster_method="metis", # Clustering algorithm
        adaptive_split=False,   # Uniform time splits
    )

    builder = ChunkBuilder()
    pool = builder.build(cfg, raw_graph)          # Generate ChunkAtomic list
    assignment = builder.assign_devices(pool, 4)  # Assign to 4 ranks

    # assignment[rank_id] = chunks assigned to this rank
    # Uses greedy load balancing by load_estimate

PipelineEngine Execution
------------------------

.. code-block:: python

    from starry_unigraph.runtime.engine import PipelineEngine
    from starry_unigraph.runtime.backend_adapters import ChunkGraphBackend
    import torch

    # Create CUDA streams for overlapping computation and communication
    compute_stream = torch.cuda.Stream()
    comm_stream    = torch.cuda.Stream()

    engine = PipelineEngine(
        compute_stream=compute_stream,
        comm_stream=comm_stream,
        overlap=True,  # Enable double-buffering (overlap communication and compute)
    )

    backend = ChunkGraphBackend(chunk_pool, rank, world_size)
    state_manager = ChunkStateManager(num_nodes=10000, feat_dim=64)

    # Training loop
    for epoch in range(num_epochs):
        train_losses = engine.run_epoch("train", batch_size=32)
        val_losses = engine.run_epoch("val", batch_size=128)
        print(f"Epoch {epoch}: train_loss={sum(train_losses)/len(train_losses):.4f}")

Custom Runner Implementation
-----------------------------

Implement a custom Runner by inheriting and implementing 5 core methods:

.. code-block:: python

    from starry_unigraph.runtime.backend import GraphBackend
    from starry_unigraph.data.chunk_atomic import ChunkAtomic
    from starry_unigraph.data.batch_data import BatchData
    import torch

    class ChunkGraphBackend(GraphBackend):
        """Custom chunk-based data backend."""

        def __init__(self, chunk_pool, rank, world_size):
            self.chunk_pool = chunk_pool
            self.rank = rank
            self.world_size = world_size

        def iter_batches(self, split, batch_size):
            """Iterate over chunks in order, yielding materialized batches."""
            for chunk in self.chunk_pool:
                local_mfg, remote_manifest = chunk.materialize(self.sampler_config)
                batch = BatchData(
                    node_ids=local_mfg.node_ids,
                    edges=local_mfg.edges,
                    timestamps=chunk.time_range,
                    metadata={"chunk_id": chunk.chunk_id}
                )
                yield batch

        def reset(self):
            """Reset chunk iterator state."""
            pass

        def describe(self):
            """Describe backend configuration."""
            return {
                "backend": "chunk",
                "num_chunks": len(self.chunk_pool),
                "num_time_slices": len(set(c.chunk_id[0] for c in self.chunk_pool)),
            }

Custom StateManager Implementation
-----------------------------------

.. code-block:: python

    from starry_unigraph.runtime.backend import StateManager
    import torch

    class ChunkStateManager(StateManager):
        """Custom state manager with temporal caching."""

        def __init__(self, num_nodes, feat_dim, world_size=1, device="cuda"):
            self.num_nodes = num_nodes
            self.feat_dim = feat_dim
            self.world_size = world_size
            self.device = device
            self.state = torch.zeros(num_nodes, feat_dim, device=device)

        def prepare(self, node_ids, timestamps):
            """Fetch state for required nodes (synchronous or async)."""
            return self.state[node_ids]

        def update(self, output, chunk):
            """Update node states after forward pass."""
            node_ids = output.node_ids
            new_state = output.embeddings
            self.state[node_ids] = new_state

        def reset(self):
            """Reset state manager."""
            self.state.zero_()

        def describe(self):
            """Describe state manager configuration."""
            return {
                "num_nodes": self.num_nodes,
                "feat_dim": self.feat_dim,
                "world_size": self.world_size,
            }

Caching Systems
---------------

Chunk mode supports three-tier caching managed by StateManager:

.. code-block:: python

    from starry_unigraph.runtime.cache.hot_cache import HotCache
    from starry_unigraph.runtime.cache.decay_cache import DecayCache

    # Hot cache: recently accessed node states
    hot_cache = HotCache(capacity=10000, feat_dim=64, device="cuda:0")

    # Decay cache: frequently accessed with time-decay eviction
    decay_cache = DecayCache(
        capacity=50000, feat_dim=64, device="cuda:0",
        decay_factor=0.95,
    )

    # Usage
    hit_mask, cached_vals = hot_cache.lookup(node_ids)
    hot_cache.update(node_ids, new_vals)
    evicted = hot_cache.evict(threshold=5)

Communication Patterns
----------------------

Chunk mode provides flexible communication via the **CommEngine** protocol:

.. code-block:: python

    from starry_unigraph.runtime.comm.comm_engine import CommEngine

    class ChunkCommEngine(CommEngine):
        """Communication primitive for cross-cluster feature exchange."""

        def async_exchange(self, send_ids, send_vals, plan):
            """All-to-all or point-to-point exchange."""
            return self.all_to_all_single(send_ids, send_vals, plan)

        def async_fetch(self, node_ids, owners):
            """Asynchronous fetch with caching."""
            pass

        def try_recv_delta(self):
            """Non-blocking query for received state updates."""
            pass

        def all_reduce_gradients(self):
            """Synchronize gradients across all ranks (DDP)."""
            pass

See Also
--------

- :doc:`../architecture/data_layer` — BatchData and ChunkAtomic structures
- :doc:`../architecture/unified_pipeline` — PipelineEngine design
- :doc:`../architecture/protocols` — StateManager and GraphBackend protocols
- :doc:`UniTraining` — Unified training entry point
