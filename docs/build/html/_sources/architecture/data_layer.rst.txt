Data Layer Reference
====================

The data layer defines unified data structures that represent graph data uniformly
across all three backends (CTDG, DTDG, Chunk). These structures form the contract
between preprocessors and runtime modules.

Overview
--------

The data layer consists of:

1. **Temporal Data Types** (raw, unpartitioned)
   - ``RawTemporalEvents`` — event stream (timestamp, src, dst, features)
   - Event loading and conversion utilities

2. **Partitioned Data Types** (after graph partitioning)
   - ``PartitionData`` — per-partition snapshot dataset
   - ``RouteData`` — routing metadata for feature exchange
   - ``TensorData`` — CSR-packed tensor lists for efficient storage

3. **Unified Batch Types** (used by training loop)
   - ``BatchData`` — unified batch container (all modes + tasks)
   - ``SampleConfig`` — task-specific sampling parameters

4. **Atomic/Chunked Units** (for chunk-based processing)
   - ``ChunkAtomic`` — time-slice × node-cluster atomic unit
   - ``ChunkBuilder`` — time + node partitioning pipeline

5. **Feature Access Protocols**
   - ``FeatureStore`` — node/edge feature shard storage
   - ``GlobalCSR`` — full-graph CSR adjacency access

Core Data Structures
--------------------

``BatchData`` — Unified Batch Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by all training loops (PipelineEngine) to represent a single batch.
Works identically for CTDG, DTDG, and Chunk modes.

.. code-block:: python

    @dataclass
    class BatchData:
        """Unified batch across all graph modes and tasks."""
        node_ids: Tensor          # [B] node IDs (for supervised tasks)
        edges: Tensor | None      # [2, E] edge pairs (for link prediction)
        labels: Tensor | None     # [B, K] or [B] labels
        timestamps: Tensor | None # [E] edge timestamps or [B] node times
        metadata: Dict[str, Any]  # Mode/task-specific fields

Usage in training:

.. code-block:: python

    batch = BatchData(
        node_ids=torch.tensor([1, 5, 12]),
        edges=torch.tensor([[1, 5], [2, 3]]),
        labels=torch.tensor([[1], [0], [1]]),
        timestamps=torch.tensor([100.5, 102.3]),
        metadata={"split": "train"}
    )
    # Pass to model forward(), task adapter computes loss/metrics

``SampleConfig`` — Task-Specific Sampling Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encapsulates how a task samples neighbors/time windows before materialization.

.. code-block:: python

    @dataclass
    class SampleConfig:
        """Task-specific sampling configuration."""
        task_type: str            # "edge_predict", "node_regress", "node_classify"
        num_hops: int | None      # Multi-hop neighborhoods
        num_neighbors: int | None # Neighbor sampling limit
        time_window: int | None   # Temporal lookback window (seconds)
        neg_sample_ratio: float   # Negative sampling ratio (edge tasks)

``TensorData`` — CSR-Packed Tensor Lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient storage of sparse graph data:

.. code-block:: python

    @dataclass
    class TensorData:
        """CSR-packed tensor list (efficient sparse representation)."""
        rowptr: Tensor     # [N+1] row pointers
        col: Tensor        # [E] column indices
        weights: Tensor    # [E] edge weights (optional)

Used for both node-to-neighbor CSR and time-series CSR (TCSR) in Chunk mode.

``PartitionData`` — Per-Partition Snapshot Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Represents all snapshots for a single partition (DTDG):

.. code-block:: python

    @dataclass
    class PartitionData:
        """All snapshots for one partition of the graph."""
        partition_id: int
        node_map: Dict[int, int]        # Global → local node ID
        snapshots: List[Dict[str, Tensor]] # Per-snapshot graph + features
        routes: RouteData                  # Routing for cross-partition edges

``RouteData`` — Routing Metadata for Feature Exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Describes how features are routed between partitions in distributed training:

.. code-block:: python

    @dataclass
    class RouteData:
        """Routing for distributed feature exchange."""
        send_index: Tensor | None  # [dst_count] local → global ranks
        recv_index: Tensor | None  # [src_count] which nodes to fetch from remote
        send_count: List[int]      # How many nodes sent to each rank
        recv_count: List[int]      # How many nodes received from each rank

Feature Access Protocols
------------------------

``FeatureStore`` — Node & Edge Feature Sharding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstraction for accessing node and edge features (sharded or centralized):

.. code-block:: python

    class FeatureStore(Protocol):
        """Node and edge feature storage (sharded or centralized)."""

        def get_node_feat(self, node_ids: Tensor) -> Tensor:
            """Fetch node features by ID. [N, F_node]"""
            ...

        def get_edge_feat(self, edge_ids: Tensor) -> Tensor:
            """Fetch edge features by ID. [E, F_edge]"""
            ...

        @property
        def node_feat_dim(self) -> int:
            """Feature dimension."""
            ...

        @property
        def edge_feat_dim(self) -> int:
            """Edge feature dimension."""
            ...

Usage:

.. code-block:: python

    node_feats = feature_store.get_node_feat(batch.node_ids)
    edge_feats = feature_store.get_edge_feat(batch.edges.flatten())

``GlobalCSR`` — Full-Graph CSR Adjacency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protocol for accessing graph structure (full adjacency or local partition):

.. code-block:: python

    class GlobalCSR(Protocol):
        """CSR-format graph adjacency (full or partitioned)."""

        @property
        def rowptr(self) -> Tensor:
            """Row pointers [N+1]."""
            ...

        @property
        def col(self) -> Tensor:
            """Column indices [E]."""
            ...

        def neighbors(self, node_id: int) -> Tensor:
            """Neighbors of node_id. [out-degree]"""
            ...

        def subgraph(self, node_ids: Tensor) -> "GlobalCSR":
            """Extract subgraph for node_ids."""
            ...

Usage (CTDG online):

.. code-block:: python

    csr = global_csr
    for src_id in batch.node_ids:
        neighbors = csr.neighbors(src_id)  # Sample K neighbors

Chunked Processing: ``ChunkAtomic`` & ``ChunkBuilder``
-------------------------------------------------------

For Chunk mode (time + node partitioning):

.. code-block:: python

    @dataclass
    class ChunkAtomic:
        """Atomic unit: time-slice × node-cluster.

        Three-level structure:
        - L1: Local TCSR (time-compressed CSR for one cluster)
        - L2: Cross-partition neighbors + edges to other clusters
        - L3: Scheduling metadata (which GPUs, sync dependencies)
        """
        chunk_id: int
        start_time: float
        end_time: float
        node_ids: Tensor
        local_tcsr: TensorData
        remote_edges: List[Tensor]
        metadata: Dict[str, Any]

Raw Temporal Data: Loading and Conversion
------------------------------------------

``RawTemporalEvents`` — Event Stream Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class RawTemporalEvents:
        """Immutable container for raw temporal events."""
        timestamps: Tensor    # [E]
        sources: Tensor       # [E]
        destinations: Tensor  # [E]
        features: Tensor      # [E, F]

Load from CSV:

.. code-block:: python

    from starry_unigraph.data import load_raw_temporal_events

    events = load_raw_temporal_events(
        csv_path="events.csv",
        time_col=0, src_col=1, dst_col=2, feat_col=3
    )
    # events.timestamps [E], events.sources [E], etc.

Snapshot Conversion:

.. code-block:: python

    from starry_unigraph.data import build_snapshot_dataset_from_events

    snapshots = build_snapshot_dataset_from_events(
        events,
        time_window=3600,  # 1-hour snapshots
        max_nodes=10000
    )
    # List[Dict[str, Tensor]] — one dict per snapshot

Data Flow Example
-----------------

Here's how data flows from raw → partitioned → training:

1. **Load Raw Events**:

   .. code-block:: python

       events = load_raw_temporal_events("data/events.csv")

2. **Preprocess & Partition** (via preprocessor):

   .. code-block:: python

       preprocessor = CTDGPreprocessor(config)
       artifacts = preprocessor.prepare_data(raw_events)
       # artifacts contains: PartitionData list, RouteData, feature stores

3. **Runtime Materializes Batches**:

   .. code-block:: python

       backend = CTDGGraphBackend(artifacts.partitions)
       for batch in backend.iter_batches(split="train", batch_size=32):
           # batch is BatchData
           forward_out = model(batch)

4. **Task Adapter Computes Loss**:

   .. code-block:: python

       adapter = task_registry.get("edge_predict")
       loss = adapter.compute_loss(forward_out, batch)
           metrics = adapter.compute_metrics(forward_out, batch)

See Also
--------

- :doc:`route_layer` — How BatchData is assembled from partitions
- :doc:`protocols` — Task and model protocols that consume BatchData
- :doc:`unified_pipeline` — PipelineEngine orchestrates data flow
- :doc:`artifact_format` — Serialized PartitionData and RouteData on disk
