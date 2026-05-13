Data Layer Reference
====================

The data layer defines the main data structures exchanged between preprocessors
and runtime modules. Some structures are shared across backends, while others
are backend-specific artifacts such as DTDG ``PartitionData`` and ``RouteData``.

Overview
--------

The data layer consists of:

1. **Temporal Data Types** (raw, unpartitioned)
   - ``RawTemporalEvents`` — event stream (src, dst, ts, weight, edge features)
   - Event loading and conversion utilities

2. **Partitioned Data Types** (after graph partitioning)
   - ``PartitionData`` — per-partition snapshot dataset
   - ``RouteData`` — routing metadata for feature exchange
   - ``TensorData`` — packed variable-length tensor lists

3. **Unified Batch Types** (used by training loop)
   - ``BatchData`` — batch structure used across unified-pipeline components
   - ``SampleConfig`` — task-specific sampling request for batch materialization

4. **Atomic/Chunked Units** (for chunk-based processing)
   - ``ChunkAtomic`` — time-slice × node-cluster atomic unit
   - ``ChunkBuilder`` — experimental time + node partitioning pipeline

5. **Feature Access Protocols**
   - ``FeatureStore`` — node/edge feature shard storage
   - ``GlobalCSR`` — full-graph CSR adjacency access

Core Data Structures
--------------------

``BatchData`` — Unified Batch Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by all training loops (PipelineEngine) to represent a single batch.
It is the intended shared batch container for the unified pipeline.

.. code-block:: python

    @dataclass
    class BatchData:
        """Unified batch across all graph modes and tasks."""
        mfg: Any
        node_ids: Tensor
        pos_src: Tensor | None = None
        pos_dst: Tensor | None = None
        neg_src: Tensor | None = None
        neg_dst: Tensor | None = None
        target_nodes: Tensor | None = None
        labels: Tensor | None = None
        timestamps: Tensor | None = None
        chunk_id: tuple | None = None
        local_node_mask: Tensor | None = None
        remote_manifest: Dict[str, Any] | None = None

Usage in training:

.. code-block:: python

    batch = BatchData(
        mfg=local_mfg,
        node_ids=torch.tensor([1, 5, 12]),
        pos_src=torch.tensor([1, 5]),
        pos_dst=torch.tensor([2, 3]),
        timestamps=torch.tensor([100.5, 102.3]),
        chunk_id=(0, 0),
    )
    # Pass to model.predict(), task adapter computes loss/metrics

``SampleConfig`` — Task-Specific Sampling Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encapsulates how a task samples neighbors/time windows before materialization.

.. code-block:: python

    @dataclass
    class SampleConfig:
        """Task-specific sampling configuration."""
        pos_src: Tensor | None = None
        pos_dst: Tensor | None = None
        neg_strategy: str = "none"
        neg_ratio: int = 1
        target_nodes: Tensor | None = None
        target_labels: Tensor | None = None
        num_neighbors: List[int] = field(default_factory=lambda: [20, 10])
        num_layers: int = 2
        sample_type: str = "temporal"
        extra: Dict[str, Any] = field(default_factory=dict)

``TensorData`` — Packed Tensor Lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient storage of per-snapshot variable-length tensors:

.. code-block:: python

    @dataclass
    class TensorData:
        """Packed variable-length tensor list."""
        ptr: List[int]     # [N+1] offsets into data
        data: Tensor       # concatenated tensor payload

Element ``i`` is stored as ``data[ptr[i]:ptr[i+1]]``. This is used by the
current DTDG partition artifacts to pack per-snapshot node IDs, edge IDs,
topology arrays, and feature tensors into a compact contiguous layout while
still supporting cheap slicing and reconstruction via ``item()`` or
``to_tensors()``.

``PartitionData`` — Per-Partition Snapshot Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Represents all snapshots for a single partition (DTDG):

.. code-block:: python

    @dataclass
    class PartitionData:
        """Packed DTDG partition artifact for one partition."""
        src_ids: TensorData                # remote source node IDs per snapshot
        dst_ids: TensorData                # local destination node IDs per snapshot
        edge_ids: TensorData               # global edge IDs per snapshot
        edge_src: TensorData               # source indices into [dst_ids, src_ids]
        edge_dst: TensorData               # destination indices into dst_ids
        node_data: Dict[str, TensorData]   # per-snapshot node features
        edge_data: Dict[str, TensorData]   # per-snapshot edge features
        routes: RouteData | None           # optional communication metadata

The implementation in
``starry_unigraph.data.partition.PartitionData`` stores one partition across
all snapshots in packed ``TensorData`` fields instead of a
``List[Dict[str, Tensor]]`` snapshot structure. It supports slicing by snapshot,
device transfer, pinning, serialization, and round-tripping to DGL blocks via
``to_blocks()`` and ``from_blocks()``.

``RouteData`` — Routing Metadata for Feature Exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Describes the execution-facing route metadata stored for distributed training:

.. code-block:: python

    @dataclass
    class RouteData:
        """Routing for distributed feature exchange."""
        send_sizes: List[List[int]]         # per-snapshot sends to each rank
        recv_sizes: List[List[int]]         # per-snapshot recvs from each rank
        send_index_ind: Tensor | None       # packed local indices gathered before send
        send_index_ptr: List[int] | None    # per-snapshot pointers into send_index_ind

This container matches the current DTDG partition artifacts. In the route-layer
terminology, it stores the execution-side route metadata used for distributed
feature exchange. It stores per-snapshot communication sizes together with a
packed send-index array used to gather local GNN outputs before ``all_to_all``.
There is no documented ``recv_index`` field in the current implementation.

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
    edge_feats = feature_store.get_edge_feat(edge_ids)

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
        """Atomic unit: time-slice × node-cluster."""
        chunk_id: Tuple[int, int]
        time_range: Tuple[float, float]
        node_set: Tensor
        tcsr_rowptr: Tensor
        tcsr_col: Tensor
        tcsr_ts: Tensor
        tcsr_edge_id: Tensor
        cross_node_ids: Tensor
        cross_node_home: Tensor
        cross_edge_count: Tensor

Raw Temporal Data: Loading and Conversion
------------------------------------------

``RawTemporalEvents`` — Event Stream Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class RawTemporalEvents:
        """Immutable container for raw temporal events."""
        src: Tensor
        dst: Tensor
        ts: Tensor
        weight: Tensor
        edge_feat: Tensor
        num_nodes: int
        num_edges: int
        source: str

Load from dataset root:

.. code-block:: python

    from starry_unigraph.data import load_raw_temporal_events

    events = load_raw_temporal_events(
        root="data",
        dataset_name="wikitalk",
        config=config,
    )
    # events.ts [E], events.src [E], events.dst [E]

Snapshot Conversion:

.. code-block:: python

    from starry_unigraph.data import build_snapshot_dataset_from_events

    snapshots = build_snapshot_dataset_from_events(
        events,
        snaps=24,
    )
    # dict containing per-snapshot graph payloads and metadata

Data Flow Example
-----------------

Here's how data flows from raw → partitioned → training:

1. **Load Raw Events**:

   .. code-block:: python

       events = load_raw_temporal_events(
           root="data",
           dataset_name="wikitalk",
           config=config,
       )

2. **Preprocess & Partition** (via preprocessor):

   .. code-block:: python

       session = SchedulerSession.from_config(config, dataset_path="data")
       artifacts = session.prepare_data()
       # artifacts is a PreparedArtifacts manifest with backend-specific payload dirs

3. **Runtime or Unified Pipeline Consumes Artifacts**:

   .. code-block:: python

       runtime = session.build_runtime()              # stable backend-native path
       engine = session.build_pipeline_engine(model)  # optional unified path

4. **Unified Pipeline Materializes Batches**:

   .. code-block:: python

       for chunk in engine.backend.iter_batches(split="train", batch_size=32):
           sample_config = engine.task_adapter.build_sample_config(
               chunk=chunk,
               model=engine.model,
               split="train",
           )
           batch = engine._materialize_batch(chunk, sample_config)

See Also
--------

- :doc:`route_layer` — How BatchData is assembled from partitions
- :doc:`protocols` — Task and model protocols that consume BatchData
- :doc:`unified_pipeline` — PipelineEngine orchestrates data flow
- :doc:`artifact_format` — Serialized PartitionData and RouteData on disk
