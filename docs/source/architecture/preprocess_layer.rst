Unified Preprocessing Pipeline
==============================

The preprocessing layer transforms raw temporal graph data into partitioned
artifacts ready for training. All three backends (CTDG, DTDG, Chunk) use a
unified preprocessing protocol, but with completely independent implementations.

Overview
--------

All preprocessors follow the ``GraphPreprocessor`` protocol:

.. code-block:: python

    class GraphPreprocessor(Protocol):
        """Unified preprocessing interface."""

        def prepare_raw(self, config: Config) -> RawTemporalEvents:
            """Load raw data from disk or memory."""
            ...

        def build_partitions(self, raw: RawTemporalEvents) -> List[Any]:
            """Partition raw data by graph mode (SPEED, METIS, chunks, etc.)."""
            ...

        def build_runtime_artifacts(
            self, partitions: List[Any]
        ) -> PreparedArtifacts:
            """Emit serialized artifacts for runtime."""
            ...

Three implementations exist:

- ``CTDGPreprocessor`` (``backends/ctdg/preprocess.py``) — Online event processing
- ``FlareDTDGPreprocessor`` (``backends/dtdg/preprocess.py``) — Snapshot-based processing
- ``ChunkPreprocessor`` (``preprocess/chunk.py``) — Time + node partitioning

CTDG Preprocessing Pipeline
----------------------------

.. code-block:: text

    Raw Events (CSV)
         ↓
    [prepare_raw] → RawTemporalEvents (in-memory tensor buffers)
         ↓
    [build_partitions] → List[CTDGPartition]
         │ (node-based partitioning: round-robin or SPEED)
         │
    [build_runtime_artifacts] → PreparedArtifacts
         └─ meta/ (config, stats)
         └─ partitions/ (partition_0.pth, ...)
         └─ memory/ (optional initial memory)

**CTDG Partitioning Strategy**:

- Load events from CSV
- Compute **SPEED partitioning** (if configured) or use **round-robin**
- Emit per-partition event files (events_partition_X.pth)
- Emit node feature matrix (node_feats.pt)
- Emit edge feature matrix (edge_feats.pt)

Example config:

.. code-block:: yaml

    data:
        graph_mode: ctdg
        source: data/events.csv
        num_partitions: 4
        partition: speed  # or "round_robin"
        speed_topk_ratio: 0.1

Code example:

.. code-block:: python

    from starry_unigraph.backends.ctdg.preprocess import CTDGPreprocessor

    prep = CTDGPreprocessor(config)
    raw = prep.prepare_raw(config)
    partitions = prep.build_partitions(raw)
    artifacts = prep.build_runtime_artifacts(partitions)

DTDG/Flare Preprocessing Pipeline
----------------------------------

.. code-block:: text

    Raw Events (CSV)
         ↓
    [prepare_raw] → RawTemporalEvents
         ↓
    [build_snapshots] → List[Snapshot] (implicit, internal)
         │ (time-windowed snapshots with node/edge features)
         │
    [build_partitions] → List[PartitionData]
         │ (METIS or random graph partitioning per snapshot)
         │
    [build_flare_routing] → List[SnapshotRoutePlan]
         │ (per-snapshot all-to-all routing)
         │
    [build_runtime_artifacts] → PreparedArtifacts
         └─ meta/ (config, stats)
         ├─ partitions/ (partition_0.pth, partition_1.pth, ...)
         ├─ routes/ (routes_0.json, routes_1.json, ...)
         └─ flare/ (partition_book.pth, route_plans.json)

**DTDG Partitioning Strategy**:

1. Load events, convert to per-snapshot graphs (time windows)
2. Extract node and edge features
3. For each snapshot, compute graph partitioning (METIS or random)
4. Build per-partition PartitionData (contains all snapshots)
5. Generate RouteData for distributed training (cross-partition edges)

Example config:

.. code-block:: yaml

    data:
        graph_mode: dtdg
        source: data/events.csv
        time_window: 3600  # 1-hour snapshots
        num_partitions: 4
        partition_method: metis  # or "random"
        # Optional:
        max_nodes_per_partition: 5000

Code example:

.. code-block:: python

    from starry_unigraph.backends.dtdg.preprocess import FlareDTDGPreprocessor

    prep = FlareDTDGPreprocessor(config)
    raw = prep.prepare_raw(config)
    partitions = prep.build_partitions(raw)  # List[PartitionData]
    artifacts = prep.build_runtime_artifacts(partitions)

Chunk Preprocessing Pipeline (Future)
--------------------------------------

.. code-block:: text

    Raw Events (CSV)
         ↓
    [prepare_raw] → RawTemporalEvents
         ↓
    [build_time_slices] → List[TimeSlice] (implicit, internal)
         │ (time-windowed groups of events)
         │
    [build_node_clusters] → List[NodeCluster]
         │ (node clustering via METIS or community detection)
         │
    [build_chunks] → List[ChunkAtomic]
         │ (Cartesian product: time_slice × node_cluster)
         │
    [build_runtime_artifacts] → PreparedArtifacts
         └─ meta/ (config, stats)
         ├─ clusters/ (cluster features, adjacency)
         └─ chunks.index (chunk metadata)

**Chunk Partitioning Strategy**:

1. Load events
2. Time-partition into slices (e.g., 1-hour windows)
3. Node-partition into clusters (e.g., 256-node communities)
4. Create ChunkAtomic for each (time_slice, node_cluster) pair
5. Emit cluster features and chunk scheduling metadata

Shared Components
-----------------

All three preprocessors use these common utilities:

**load_raw_temporal_events()**

.. code-block:: python

    from starry_unigraph.data import load_raw_temporal_events

    events = load_raw_temporal_events(
        csv_path="data/events.csv",
        time_col=0,
        src_col=1,
        dst_col=2,
        feat_col=3,
        device="cuda:0"
    )

**build_snapshot_dataset_from_events()**

.. code-block:: python

    from starry_unigraph.data import build_snapshot_dataset_from_events

    snapshots = build_snapshot_dataset_from_events(
        events,
        time_window=3600,  # seconds
        max_nodes=None,
        device="cuda:0"
    )
    # Returns: List[DGLGraph] with timestamps

Extensibility: Adding a Custom Preprocessor
---------------------------------------------

To add a new graph mode (e.g., "streaming"):

1. Implement ``GraphPreprocessor`` protocol:

.. code-block:: python

    from starry_unigraph.preprocess.base import GraphPreprocessor

    class StreamingPreprocessor(GraphPreprocessor):
        """Streaming graph preprocessing."""

        def prepare_raw(self, config):
            # Load from streaming source
            return RawTemporalEvents(...)

        def build_partitions(self, raw):
            # Partition for streaming (no global partitions)
            return [StreamingPartition(...)]

        def build_runtime_artifacts(self, partitions):
            # Emit to disk
            return PreparedArtifacts(...)

2. Register in dispatcher:

.. code-block:: python

    # session.py
    if graph_mode == "streaming":
        preprocessor = StreamingPreprocessor(config)

3. Implement corresponding ``GraphBackend``:

.. code-block:: python

    from starry_unigraph.runtime.backend import GraphBackend

    class StreamingGraphBackend(GraphBackend):
        """Streaming data materialization."""

        def iter_batches(self, split, batch_size):
            for batch in self.source.iter_batches(split, batch_size):
                yield BatchData(...)

Comparison: CTDG vs DTDG vs Chunk
----------------------------------

.. list-table::
   :header-rows: 1

   * - Aspect
     - CTDG
     - DTDG
     - Chunk

   * - **Input**
     - Event stream
     - Event stream → snapshots
     - Event stream

   * - **Partition Type**
     - Node-based (round-robin/SPEED)
     - Node-based (METIS/random per snapshot)
     - Time + Node (Cartesian)

   * - **Output Unit**
     - Event sequence per partition
     - All snapshots for one partition
     - Chunk (time × node)

   * - **Materialization**
     - Online (event-by-event)
     - Snapshot batch
     - Chunk materialization

   * - **Multi-Machine**
     - ✗ (no partition boundaries)
     - ✓ (via all-to-all routes)
     - ✓ (via chunk scheduling)

   * - **State**
     - Memory banks (replica or partition)
     - RNN states per snapshot
     - RNN states per chunk

Preprocessing Flow in SchedulerSession
---------------------------------------

The unified entry point (SchedulerSession) dispatches to the right preprocessor:

.. code-block:: python

    # session.py
    def prepare_data(self, config):
        graph_mode = config.data.get('graph_mode') or infer_from_model(config)

        if graph_mode == "ctdg":
            preprocessor = CTDGPreprocessor(config)
        elif graph_mode == "dtdg":
            preprocessor = FlareDTDGPreprocessor(config)
        elif graph_mode == "chunk":
            preprocessor = ChunkPreprocessor(config)
        else:
            raise ValueError(f"Unknown graph_mode: {graph_mode}")

        # All follow same protocol
        raw = preprocessor.prepare_raw(config)
        partitions = preprocessor.build_partitions(raw)
        artifacts = preprocessor.build_runtime_artifacts(partitions)

        return artifacts

See Also
--------

- :doc:`data_layer` — Data structures used by preprocessors
- :doc:`route_layer` — Routing created during preprocessing
- :doc:`artifact_format` — Artifact files on disk
- Source: ``backends/ctdg/preprocess.py``, ``backends/dtdg/preprocess.py``, ``preprocess/chunk.py``
