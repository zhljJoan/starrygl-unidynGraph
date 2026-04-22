Unified Preprocessing Pipeline
==============================

The preprocessing layer transforms raw temporal graph data into partition-aware
artifacts that the runtime can load directly. All supported graph modes follow
the same high-level lifecycle:

1. load raw temporal data
2. partition or chunk that data according to the graph mode
3. emit serialized artifacts for later distributed execution

This shared structure is exposed through the preprocessing contracts in
``starry_unigraph.preprocess`` and coordinated by
``starry_unigraph.session.SchedulerSession``.

Overview
--------

The base protocol lives in ``starry_unigraph.preprocess.base`` and defines the
common responsibilities for preprocessing implementations.

The current codebase contains three main entry points:

- ``starry_unigraph.backends.ctdg.preprocess`` for continuous-time event
  streams
- ``starry_unigraph.backends.dtdg.preprocess`` for discrete-time snapshot
  preparation
- ``starry_unigraph.preprocess.chunk`` for the experimental chunk path

CTDG Preprocessing
------------------

For CTDG workloads, preprocessing keeps time continuous and prepares the event
stream for distributed chronological execution.

.. code-block:: text

    Raw events
        ↓
    event partitioning
        ↓
    routing / sampling metadata generation
        ↓
    serialized CTDG artifacts

The CTDG preprocessor is responsible for:

- loading timestamped interactions and optional node or edge features
- partitioning events for distributed workers
- generating metadata used by temporal neighbor sampling
- emitting partition artifacts for train, eval, and predict phases

DTDG Preprocessing
------------------

For DTDG workloads, preprocessing first organizes the temporal graph into
snapshots and then prepares partition-aware runtime payloads for sliding-window
training.

.. code-block:: text

    Raw events
        ↓
    snapshot construction
        ↓
    structural partitioning
        ↓
    route planning
        ↓
    serialized DTDG artifacts

The DTDG preprocessor is responsible for:

- bucketing events into ordered snapshots
- constructing per-snapshot graph payloads
- partitioning topology for distributed execution
- materializing route metadata consumed by the DTDG runtime loader

Chunk Preprocessing
-------------------

The chunk path is experimental. It compiles temporal workloads into smaller
spatio-temporal units that can later be scheduled with finer granularity.

.. code-block:: text

    Raw events or snapshots
        ↓
    temporal slicing
        ↓
    topology-local grouping
        ↓
    chunk dependency construction
        ↓
    serialized chunk artifacts

The chunk preprocessor is responsible for:

- defining chunk boundaries in time and topology
- recording chunk dependency metadata
- emitting chunk payloads and scheduling-oriented manifests

Dispatch in ``SchedulerSession``
--------------------------------

``SchedulerSession.prepare_data()`` is the library-level entry point that
selects the correct preprocessor based on ``data.graph_mode`` and the active
configuration.

At a high level, the dispatch looks like this:

.. code-block:: python

    if graph_mode == "ctdg":
        preprocessor = CTDGPreprocessor(...)
    elif graph_mode == "dtdg":
        preprocessor = DTDGPreprocessor(...)
    elif graph_mode == "chunk":
        preprocessor = ChunkPreprocessor(...)
    else:
        raise ValueError(f"Unsupported graph_mode: {graph_mode}")

    preprocessor.run(...)

This gives StarryGL a consistent user-facing lifecycle even though the actual
partitioning and serialization logic differs by backend.

Extending the Preprocessing Layer
---------------------------------

To add a new preprocessing path, implement the preprocessing contract, register
the new mode in the session-level dispatcher, and provide a runtime loader that
can consume the emitted artifacts.

In practice, that means defining:

- how raw temporal data is loaded
- how it is partitioned or chunked
- which manifests and payloads are written to disk
- how the corresponding runtime reconstructs execution objects later

See Also
--------

- :doc:`artifact_format` for the serialized outputs produced here
- :doc:`data_layer` for the in-memory objects used after loading artifacts
- :doc:`route_layer` for the communication metadata built during preprocessing
