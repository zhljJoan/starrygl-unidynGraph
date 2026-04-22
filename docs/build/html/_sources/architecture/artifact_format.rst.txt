Artifact Format Reference
=========================

Artifacts are the serialized outputs of preprocessing. They contain the
partitioned graph data, routing metadata, and runtime manifests required to
launch distributed training, evaluation, and prediction.

The exact files differ by graph mode, but all artifact directories follow the
same idea: preprocessing produces a stable on-disk boundary, and runtime
loaders reconstruct backend-specific execution objects from it.

Common Layout
-------------

A prepared artifact root usually contains a small amount of shared metadata
plus one or more backend-specific subdirectories.

.. code-block:: text

    artifact_root/
    ├── meta/
    │   ├── config.yaml
    │   ├── manifest.json
    │   └── statistics.json
    ├── partitions/
    ├── routes/
    ├── sampling/
    ├── flare/
    └── chunk/

Not every directory is present for every mode. For example, DTDG artifacts
commonly populate ``flare/``, while CTDG artifacts more often rely on
``partitions/``, ``routes/``, and sampling-related metadata.

Common Metadata Files
---------------------

``meta/config.yaml``
   A copy of the resolved runtime configuration used during preprocessing.

``meta/manifest.json``
   A lightweight schema declaration describing the artifact root, graph mode,
   partition count, and available payload groups.

``meta/statistics.json``
   Dataset-level summary statistics such as node count, edge count, split
   sizes, or snapshot/event ranges.

A simplified manifest typically looks like this:

.. code-block:: json

    {
        "version": "0.1.1",
        "graph_mode": "ctdg",
        "num_partitions": 4,
        "artifacts": ["meta", "partitions", "routes", "sampling"]
    }

CTDG Artifacts
--------------

Continuous-time preprocessing emits partition-aware event data plus the runtime
metadata required for temporal neighbor sampling and memory synchronization.

Typical layout:

.. code-block:: text

    ctdg_artifacts/
    ├── meta/
    │   ├── config.yaml
    │   ├── manifest.json
    │   └── statistics.json
    ├── partitions/
    │   ├── part_000.pth
    │   ├── part_001.pth
    │   └── ...
    ├── routes/
    │   ├── manifest.json
    │   └── ...
    └── sampling/
        ├── index.json
        └── ...

The partition payload usually stores:

- chronologically ordered local events
- node and edge feature tensors needed by the local rank
- node-partition maps or routing indices for remote lookups
- split metadata for training, evaluation, and inference

DTDG Artifacts
--------------

Discrete-time preprocessing emits snapshot-aware partitions and the metadata
needed by the DTDG runtime loader to reconstruct sliding-window execution.

Typical layout:

.. code-block:: text

    dtdg_artifacts/
    ├── meta/
    │   ├── config.yaml
    │   ├── manifest.json
    │   └── statistics.json
    ├── flare/
    │   ├── manifest.json
    │   ├── part_000.pth
    │   ├── part_001.pth
    │   └── ...
    └── routes/
        ├── manifest.json
        └── ...

The DTDG partition payload usually stores:

- per-partition snapshot tensors or graph objects
- structural partition metadata
- cross-partition routing plans for each time step or chunk group
- auxiliary state needed to resume sliding-window training

Chunk Artifacts
---------------

The chunk path is still experimental, but the artifact contract follows the
same pattern: preprocessing compiles the workload into chunk-indexed payloads
plus scheduling metadata.

Typical layout:

.. code-block:: text

    chunk_artifacts/
    ├── meta/
    │   ├── manifest.json
    │   └── statistics.json
    ├── chunk/
    │   ├── manifest.json
    │   ├── chunks.index
    │   └── ...
    └── routes/
        └── ...

These artifacts generally describe:

- time-bounded chunk units
- topology-local chunk payloads
- chunk-to-chunk dependency metadata
- scheduling hints for communication and execution overlap

Loading Artifacts
-----------------

Different runtimes consume different parts of the artifact tree:

- **CTDG** uses the CTDG runtime session and loader stack under
  ``starry_unigraph.backends.ctdg.runtime``.
- **DTDG** uses the runtime loaders under
  ``starry_unigraph.backends.dtdg.runtime``.
- **Chunk** uses the experimental chunk preprocessing and runtime path when
  enabled.

In practice, most users do not construct these runtime objects manually.
Instead, :class:`starry_unigraph.session.SchedulerSession` reads the config,
locates the prepared artifacts, and dispatches to the correct backend.

Serialization Notes
-------------------

- Tensor-heavy payloads are typically stored with ``torch.save()`` in
  ``.pth`` files.
- Human-readable metadata and manifests are stored as JSON or YAML.
- Backend-specific graph objects may embed DGL or PyTorch-native structures.
- Compatibility is managed at the manifest level rather than by promising a
  single immutable file tree for every backend.

See Also
--------

- :doc:`preprocess_layer` for how artifacts are produced
- :doc:`data_layer` for the in-memory structures reconstructed from artifacts
- :doc:`route_layer` for the routing metadata consumed during distributed
  execution
