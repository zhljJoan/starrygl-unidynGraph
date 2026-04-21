Artifact Format Reference
==========================

Artifacts are the serialized outputs of preprocessing. They contain partitioned
graph data, routing metadata, and runtime initialization files ready for training.

Artifacts are produced by preprocessors and consumed by runtime loaders.

Directory Structure
-------------------

After running preprocessing, the output directory contains:

.. code-block:: text

    output_dir/
    ├── meta/                          # Metadata (all modes)
    │   ├── config.yaml                # Original config (for reproducibility)
    │   ├── manifest.json              # Artifact version and schema
    │   └── statistics.json            # Dataset stats (node count, edge count, etc.)
    │
    ├── partitions/                    # Per-partition data (DTDG + Chunk)
    │   ├── partition_0.pth
    │   ├── partition_1.pth
    │   └── ...
    │
    ├── routes/                        # Routing metadata (DTDG)
    │   ├── routes_0.json
    │   ├── routes_1.json
    │   └── ...
    │
    ├── snapshots/                     # Snapshot graphs (DTDG)
    │   ├── snapshot_0.pt
    │   ├── snapshot_1.pt
    │   └── ...
    │
    ├── flare/                         # Flare-specific (DTDG only)
    │   ├── partition_book.pth
    │   └── route_plans.json
    │
    └── chunk/                         # Chunk-specific (future)
        ├── clusters/
        ├── chunks.index
        └── ...

Common Files
------------

**meta/manifest.json** — Artifact Schema Version

.. code-block:: json

    {
        "version": "0.1.0",
        "graph_mode": "ctdg",
        "task_type": "edge_predict",
        "preprocessing_date": "2026-04-22T12:34:56Z",
        "schema": {
            "partitions": "list[Dict]",
            "routes": "list[Dict]",
            "snapshots": "list[torch.Tensor]"
        },
        "num_partitions": 4,
        "num_snapshots": 100,
        "num_nodes": 10000,
        "num_edges": 50000,
        "node_feat_dim": 64,
        "edge_feat_dim": 16
    }

**meta/statistics.json** — Dataset Statistics

.. code-block:: json

    {
        "num_nodes": 10000,
        "num_edges": 50000,
        "num_temporal_events": 100000,
        "time_span": 86400.5,
        "avg_degree": 5.0,
        "max_degree": 1250,
        "node_feat_dim": 64,
        "edge_feat_dim": 16,
        "splits": {
            "train": {
                "num_events": 60000,
                "num_unique_nodes": 8000
            },
            "val": {
                "num_events": 20000,
                "num_unique_nodes": 6000
            },
            "test": {
                "num_events": 20000,
                "num_unique_nodes": 6500
            }
        }
    }

CTDG Artifacts
--------------

CTDG preprocessing produces:

.. code-block:: text

    ctdg_artifacts/
    ├── meta/
    │   ├── manifest.json
    │   └── statistics.json
    ├── partitions/
    │   ├── partition_0.pth        # CTDGPartition (events, node features)
    │   ├── partition_1.pth
    │   └── ...
    └── memory/
        ├── initial_memory_0.pth   # Initial memory bank (optional)
        └── ...

**partition_X.pth** — Event Partition

.. code-block:: python

    # Serialized CTDGPartition dataclass
    {
        "partition_id": 0,
        "events": {
            "timestamps": Tensor([E]),
            "sources": Tensor([E]),
            "destinations": Tensor([E]),
            "features": Tensor([E, F])
        },
        "node_features": Tensor([N, F]),
        "edge_features": Tensor([num_edges, F_edge]),
        "node_map": {1: 0, 5: 1, ...}  # global → local node ID
    }

DTDG/Flare Artifacts
---------------------

DTDG preprocessing produces (Flare backend):

.. code-block:: text

    dtdg_artifacts/
    ├── meta/
    │   ├── manifest.json
    │   └── statistics.json
    ├── partitions/
    │   ├── partition_0.pth        # PartitionData (all snapshots)
    │   ├── partition_1.pth
    │   └── ...
    ├── routes/
    │   ├── routes_0.json          # RouteData per partition
    │   ├── routes_1.json
    │   └── ...
    └── flare/
        ├── partition_book.pth     # DTDGPartitionBook
        └── route_plans.json       # SnapshotRoutePlan[]

**partition_X.pth** — Per-Partition All Snapshots

.. code-block:: python

    # Serialized PartitionData
    {
        "partition_id": 0,
        "node_map": {1: 0, 5: 1, ...},  # global → local
        "snapshots": [
            {
                # Snapshot 0
                "timestamp": 1000.0,
                "graph": dgl_graph,
                "ndata": {"x": Tensor([N, F])},
                "edata": {"x": Tensor([E, F_edge])},
                "node_ids": Tensor([...]),
            },
            {
                # Snapshot 1
                ...
            },
            ...
        ]
    }

**routes_X.json** — Routing Metadata for Partition X

.. code-block:: text

    {
        "partition_id": 0,
        "num_snapshots": 100,
        "snapshots": [
            {
                "snapshot_id": 0,
                "send_index": [0, 0, 1, 1],
                "send_count": [2, 2],
                "recv_index": [15, 42, 7],
                "recv_count": [3]
            }
        ]
    }

**flare/partition_book.pth** — Global Partition Mapping

.. code-block:: python

    # Serialized DTDGPartitionBook
    {
        "num_partitions": 4,
        "node_to_partition": Tensor([partition_id for node_id]),
        "partition_to_nodes": [
            Tensor([0, 1, 5, 10, ...]),  # Partition 0 nodes
            Tensor([2, 3, 7, 11, ...]),  # Partition 1 nodes
            ...
        ],
        "partition_gpus": {
            "0": 0,  # Partition 0 → GPU 0
            "1": 1,  # Partition 1 → GPU 1
            ...
        }
    }

**flare/route_plans.json** — Per-Snapshot Route Plans

.. code-block:: text

    [
        {
            "snapshot_id": 0,
            "timestamp": 1000.0,
            "routes": [
                {
                    "partition_id": 0,
                    "send_to": [1, 2],
                    "recv_from": [0, 2],
                    "num_send": 15,
                    "num_recv": 12
                }
            ]
        }
    ]

Chunk Artifacts (Future)
------------------------

Chunk preprocessing will produce:

.. code-block:: text

    chunk_artifacts/
    ├── meta/
    │   └── manifest.json
    ├── clusters/
    │   ├── cluster_0/
    │   │   ├── indices.pt         # Node IDs in cluster
    │   │   ├── features.pt        # Node features
    │   │   └── subgraph_csr.pt    # CSR adjacency
    │   └── ...
    ├── chunks.index               # Time-slice → cluster→chunk mapping
    └── schedules.json             # GPU allocation schedule

**chunks.index** — Chunk Metadata

.. code-block:: text

    {
        "num_chunks": 1000,
        "num_time_slices": 100,
        "num_clusters": 10,
        "chunks": [
            {
                "chunk_id": 0,
                "time_slice": 0,
                "cluster_id": 0,
                "start_time": 1000.0,
                "end_time": 1360.0,
                "num_nodes": 256,
                "num_edges": 512
            }
        ]
    }

Loading Artifacts
-----------------

**CTDG**: Load via CTDGSession

.. code-block:: python

    from starry_unigraph.backends.ctdg.runtime import CTDGSession

    session = CTDGSession(
        artifacts_path="ctdg_artifacts/",
        batch_size=32
    )

**DTDG**: Load via FlareRuntimeLoader

.. code-block:: python

    from starry_unigraph.backends.dtdg.runtime import FlareRuntimeLoader

    loader = FlareRuntimeLoader(
        artifacts_path="dtdg_artifacts/",
        num_gpus=4
    )
    backend = loader.build_graph_backend()

**Chunk**: Load via ChunkRuntimeLoader (future)

.. code-block:: python

    from starry_unigraph.runtime.chunk import ChunkRuntimeLoader

    loader = ChunkRuntimeLoader(
        artifacts_path="chunk_artifacts/",
        batch_size=64
    )

Serialization Notes
-------------------

- **PyTorch tensors**: Saved via ``torch.save()`` in ``.pth`` files
- **DGL graphs**: Embedded in ``.pth`` dicts (serializable via pickle)
- **Metadata**: JSON for human readability and compatibility
- **Compression**: Optional gzip on artifact files (set in config)

Version Compatibility
---------------------

Artifacts include a ``version`` field for backward compatibility:

.. code-block:: python

    if manifest["version"] != EXPECTED_VERSION:
        warn(f"Artifact version mismatch: {manifest['version']}")

See Also
--------

- :doc:`preprocess_layer` — How artifacts are created
- :doc:`data_layer` — Data structures serialized in artifacts
- :doc:`route_layer` — RouteData serialization in routes_X.json
