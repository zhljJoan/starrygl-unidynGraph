CTDG Training Guide
===================

The CTDG (Continuous-Time Dynamic Graph) path models temporal graphs as event streams,
where each edge carries a precise timestamp.  StarryUniGraph's CTDG backend is built on
the event-driven paradigm supporting memory-augmented temporal GNN models such as
TGN, JODIE.

Pipeline Overview
-----------------

.. code-block:: text

    Raw Data  (edge list + timestamps)
         │
         ▼
    CTDGPreprocessor               ← backends/ctdg/preprocess.py
         │  prepare_raw()
         │  build_partitions()
         │  build_runtime_artifacts()
         ▼
    PreparedArtifacts  (artifacts/)
         │
         ▼
    CTDGSession                    ← backends/ctdg/runtime/session.py
         │  build_runtime()
         │  iter_train / iter_eval / iter_predict
         │  train_step / eval_step / predict_step
         ▼
    Results  (loss, metrics, predictions)

Quick Start
-----------

Set ``data.graph_mode: ctdg`` (or use a ``tgn``-family model) in the config, then:

.. code-block:: bash

    # Single-process prepare for an 8-worker job
    WORLD_SIZE=8 /home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
        -m starry_unigraph --config configs/tgn_wiki.yaml \
        --artifact-root /shared/artifacts/WIKI --phase prepare

    # Copy /shared/artifacts/WIKI to every node if storage is not shared

    # Multi-node / multi-GPU train + validate on node 0
    /home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun \
        --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=node0 --master_port=29500 \
        -m starry_unigraph --config configs/tgn_wiki.yaml \
        --artifact-root /shared/artifacts/WIKI --phase train

    # Multi-node / multi-GPU train + validate on node 1
    /home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun \
        --nnodes=2 --node_rank=1 --nproc_per_node=4 \
        --master_addr=node0 --master_port=29500 \
        -m starry_unigraph --config configs/tgn_wiki.yaml \
        --artifact-root /shared/artifacts/WIKI --phase train

    # Multi-node / multi-GPU predict on test split on node 0
    /home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun \
        --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=node0 --master_port=29501 \
        -m starry_unigraph --config configs/tgn_wiki.yaml \
        --artifact-root /shared/artifacts/WIKI --phase predict

    # Then rerun the same predict command on node 1 with --node_rank=1

The ``prepare`` step is intentionally single-process, but it must still know the final
distributed partition count.  Set ``WORLD_SIZE`` to ``nnodes * nproc_per_node`` during
prepare so the generated artifacts match the later ``torchrun`` job.

In distributed CTDG mode, all ranks advance through the same global time windows.
Each rank only trains on the local edges assigned to it by the prepared partition
artifacts.  The current ownership rule is ``edge_owner = src_node_partition``.


Override hyperparameters without editing the config:

.. code-block:: bash

    /home/zlj/.miniconda3/envs/tgnn_3.10/bin/torchrun \
        --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=node0 --master_port=29500 \
        -m starry_unigraph --config configs/tgn_wiki.yaml \
        --artifact-root /shared/artifacts/WIKI --phase train \
        --epochs 50 --lr 0.0001 --batch-size 200

Config File
-----------

Default: ``configs/tgn_wiki.yaml``

.. code-block:: yaml

    model:
      name:    tgn
      family:  tgn
      task:    temporal_link_prediction
      hidden_dim:    100
      memory_dim:    100
      time_dim:      100
      embedding_dim: 100
      num_neighbors: 10

    data:
      name:       WIKI
      root:       [DATA_ROOT]
      format:     auto
      graph_mode: ctdg          # ← selects CTDG path

    train:
      epochs:        100
      batch_size:    200        # number of events per batch
      lr:            0.0001
      eval_interval: 1

    runtime:
      backend:    torch
      device:     cuda
      checkpoint: artifacts/ctdg/checkpoints/tgn_wiki.pt

    dist:
      world_size: 8             # target global workers for prepare/train/predict
      backend:    nccl

    ctdg:
      pipeline:          online
      mailbox_slots:     1
      historical_alpha:  0.5
      async_sync:        true

Key parameters:

- ``model.memory_dim`` — dimension of per-node memory vectors
- ``model.num_neighbors`` — temporal neighbors sampled per event
- ``train.batch_size`` — number of events per training batch
- ``data.graph_mode`` — must be ``ctdg``

Artifact Layout
---------------

.. code-block:: text

    <artifact_root>/
    ├── meta/
    │   └── artifacts.json      # graph_mode, num_nodes, num_edges, feature routes
    ├── partitions/
    │   ├── manifest.json
    │   ├── node_parts.pt       # node_id -> partition_id
    │   └── edge_parts.pt       # edge_id -> partition_id (owner by src partition)
    ├── routes/
    │   └── manifest.json
    └── sampling/
        ├── index.json
        └── boundaries.json     # global time-window batching summary

Core Class Reference
--------------------

**CTDGPreprocessor** — ``backends/ctdg/preprocess.py``

.. code-block:: python

    from starry_unigraph.backends.ctdg import CTDGPreprocessor
    preprocessor = CTDGPreprocessor()
    prepared = preprocessor.run(session_ctx)

**CTDGSession** — ``backends/ctdg/runtime/session.py``

.. code-block:: python

    from starry_unigraph.backends.ctdg import CTDGSession
    session = CTDGSession()
    session.build_runtime(session_ctx)

    for batch in session.iter_train(session_ctx):
        result = session.train_step(batch)   # {"loss": float, "num_events": int, ...}

    for batch in session.iter_eval(session_ctx, split="val"):
        result = session.eval_step(batch)

Using SchedulerSession
-----------------------

``SchedulerSession`` is the recommended API.  When ``graph_mode == "ctdg"`` it
creates a ``CTDGSession`` internally:

.. code-block:: python

    from starry_unigraph.session import SchedulerSession

    session = SchedulerSession.from_config("configs/tgn_wiki.yaml")
    session.prepare_data()      # runs CTDGPreprocessor
    session.build_runtime()     # creates CTDGSession
    result = session.run_task() # full training loop
    preds  = session.predict()  # test-split inference

Notes
-----
- ``prepare`` must complete before ``train``, using the same ``artifact_root``.
- For multi-node CTDG runs, ``prepare`` should use the same global partition count as
  the later distributed job.  Either set ``dist.world_size`` in config or export
  ``WORLD_SIZE`` before running ``--phase prepare``.
- Temporal neighbor sampling requires the native C++ extension
  ``libstarrygl_sampler.so``, built inside the ``tgnn_3.10`` conda environment.
  See :doc:`../guide/install_guide`.
