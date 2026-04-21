CTDG Training Guide
===================

The CTDG (Continuous-Time Dynamic Graph) path models temporal graphs as event streams,
where each edge carries a precise timestamp.  StarryUniGraph's CTDG backend is built on
an online, event-driven paradigm supporting memory-augmented temporal GNN models such as
TGN and MemShare.

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

    # Preprocess (run once)
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase prepare

    # Train + validate
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase train

    # Predict on test split
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase predict

    # Full pipeline in one command
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase all

Override hyperparameters without editing the config:

.. code-block:: bash

    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase train \
        --epochs 50 --lr 0.0001 --batch-size 200 --device cuda:0

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
      root:       /mnt/data/zlj/starrygl-data/raw
      format:     auto
      graph_mode: ctdg          # ← selects CTDG path

    train:
      epochs:        100
      batch_size:    200        # number of events per batch
      lr:            0.0001
      eval_interval: 1

    runtime:
      backend:    torch
      device:     cuda:0
      checkpoint: artifacts/ctdg/checkpoints/tgn_wiki.pt

    dist:
      world_size: 1             # CTDG is single-GPU
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
    │   └── manifest.json
    ├── routes/
    │   └── manifest.json
    └── sampling/
        └── ...                 # Temporal-CSR neighbor index

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

- CTDG runs in single-GPU / single-process mode.
  The experimental multi-GPU variant is ``train_tgn_dist.py``.
- ``prepare`` must complete before ``train``, using the same ``artifact_root``.
- Temporal neighbor sampling requires the native C++ extension
  ``libstarrygl_sampler.so``, built inside the ``tgnn_3.10`` conda environment.
  See :doc:`../guide/install_guide`.
