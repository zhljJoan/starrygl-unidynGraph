DTDG Training Guide (Flare)
===========================

The DTDG (Discrete-Time Dynamic Graph) path models temporal graphs as snapshot
sequences.  StarryUniGraph's DTDG backend uses the **Flare** architecture for
high-throughput multi-GPU data-parallel training, with built-in support for
MPNN-LSTM and other snapshot-aware GNN models.

Pipeline Overview
-----------------

.. code-block:: text

    Raw Data  (edge list + time-bucket labels)
         │
         ▼
    FlareDTDGPreprocessor          ← backends/dtdg/preprocess.py
         │  prepare_raw()
         │  build_partitions()     → N PartitionData files (part_000.pth …)
         │  build_runtime_artifacts()
         ▼
    PreparedArtifacts  (artifacts/)
         │
         ▼
    FlareRuntimeLoader             ← backends/dtdg/runtime/session_loader.py
         │  iter_train()  → STGraphBlob  (multi-frame training batch)
         │  iter_eval()   → DTDGBatch    (single-frame evaluation batch)
         │  run_train_step / run_eval_step / run_predict_step
         ▼
    Results  (loss, metrics, predictions)

Quick Start
-----------

Set ``data.graph_mode: dtdg`` (or use an ``mpnn_lstm``-family model) in the config:

.. code-block:: bash

    # Step 1 — preprocess (single process, run once)
    python -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase prepare

    # Step 2 — 4-GPU distributed training
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase train

    # Step 3 — 4-GPU prediction
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase predict

Override hyperparameters:

.. code-block:: bash

    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase train \
        --epochs 100 --lr 0.001 --batch-size 32

Single-GPU debugging (set ``dist.world_size: 1`` in the config):

.. code-block:: bash

    python -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase all

Config File
-----------

Default: ``configs/mpnn_lstm_4gpu.yaml``

.. code-block:: yaml

    model:
      name:       mpnn_lstm
      family:     mpnn_lstm
      task:       snapshot_node_regression
      hidden_dim: 16
      window:
        size: 4             # LSTM history window (number of snapshots)

    data:
      name:       rec-amazon-ratings
      root:       /mnt/data/zlj/starrygl-data/raw
      format:     auto
      graph_mode: dtdg      # ← selects DTDG / Flare path

    train:
      snaps:         200    # number of snapshots used for training
      epochs:        200
      batch_size:    32
      lr:            0.001
      eval_interval: 1

    runtime:
      backend:    torch
      device:     cuda
      checkpoint: /mnt/data/zlj/starrygl-artifacts/rec-amazon-ratings/mpnn_lstm_4gpu.pt

    dist:
      backend:    nccl
      world_size: 4         # must equal --nproc_per_node

    dtdg:
      pipeline:    flare_native   # enables Flare partition communication
      chunk_order: rand           # cluster ordering within a snapshot
      chunk_decay: half

Key parameters:

- ``model.window.size`` — LSTM history depth; larger → more GPU memory
- ``train.snaps`` — reduce for faster debugging (e.g. ``20``)
- ``dtdg.pipeline`` — must be ``flare_native``
- ``dist.world_size`` — must match ``--nproc_per_node``; ``prepare`` and ``train``
  must use the same value

Artifact Layout
---------------

.. code-block:: text

    <artifact_root>/
    ├── meta/
    │   └── artifacts.json      # graph_mode, num_parts, snapshot_count, pipeline
    ├── partitions/
    │   └── manifest.json
    ├── routes/
    │   └── manifest.json
    └── flare/
        ├── manifest.json
        ├── part_000.pth        # PartitionData for rank 0
        ├── part_001.pth
        ├── part_002.pth
        └── part_003.pth

Core Class Reference
--------------------

**FlareDTDGPreprocessor** — ``backends/dtdg/preprocess.py``

.. code-block:: python

    from starry_unigraph.backends.dtdg import FlareDTDGPreprocessor
    preprocessor = FlareDTDGPreprocessor()
    prepared = preprocessor.run(session_ctx)   # writes part_000.pth … part_N.pth

**FlareRuntimeLoader** — ``backends/dtdg/runtime/session_loader.py``

.. code-block:: python

    from starry_unigraph.backends.dtdg import FlareRuntimeLoader

    loader = FlareRuntimeLoader.from_partition_data(
        data=partition_data, device=device,
        rank=rank, world_size=world_size, config=config,
    )

    for blob in loader.iter_train(split="train"):       # → STGraphBlob
        result = loader.run_train_step(runtime, blob)

    for batch in loader.iter_eval(split="val"):         # → DTDGBatch
        result = loader.run_eval_step(runtime, batch)

Using SchedulerSession
-----------------------

.. code-block:: python

    from starry_unigraph.session import SchedulerSession
    from starry_unigraph.types import SessionContext
    from pathlib import Path

    ctx = SessionContext(
        config=config,
        project_root=Path("."),
        artifact_root=Path("/mnt/data/zlj/starrygl-artifacts/rec-amazon-ratings"),
        dist=dist_ctx,
        ...
    )
    session = SchedulerSession(session_ctx=ctx, model_spec=..., task_adapter=...)

    # single process
    session.prepare_data()          # runs FlareDTDGPreprocessor

    # multi-process (launched by torchrun)
    session.build_runtime()         # creates FlareRuntimeLoader + DDP model
    for epoch in range(epochs):
        session.run_epoch("train")
        session.run_epoch("val")

Multi-GPU Notes
---------------

- Run ``prepare`` as a **single process** (no ``torchrun``).
  It reads ``dist.world_size`` from the config and generates that many partition files.
- ``train`` and ``predict`` must be launched with
  ``torchrun --nproc_per_node=<world_size>``.
- Each rank automatically loads ``part_<rank>.pth``; no manual sharding is needed.
- DDP gradient synchronization is handled internally by ``init_flare_training()``.

Performance Tips
----------------

- Increase ``model.window.size`` to capture longer temporal dependencies (uses more
  GPU memory).
- Set ``dtdg.chunk_order: rand`` to randomize cluster ordering and reduce locality bias.
- Reduce ``train.snaps`` or ``train.batch_size`` if running out of memory.
- ``STGraphLoader`` uses ``pin_memory`` and a dedicated ``torch.cuda.Stream`` for
  asynchronous CPU-to-GPU transfer; this is enabled automatically.
