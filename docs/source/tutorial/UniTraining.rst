Unified Training Entry Point
=============================

StarryUniGraph exposes a **single command-line entry point** for all three graph modes.
The graph mode (``ctdg`` / ``dtdg`` / ``chunk``) is determined entirely by the config
file — there is no ``--mode`` flag.

.. code-block:: bash

    python -m starry_unigraph --config CONFIG [--phase PHASE] [OPTIONS]

The config file is the single source of truth.  Pass ``data.graph_mode`` explicitly,
or let the system infer it from ``model.family``.

Quick Reference
---------------

.. code-block:: bash

    # ── CTDG (single GPU, online event stream) ──────────────────────────────
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase all

    # ── DTDG / Flare (multi-GPU, snapshots) ─────────────────────────────────
    python -m starry_unigraph --config configs/mpnn_lstm_4gpu.yaml --phase prepare
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase train
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml --phase predict

    # ── Chunk (unified pipeline, framework ready) ───────────────────────────
    python -m starry_unigraph --config configs/chunk_default.yaml --phase prepare

Command-Line Options
--------------------

.. code-block:: text

    required:
      --config PATH            YAML config file path

    optional:
      --phase {prepare,train,predict,all}
                               Execution phase (default: all)
      --artifact-root PATH     Override artifact output root directory

    hyperparameter overrides (applied on top of the config file):
      --epochs N               Training epochs
      --lr FLOAT               Learning rate
      --batch-size N           Batch size
      --device DEVICE          Compute device (e.g. cuda:0)

Phases
------

``prepare``
  Single-process data preprocessing.  Reads raw data, writes artifact files to disk.
  Always run **without** ``torchrun``.

  .. code-block:: bash

      python -m starry_unigraph --config configs/mpnn_lstm_4gpu.yaml --phase prepare

``train``
  Distributed training + validation.  For multi-GPU modes use ``torchrun``.

  .. code-block:: bash

      torchrun --nproc_per_node=4 -m starry_unigraph \
          --config configs/mpnn_lstm_4gpu.yaml --phase train

``predict``
  Load the latest checkpoint and run inference on the test split.

  .. code-block:: bash

      torchrun --nproc_per_node=4 -m starry_unigraph \
          --config configs/mpnn_lstm_4gpu.yaml --phase predict

``all``
  Run ``prepare → train → predict`` sequentially.
  Suitable for single-GPU modes (CTDG).
  For multi-GPU modes, run the phases separately.

Config File Format
------------------

All three modes share the same YAML structure.  Mode-specific sections
(``ctdg:`` / ``dtdg:`` / ``chunk:``) are ignored when not applicable.

.. code-block:: yaml

    # ── identity ────────────────────────────────────────────────────────────
    model:
      name:    <model-name>        # e.g. tgn, mpnn_lstm
      family:  <model-family>      # e.g. tgn, mpnn_lstm
      task:    <task-name>         # e.g. temporal_link_prediction

    # ── data ────────────────────────────────────────────────────────────────
    data:
      root:       /path/to/raw/data
      name:       <dataset-name>
      format:     auto
      graph_mode: <ctdg|dtdg|chunk>   # required; or inferred from model.family
      split_ratio:
        train: 0.7
        val:   0.15
        test:  0.15

    # ── training ────────────────────────────────────────────────────────────
    train:
      epochs:        100
      batch_size:    200
      lr:            0.0001
      eval_interval: 1

    # ── runtime ─────────────────────────────────────────────────────────────
    runtime:
      backend:    torch
      device:     cuda
      checkpoint: /path/to/checkpoint.pt

    # ── distributed ─────────────────────────────────────────────────────────
    dist:
      backend:    nccl
      world_size: 1          # set to GPU count for DTDG/Chunk multi-GPU

    # ── mode-specific (only the section matching graph_mode is active) ───────
    ctdg:
      pipeline: online
      ...

    dtdg:
      pipeline: flare_native
      ...

    chunk:
      ...

Graph Mode Inference
--------------------

If ``data.graph_mode`` is absent, the system infers it from ``model.family``:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - ``model.family``
     - Inferred mode
     - Notes
   * - ``tgn``, ``dyrep``, ``jodie``, ``tgat``, ``apan``
     - ``ctdg``
     - Online event-stream models
   * - ``mpnn_lstm``, ``tgcn``, ``evolvegcn``
     - ``dtdg``
     - Snapshot-based models
   * - *(any other)*
     - error
     - Set ``data.graph_mode`` explicitly

Internal Dispatch
-----------------

The entry point routes all operations through ``SchedulerSession``:

.. code-block:: text

    python -m starry_unigraph  →  starry_unigraph/__main__.py
                                         │
                                         ▼
                                  SchedulerSession
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                    graph_mode=ctdg  graph_mode=dtdg  graph_mode=chunk
                          │              │              │
                    CTDGPreprocessor  FlareDTDG-     ChunkPreprocessor
                    CTDGSession       Preprocessor   ChunkRuntimeLoader
                                      FlareRuntime-
                                      Loader

``prepare_data()`` selects the preprocessor; ``build_runtime()`` selects the runtime
loader.  Both dispatch on ``data.graph_mode`` from ``PreparedArtifacts``.

Artifact Root Resolution
------------------------

The artifact root is resolved in this order:

1. ``--artifact-root PATH`` (CLI flag)
2. Parent directory of ``runtime.checkpoint`` in the config (two levels up)
3. ``<cwd>/artifacts/<data.name>``

All three phases must use the same artifact root.

Hyperparameter Overrides
-------------------------

CLI flags override the corresponding config keys without editing the file:

.. code-block:: bash

    # Quick debug run: 5 epochs, smaller batch
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase train \
        --epochs 5 --batch-size 64

    # Different device
    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase train \
        --device cuda:1

Mode-Specific Details
---------------------

.. toctree::
   :maxdepth: 1

   CTDGTraining
   DTDGTraining
   ChunkTraining
