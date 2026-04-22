DTDG Training Guide
===================

The **Discrete-Time Dynamic Graph (DTDG)** path models a temporal graph as an
ordered sequence of snapshots:

.. math::

   \mathcal{G} = \{G_1, G_2, \dots, G_T\}

Each snapshot captures graph structure within a specific time bucket, such as
one hour, one day, or one week. This mode is appropriate when you care about
graph state over explicit time windows rather than the exact ordering of
individual events.

When To Use DTDG
----------------

DTDG is usually the right choice when:

- your data is already bucketed into snapshots
- your model reasons over graph states at discrete time steps
- preserving intra-snapshot topology matters more than exact event order

What StarryGL Provides for DTDG
-------------------------------

Building a distributed DTDG pipeline from scratch is difficult. Developers
usually have to solve graph partitioning, multi-GPU coordination, windowed
state propagation, and memory pressure at the same time.

StarryGL abstracts most of that systems complexity away. The DTDG stack
provides:

**Out-of-the-Box Distributed Execution**
   StarryGL partitions the graph, assigns chunks to ranks, and manages
   cross-device communication automatically.

**Ready-to-Use Temporal Models**
   Built-in models such as ``mpnn_lstm``, ``tgcn``, and ``evolvegcn`` are
   already aligned with the DTDG runtime.

**Unified Session API**
   :class:`SchedulerSession` provides a small lifecycle surface centered on
   ``prepare_data()``, ``build_runtime()``, ``run_epoch()``, and
   ``predict()``.

**Transparent Memory Optimization (Flare Backend)**
   The Flare backend keeps recent snapshots at high fidelity while compacting
   older history, making long temporal windows more practical on limited GPU
   memory.

End-to-End Workflow
-------------------

The DTDG path follows a consistent three-stage workflow:
**Prepare -> Train and Validate -> Predict**.

1. **Prepare**
   Run preprocessing once as a single process to generate partition-aware
   artifacts.

2. **Train and Validate**
   Launch distributed training with ``torchrun``. The unified CLI runs
   ``split="train"`` and ``split="val"`` inside each epoch, so validation is
   part of the normal training phase.

3. **Predict**
   Load the saved checkpoint and run inference on the test split.

Annotated End-to-End Example
----------------------------

The following example shows the complete DTDG flow with comments for artifact
generation, training, validation, and inference.

.. code-block:: bash

   # 1. Prepare artifacts once.
   #    This step partitions the snapshot graph according to dist.world_size.
   python -m starry_unigraph \
       --config configs/mpnn_lstm_4gpu.yaml \
       --phase prepare

   # 2. Train on 4 GPUs.
   #    During --phase train, StarryGL runs both training and validation:
   #      - session.run_epoch(split="train")
   #      - session.run_epoch(split="val")
   #    It also saves a checkpoint at the end.
   torchrun --nproc_per_node=4 -m starry_unigraph \
       --config configs/mpnn_lstm_4gpu.yaml \
       --phase train

   # 3. Run test-time inference.
   #    This loads the checkpoint and executes session.predict(split="test").
   torchrun --nproc_per_node=4 -m starry_unigraph \
       --config configs/mpnn_lstm_4gpu.yaml \
       --phase predict

If you want the same flow in Python, the high-level lifecycle is:

.. code-block:: python

   from starry_unigraph.session import SchedulerSession

   session = SchedulerSession(session_ctx=ctx, model_spec=..., task_adapter=...)

   # Offline artifact generation
   session.prepare_data()

   # Distributed runtime setup
   session.build_runtime()

   # One epoch of training
   train_result = session.run_epoch(split="train")

   # Validation on the same runtime
   val_result = session.run_epoch(split="val")

   # Test inference
   prediction = session.predict(split="test")

How DTDG Models Are Formulated
------------------------------

Most DTDG models follow a two-stage structure. StarryGL standardizes this
pattern so you can focus on the model itself rather than on distributed state
management.

The goal is to learn a time-aware node embedding :math:`h_v^t` for node
:math:`v` at time step :math:`t`.

1. **Spatial Aggregation**

   A GNN extracts structural information from the current snapshot:

   .. math::

      x_v^t = \text{GNN}(G_t, X_t)

2. **Temporal Update**

   A recurrent module integrates the current structural embedding with the
   hidden state from the previous time step:

   .. math::

      h_v^t = \text{RNN}(h_v^{t-1}, x_v^t)

In StarryGL, built-in models such as ``mpnn_lstm`` and ``tgcn`` follow this
exact formulation.

Built-in DTDG Models
--------------------

Select the model through ``model.name`` in the config.

**T-GCN (``tgcn``)**
   Combines graph convolution with a GRU-style temporal update. It is a good
   fit when topology and short-term temporal variation are tightly coupled.

**MPNN-LSTM (``mpnn_lstm``)**
   Combines message passing with stacked LSTM layers. This is the default model
   in the quickstart config and a strong general baseline.

**EvolveGCN (``evolvegcn``)**
   Evolves GCN parameters over time rather than keeping a fixed graph
   convolution kernel across snapshots.

Sliding Window Training
-----------------------

DTDG training captures temporal dependence by processing ordered windows of
snapshots instead of isolated frames or the full history at once.

The key parameters are:

**``model.window.size``**
   Controls the length of the snapshot window. Larger windows capture deeper
   temporal context but increase memory usage.

**``train.snaps``**
   Controls how many snapshots are traversed during an epoch.

**``dtdg.num_full_snaps``**
   Controls how many of the newest snapshots remain full-fidelity blocks in
   the Flare pipeline before older snapshots are compacted.

What the Config Means
---------------------

The default DTDG example is ``configs/mpnn_lstm_4gpu.yaml``:

.. code-block:: yaml

   model:
     name: mpnn_lstm
     family: mpnn_lstm
     task: snapshot_node_regression
     hidden_dim: 16
     window:
       size: 4

   data:
     name: rec-amazon-ratings
     graph_mode: dtdg

   train:
     epochs: 200
     batch_size: 32
     snaps: 200

   dtdg:
     pipeline: flare_native
     chunk_order: rand
     chunk_decay: half
     num_full_snaps: 2

   dist:
     world_size: 4

Key configuration takeaways:

- ``data.graph_mode: dtdg`` selects the DTDG backend.
- ``model.name`` selects the concrete temporal model.
- ``model.window.size`` controls how far the model looks into history.
- ``train.snaps`` controls epoch coverage over snapshots.
- ``dtdg.pipeline: flare_native`` enables the optimized Flare execution path.
- ``dist.world_size`` must match ``torchrun --nproc_per_node``.

Core Module Reference
---------------------

If you want to customize the pipeline or inspect the data flow, these are the
main modules to read.

1. **FlareDTDGPreprocessor**

   Location: ``starry_unigraph/backends/dtdg/preprocess.py``

   Role: Builds DTDG artifacts and partition-aware preprocessing outputs.

2. **FlareRuntimeLoader**

   Location: ``starry_unigraph/backends/dtdg/runtime/session_loader.py``

   Role: Loads the partition owned by a rank and exposes iterators for windowed
   training and single-frame evaluation or prediction.

3. **SchedulerSession**

   Location: ``starry_unigraph/session.py``

   Role: Connects preprocessing, runtime setup, epoch execution, checkpointing,
   and prediction.

How To Add a New DTDG Model
---------------------------

DTDG currently has a clean model factory, so adding a new model to the
existing chain is relatively direct.

1. Implement the model module in
   ``starry_unigraph/backends/dtdg/runtime/models.py``. The forward path should
   follow the same contract as the existing Flare models: accept either a
   single graph or an ``STGraphBlob`` window and return predictions plus the
   updated temporal state.

2. Register the model in the DTDG factory in
   ``starry_unigraph/backends/dtdg/runtime/models.py`` by adding it to
   ``MODEL_SPECS``. This is what makes ``build_flare_model(...)`` able to
   instantiate it during runtime setup.

3. Register the model name in
   ``starry_unigraph/registry/model_registry.py`` so that
   ``SchedulerSession`` can resolve the model spec from the config.

4. If the new model family should imply DTDG mode automatically, add it to
   ``MODEL_GRAPH_MODES`` in
   ``starry_unigraph/config/schema.py``.

5. Set the config accordingly:

   .. code-block:: yaml

      model:
        name: your_new_dtdg_model
        family: your_new_dtdg_model
        task: snapshot_node_regression
        hidden_dim: 128
        window:
          size: 4

      data:
        graph_mode: dtdg

If the new model also requires a new prediction target or loss contract, you
may additionally need to extend ``TaskRegistry`` in
``starry_unigraph/registry/task_registry.py``.

Practical Performance Tips
--------------------------

**Memory vs. Context**
   Increase ``model.window.size`` when you need longer context, but reduce
   ``train.batch_size`` first if memory becomes tight.

**Reduce Locality Bias**
   Keep ``dtdg.chunk_order: rand`` enabled so the model does not overfit to a
   fixed local chunk order.

**Configuration Consistency**
   Use the same ``dist.world_size`` in both ``prepare`` and ``train``.

**Async Transfers**
   ``FlareRuntimeLoader`` uses pinned memory and asynchronous CPU-to-GPU
   transfer so data loading is less likely to dominate iteration time.
