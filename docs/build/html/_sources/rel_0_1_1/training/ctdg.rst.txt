CTDG Training Guide
===================

The **Continuous-Time Dynamic Graph (CTDG)** path models the graph as a
continuous stream of timestamped interactions:

.. math::

   \mathcal{E} = \{(u_i, v_i, t_i, x_i)\}_{i=1}^{N}

Each interaction carries an exact timestamp. This mode is appropriate when
fine-grained event order matters, such as in temporal link prediction, fraud
detection, recommendation, and transaction modeling.

When To Use CTDG
----------------

CTDG is usually the right choice when:

- exact event ordering matters
- your model depends on memory, mailbox, or other time-aware state
- temporal dependencies are defined by interaction streams rather than snapshots

What StarryGL Provides for CTDG
-------------------------------

CTDG training is fundamentally event-driven. Every batch depends on the memory
state produced by earlier events, which makes distributed execution difficult
to implement correctly by hand.

StarryGL provides:

**Transparent Hotspot Memory Sharing**
   Frequently accessed nodes can dominate state synchronization cost. StarryGL
   includes built-in support for synchronizing hotspot memory more efficiently
   across ranks.

**Automated Event-Driven Runtime**
   The runtime handles chronological memory updates, temporal neighbor
   sampling, mailbox management, and cross-rank state consistency.

**Ready-to-Use Memory Models**
   Built-in model families such as ``tgn`` and ``jodie`` run directly on the
   CTDG runtime.

**Unified Session API**
   The same high-level :class:`SchedulerSession` lifecycle is used for prepare,
   train, validate, and predict.

End-to-End Workflow
-------------------

The CTDG path follows the same high-level workflow as DTDG:
**Prepare -> Train and Validate -> Predict**.

1. **Prepare**
   Run preprocessing once to build partition-aware artifacts and routing state.

2. **Train and Validate**
   Launch the distributed runtime with ``torchrun``. As in DTDG mode,
   ``--phase train`` runs both the training split and the validation split
   inside each epoch.

3. **Predict**
   Load the saved checkpoint and run inference on the test split.

Annotated End-to-End Example
----------------------------

The following example covers CTDG artifact generation, training, validation,
and inference.

.. code-block:: bash

   # 1. Prepare CTDG artifacts once.
   #    This computes partition-aware metadata using dist.world_size.
   python -m starry_unigraph \
       --config configs/tgn_wikitalk.yaml \
       --phase prepare

   # 2. Train on 4 GPUs.
   #    During --phase train, StarryGL executes:
   #      - session.run_epoch(split="train")
   #      - session.run_epoch(split="val")
   #    and saves the resulting checkpoint.
   torchrun --nproc_per_node=4 -m starry_unigraph \
       --config configs/tgn_wikitalk.yaml \
       --phase train

   # 3. Run test-time inference.
   #    This loads the checkpoint and calls session.predict(split="test").
   torchrun --nproc_per_node=4 -m starry_unigraph \
       --config configs/tgn_wikitalk.yaml \
       --phase predict

The same lifecycle in Python is:

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

How CTDG Models Are Formulated
------------------------------

CTDG models are usually organized around a three-stage memory pipeline for an
event between nodes :math:`u` and :math:`v` at time :math:`t`.

1. **Message Generation**

   .. math::

      m_u(t) = \text{Message}(s_u(t^-), s_v(t^-), t, x_e)

2. **Memory Update**

   .. math::

      s_u(t) = \text{RNN}(s_u(t^-), m_u(t))

3. **Temporal Embedding Computation**

   .. math::

      z_u(t) = \text{TemporalAttention}(s_u(t), \{s_k(t_k)\}_{k \in \mathcal{N}(u, t)})

StarryGL handles the hard parts around this formulation: mailbox buffering,
chronological state updates, temporal neighbor lookup, and distributed memory
synchronization.

Built-in CTDG Models
--------------------

Select the model family through ``model.family``.

**TGN (Temporal Graph Network)**
   Maintains per-node memory, updates it chronologically after events, and
   computes embeddings using temporal neighbor information.

**JODIE**
   Models the temporal co-evolution of interacting entities and is particularly
   suitable for user-item style event streams.

These model families share the same memory, mailbox, and temporal sampling
runtime in StarryGL.

Event-Driven Batching and Sampling
----------------------------------

Unlike DTDG, CTDG does not slide over snapshots. It processes a chronological
event stream, so batching and sampling parameters directly affect both accuracy
and systems behavior.

**``train.batch_size``**
   Number of chronological events processed in one forward-backward pass.

**``sampling.neighbor_limit``**
   Number of recent temporal interactions used when building temporal neighbor
   context.

**``ctdg.mailbox_slots``**
   Number of pending message slots retained for each node.

**``ctdg.async_sync``**
   Whether distributed memory synchronization is overlapped with later
   computation.

What the Config Means
---------------------

The default CTDG example is ``configs/tgn_wikitalk.yaml``:

.. code-block:: yaml

   model:
     name: tgn
     family: tgn
     task: temporal_link_prediction
     hidden_dim: 100

   data:
     name: WikiTalk
     graph_mode: ctdg

   train:
     epochs: 50
     batch_size: 3000
     lr: 0.0004

   ctdg:
     pipeline: online
     mailbox_slots: 1
     historical_alpha: 0.5
     async_sync: true

   sampling:
     neighbor_limit: [20]
     strategy: recent
     history: 1

   dist:
     world_size: 4

Key configuration takeaways:

- ``model.family`` selects the CTDG model family.
- ``train.batch_size`` controls the number of events per update.
- ``ctdg.pipeline: online`` selects the event-driven runtime.
- ``ctdg.mailbox_slots`` and ``ctdg.async_sync`` control memory behavior and
  synchronization overlap.
- ``sampling.neighbor_limit`` controls temporal neighbor depth.
- ``dist.world_size`` must match ``torchrun --nproc_per_node``.

Core Module Reference
---------------------

If you want to customize the CTDG pipeline, start with these modules.

1. **CTDGPreprocessor**

   Location: ``starry_unigraph/backends/ctdg/preprocess.py``

   Role: Builds CTDG artifacts, partition manifests, and routing metadata.

2. **CTDGMemoryBank**

   Location: ``starry_unigraph/backends/ctdg/runtime/memory.py``

   Role: Stores per-node memory, mailbox state, and pending distributed syncs.

3. **CTDG Runtime Factory and Runtime**

   Locations:
   ``starry_unigraph/backends/ctdg/runtime/factory.py`` and
   ``starry_unigraph/backends/ctdg/runtime/runtime.py``

   Role: Build the online runtime, wire together memory, sampling, model, and
   distributed execution, and run train, eval, and predict steps.

4. **SchedulerSession**

   Location: ``starry_unigraph/session.py``

   Role: Exposes the same unified high-level lifecycle as in DTDG mode.

How To Add a New CTDG Model
---------------------------

CTDG now uses an explicit model factory, similar to DTDG. The runtime builder
dispatches on ``model.family`` through ``build_ctdg_model(...)``. This means
new CTDG model families can be attached to the existing runtime without
rewiring the whole execution path.

1. Implement the new model in
   ``starry_unigraph/backends/ctdg/runtime/models.py`` or in a neighboring
   runtime module. The model should remain compatible with the current online
   runtime assumptions around memory state, temporal sampling, and task-level
   outputs.

2. Register the model in the CTDG factory in
   ``starry_unigraph/backends/ctdg/runtime/models.py`` by adding it to
   ``CTDG_MODEL_SPECS``. This is what makes ``build_ctdg_model(...)`` select
   your new implementation from ``model.family``.

3. Register the new family in
   ``starry_unigraph/registry/model_registry.py`` so that
   ``SchedulerSession`` can resolve it from the config.

4. Add the family to ``MODEL_GRAPH_MODES`` in
   ``starry_unigraph/config/schema.py`` if you want ``data.graph_mode`` to be
   inferred automatically as ``ctdg``.

5. Configure the new model:

   .. code-block:: yaml

      model:
        name: your_new_ctdg_model
        family: your_new_ctdg_model
        task: temporal_link_prediction
        hidden_dim: 128

      data:
        graph_mode: ctdg

      ctdg:
        pipeline: online

If the new model requires different construction arguments or a different
runtime wiring pattern, you may additionally need to extend
``build_ctdg_model(...)`` in
``starry_unigraph/backends/ctdg/runtime/models.py``.

If the new model changes the prediction contract or loss definition, you may
also need to update the relevant task adapter in
``starry_unigraph/registry/task_registry.py``.

Practical Performance Tips
--------------------------

**Start Small**
   Reduce ``train.batch_size`` first when debugging runtime stability.

**Watch Synchronization Cost**
   If throughput stalls, inspect hotspot behavior and asynchronous sync
   settings before scaling model size.

**Keep Configurations Consistent**
   Use the same ``dist.world_size`` assumptions in ``prepare`` and ``train``.

**Tune Sampling Before Scaling**
   ``sampling.neighbor_limit`` and mailbox-related settings often matter more
   than model width in early optimization.
