Protocol Reference
==================

StarryGL currently exposes a small set of protocol-style interfaces used by the
unified pipeline and by shared runtime components. These interfaces define the
coordination boundaries between backend adapters, task logic, state handling,
and model execution.

Overview
--------

The main protocol definitions live in:

- ``starry_unigraph.runtime.backend`` for ``GraphBackend`` and
  ``StateManager``
- ``starry_unigraph.registry.task_adapter`` for ``TaskAdapter``
- ``starry_unigraph.models.base`` for ``TemporalModel``
- ``starry_unigraph.data.feature_store`` for ``FeatureStore``
- ``starry_unigraph.data.global_csr`` for ``GlobalCSR``

In the current codebase, these protocols are primarily composed by
``PipelineEngine`` in ``starry_unigraph.runtime.engine``.

Core Protocols
--------------

``GraphBackend`` — Chunk Provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GraphBackend`` abstracts how an execution backend yields work items to the
unified pipeline.

.. code-block:: python

    class GraphBackend(Protocol):
        def iter_batches(self, split: str, batch_size: int) -> Iterator[ChunkAtomic]:
            ...

        def reset(self) -> None:
            ...

        def describe(self) -> Dict[str, Any]:
            ...

Important detail: the current protocol yields ``ChunkAtomic`` objects, not
fully materialized ``BatchData``. Task-specific sampling and batch
materialization happen later inside ``PipelineEngine``.

Current adapters:

- ``CTDGGraphBackend`` in ``runtime/backend_adapters.py``
- ``FlareGraphBackend`` in ``runtime/backend_adapters.py``
- ``ChunkGraphBackend`` in ``runtime/backend_adapters.py``

These adapters wrap existing runtimes and convert their outputs into
``ChunkAtomic`` work units for the unified path.

``TaskAdapter`` — Task Logic + Sampling Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TaskAdapter`` encapsulates task-specific sampling intent, loss computation,
metric computation, and output formatting.

.. code-block:: python

    class TaskAdapter(Protocol):
        def build_sample_config(
            self,
            chunk: Any,
            model: Any,
            split: str,
        ) -> SampleConfig:
            ...

        def compute_loss(
            self,
            model_output: Dict[str, Tensor],
            batch: BatchData,
        ) -> Tensor:
            ...

        def compute_metrics(
            self,
            model_output: Dict[str, Tensor],
            batch: BatchData,
        ) -> Dict[str, float]:
            ...

        def format_output(
            self,
            model_output: Dict[str, Tensor],
            batch: BatchData,
        ) -> Dict[str, Any]:
            ...

The current protocol is defined in ``registry/task_adapter.py`` together with
``SampleConfig`` and companion task-facing batch definitions. The unified
engine imports the matching concrete batch container from ``data/batch_data.py``.

``StateManager`` — Iteration State Hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``StateManager`` handles state preparation before model execution and state
update after model execution.

.. code-block:: python

    class StateManager(Protocol):
        def prepare(
            self,
            node_ids: Tensor,
            timestamps: Optional[Tensor] = None,
        ) -> Dict[str, Any]:
            ...

        def update(
            self,
            model_output: Dict[str, Tensor],
            chunk: ChunkAtomic,
        ) -> None:
            ...

        def reset(self) -> None:
            ...

        def describe(self) -> Dict[str, Any]:
            ...

The default unified-pipeline integration currently uses
``DummyStateManager`` from ``runtime/backend_adapters.py``. Backend-specific
state systems continue to exist alongside this shared abstraction.

``TemporalModel`` — Embedding/State Core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model-side protocol is narrower than the task-facing model wrapper. It
describes the temporal backbone that consumes a message-flow graph plus state.

.. code-block:: python

    class TemporalModel(Protocol):
        def forward(
            self,
            mfg: Any,
            state: Dict[str, Tensor],
        ) -> Tensor:
            ...

        def compute_state_update(
            self,
            embeddings: Tensor,
            batch: BatchData,
        ) -> Dict[str, Tensor]:
            ...

In contrast, ``PipelineEngine`` currently calls ``model.predict(state, batch)``
on the concrete wrapped model object it is given. So the ``TemporalModel``
protocol should be read as the backbone-level contract, not as a literal
description of the engine's current top-level call site.

Shared Data Access Protocols
----------------------------

``FeatureStore`` — Feature Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FeatureStore`` lives in ``starry_unigraph.data.feature_store``:

.. code-block:: python

    class FeatureStore(Protocol):
        def get_node_feat(self, node_ids: Tensor) -> Tensor:
            ...

        def get_edge_feat(self, edge_ids: Tensor) -> Tensor:
            ...

        @property
        def node_feat_dim(self) -> int:
            ...

        @property
        def edge_feat_dim(self) -> int:
            ...

``GlobalCSR`` — Graph Structure Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GlobalCSR`` lives in ``starry_unigraph.data.global_csr``:

.. code-block:: python

    class GlobalCSR(Protocol):
        @property
        def rowptr(self) -> Tensor:
            ...

        @property
        def col(self) -> Tensor:
            ...

        def neighbors(self, node_id: int) -> Tensor:
            ...

        def subgraph(self, node_ids: Tensor) -> "GlobalCSR":
            ...

Current Status
--------------

The protocol layer is useful for understanding the current architecture. In
practice:

- ``GraphBackend`` / ``StateManager`` / ``TaskAdapter`` are real protocol
  surfaces used by ``PipelineEngine``.
- ``PipelineEngine`` is the shared orchestration entry for this interface set.
- The stable DTDG and CTDG training flows still rely heavily on their
  backend-specific runtime stacks.
- Some dataclasses exist in both ``data/`` and ``registry/task_adapter.py`` to
  bridge the task-adapter layer and the runtime-facing batch containers.

See Also
--------

- :doc:`unified_pipeline` — How these interfaces are composed today
- :doc:`data_layer` — Concrete containers referenced by the protocols
- Source: ``runtime/backend.py``, ``runtime/backend_adapters.py``,
  ``registry/task_adapter.py``, ``models/base.py``
