Protocol Reference
==================

Protocols are abstract interfaces that enable composable training loops.
They decouple task logic (edge prediction, node regression) from graph modes
(CTDG, DTDG, Chunk) and model architectures.

All protocols use Python's ``typing.Protocol`` for structural subtyping (duck typing).

Overview
--------

.. code-block:: text

    Training Loop (PipelineEngine)
           │
           ├─ GraphBackend    [iter_batches] ──→ Batch data
           ├─ TaskAdapter     [compute_loss, metrics]
           ├─ StateManager    [prepare, update state]
           ├─ TemporalModel   [forward, compute_state_update]
           └─ Task Head       [score predictions]

Each protocol is independent and can be swapped:

- Implement ``GraphBackend`` for a new graph mode
- Implement ``TaskAdapter`` for a new task type
- Implement ``TemporalModel`` for a new backbone

Core Protocols
--------------

``GraphBackend`` — Batch Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstracts data iteration across graph modes.

.. code-block:: python

    class GraphBackend(Protocol):
        """Unified interface for batching across all graph modes."""

        def iter_batches(self, split: str, batch_size: int) -> Iterator[BatchData]:
            """Yield batches for a split (train/val/test).

            Args:
                split: One of "train", "val", "test"
                batch_size: Batch size

            Yields:
                BatchData objects with node_ids, edges, labels, etc.
            """
            ...

        def reset(self) -> None:
            """Reset internal state (e.g., current batch pointer)."""
            ...

        def describe(self) -> str:
            """Human-readable description (mode, dataset, partitions)."""
            ...

Implementations:

- ``CTDGGraphBackend`` — Wraps CTDGSession, yields event batches
- ``FlareGraphBackend`` — Wraps FlareRuntimeLoader, yields snapshot batches
- ``ChunkGraphBackend`` — Wraps ChunkRuntimeLoader, yields chunk batches

``TaskAdapter`` — Task-Specific Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encapsulates task-specific computations (loss, metrics, sampling).

.. code-block:: python

    class TaskAdapter(Protocol):
        """Task-specific logic (loss, metrics, output formatting)."""

        def build_sample_config(self) -> SampleConfig:
            """Create sampling config for this task.

            Returns:
                SampleConfig with num_neighbors, time_window, etc.
            """
            ...

        def compute_loss(
            self, model_output: Tensor, batch: BatchData
        ) -> Tensor:
            """Compute loss for this task.

            Args:
                model_output: Predictions from model
                batch: Input batch with labels

            Returns:
                Scalar loss tensor
            """
            ...

        def compute_metrics(
            self, model_output: Tensor, batch: BatchData
        ) -> Dict[str, float]:
            """Compute evaluation metrics.

            Returns:
                Dict of metric names to values (e.g., "auc": 0.95)
            """
            ...

        def format_output(self, model_output: Tensor) -> Dict[str, Tensor]:
            """Format predictions for external use.

            Returns:
                Dict with task-specific keys (e.g., "scores" for edge tasks)
            """
            ...

Implementations:

- ``EdgePredictAdapter`` — Link prediction (BCE loss, AUC/AP metrics)
- ``NodeRegressAdapter`` — Node regression (MSE loss, MAE/RMSE metrics)
- ``NodeClassifyAdapter`` — Node classification (cross-entropy, accuracy)

``StateManager`` — Stateful Model Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manages RNN/LSTM state across time steps or snapshots.

.. code-block:: python

    class StateManager(Protocol):
        """Manages node state (memory, RNN hidden, LSTM hidden+cell)."""

        def prepare(
            self, node_ids: Tensor, timestamps: Tensor | None = None
        ) -> Dict[int, Tensor]:
            """Initialize state for nodes and time point.

            Args:
                node_ids: [B] node IDs
                timestamps: [B] optional time values

            Returns:
                Dict mapping node_id → state tensor(s)
            """
            ...

        def update(
            self, node_ids: Tensor, output: Tensor, chunk_id: int | None = None
        ) -> None:
            """Update state after model produces output.

            Args:
                node_ids: [B] nodes updated
                output: [B, D] model output / new state
                chunk_id: Optional chunk ID for chunked processing
            """
            ...

        def reset(self) -> None:
            """Clear all state (e.g., between epochs)."""
            ...

        def describe(self) -> str:
            """Human-readable state description."""
            ...

Implementations:

- ``DummyStateManager`` — No-op state (for stateless tasks)
- ``CTDGMemoryManager`` — GRU-based memory bank (CTDG)
- ``RNNStateManager`` — Per-node RNN state (DTDG/Chunk)

``TemporalModel`` — Forward & State Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core model inference interface.

.. code-block:: python

    class TemporalModel(Protocol):
        """Temporal GNN model with optional state."""

        def forward(
            self,
            batch: BatchData,
            state: Dict[int, Tensor] | None = None,
        ) -> Tensor:
            """Forward pass on batch.

            Args:
                batch: BatchData with node_ids, edges, etc.
                state: Optional per-node state from StateManager

            Returns:
                [B, output_dim] predictions or embeddings
            """
            ...

        def compute_state_update(
            self, batch: BatchData, output: Tensor
        ) -> Dict[int, Tensor] | None:
            """Compute new state from output.

            Optional: return None if stateless.

            Returns:
                Dict node_id → new state, or None
            """
            ...

Implementations:

- Any PyTorch ``nn.Module`` with compatible forward() signature
- Can use task heads (EdgePredictHead, NodeRegressHead, NodeClassifyHead)
- Can combine backbone (GCNStack, TemporalTransformerConv) + head

``FeatureStore`` — Feature Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`data_layer` for full reference.

.. code-block:: python

    class FeatureStore(Protocol):
        """Node and edge feature storage."""

        def get_node_feat(self, node_ids: Tensor) -> Tensor:
            """[N, F_node]"""
            ...

        def get_edge_feat(self, edge_ids: Tensor) -> Tensor:
            """[E, F_edge]"""
            ...

        @property
        def node_feat_dim(self) -> int: ...

        @property
        def edge_feat_dim(self) -> int: ...

``GlobalCSR`` — Graph Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`data_layer` for full reference.

.. code-block:: python

    class GlobalCSR(Protocol):
        """CSR adjacency access."""

        @property
        def rowptr(self) -> Tensor: ...

        @property
        def col(self) -> Tensor: ...

        def neighbors(self, node_id: int) -> Tensor: ...

        def subgraph(self, node_ids: Tensor) -> "GlobalCSR": ...

Example: Using Protocols in Custom Code
----------------------------------------

**Implement a Custom Task**:

.. code-block:: python

    from starry_unigraph.registry.task_adapter import TaskAdapter
    from starry_unigraph.data import BatchData, SampleConfig

    class MyCustomTask(TaskAdapter):
        """My task: node multi-label classification."""

        def build_sample_config(self) -> SampleConfig:
            return SampleConfig(
                task_type="custom_multilabel",
                num_hops=2,
                num_neighbors=10
            )

        def compute_loss(self, model_output: Tensor, batch: BatchData) -> Tensor:
            return F.binary_cross_entropy_with_logits(
                model_output, batch.labels.float()
            )

        def compute_metrics(self, model_output: Tensor, batch: BatchData) -> Dict[str, float]:
            pred = (torch.sigmoid(model_output) > 0.5).float()
            acc = (pred == batch.labels).float().mean().item()
            return {"accuracy": acc}

        def format_output(self, model_output: Tensor) -> Dict[str, Tensor]:
            return {"logits": model_output, "probs": torch.sigmoid(model_output)}

Register:

.. code-block:: python

    from starry_unigraph.registry import task_registry

    task_registry.register("my_custom_task", MyCustomTask())

Use in training:

.. code-block:: python

    task_adapter = task_registry.get("my_custom_task")
    for batch in backend.iter_batches("train", batch_size=32):
        output = model(batch)
        loss = task_adapter.compute_loss(output, batch)
        loss.backward()

**Implement a Custom Backend**:

.. code-block:: python

    from starry_unigraph.runtime.backend import GraphBackend
    from starry_unigraph.data import BatchData

    class MyCustomBackend(GraphBackend):
        """My graph mode: special edge streaming."""

        def iter_batches(self, split: str, batch_size: int) -> Iterator[BatchData]:
            for batch_edges in self.edge_stream.iter_batches(split, batch_size):
                yield BatchData(
                    edges=batch_edges,
                    labels=...,
                    timestamps=...,
                    metadata={"split": split}
                )

        def reset(self) -> None:
            self.edge_stream.reset()

        def describe(self) -> str:
            return f"MyCustomBackend(mode=streaming, edges={len(self.edges)})"

Use in training:

.. code-block:: python

    backend = MyCustomBackend(config)
    engine = PipelineEngine(backend, task_adapter, state_manager, model)
    engine.run_epoch("train", batch_size=32)

Protocol Hierarchy & Composition
---------------------------------

.. code-block:: text

    PipelineEngine
        ├─ GraphBackend (iter_batches, reset, describe)
        ├─ TaskAdapter (build_sample_config, compute_loss, etc.)
        ├─ StateManager (prepare, update, reset)
        ├─ TemporalModel (forward, compute_state_update)
        │   └─ Backbone (GCNStack, TemporalTransformerConv)
        │   └─ Task Head (EdgePredictHead, NodeRegressHead)
        └─ Metrics (computed via TaskAdapter)

Each layer is independent:

- Multiple models can use the same GraphBackend
- Multiple backends can use the same TaskAdapter
- New components only need to satisfy the protocol

See Also
--------

- :doc:`unified_pipeline` — How protocols compose in PipelineEngine
- :doc:`data_layer` — Data structures passed through protocols
- Source: ``runtime/backend.py``, ``registry/task_adapter.py``, ``models/base.py``
