Unified Training Pipeline
==========================

The ``PipelineEngine`` orchestrates training by composing protocols:
GraphBackend (data), TaskAdapter (loss), StateManager (state), and TemporalModel (forward).

This enables all 9 combinations of graph mode × task type without mode-specific branching.

Architecture
------------

.. code-block:: text

    PipelineEngine
    ├─ GraphBackend.iter_batches()  ──→ BatchData
    │  └─ (CTDG/DTDG/Chunk-specific)
    │
    ├─ BatchData ──→ materialize()
    │  └─ task_adapter.build_sample_config()
    │
    ├─ TemporalModel.forward(batch)  ──→ output
    │  └─ (any PyTorch model)
    │
    ├─ TaskAdapter.compute_loss()  ──→ loss
    │  └─ (task-specific)
    │
    ├─ TaskAdapter.compute_metrics()  ──→ metrics
    │  └─ (AUC, RMSE, accuracy, etc.)
    │
    ├─ StateManager.prepare()  ──→ state
    │
    └─ StateManager.update()  ──→ state

Complete Training Loop
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    engine = PipelineEngine(
        backend=backend,           # GraphBackend implementation
        task_adapter=adapter,      # TaskAdapter implementation
        state_manager=state_mgr,   # StateManager implementation
        model=model,               # TemporalModel (nn.Module)
    )

    # Full epoch
    losses, metrics = engine.run_epoch("train", batch_size=32)

    # Per-batch control
    for batch_idx, batch in enumerate(engine.iter_batches_with_step("train", batch_size=32)):
        loss = batch["loss"]
        metrics = batch["metrics"]
        # Custom processing...

Implementation: ``runtime/engine.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class PipelineEngine:
        """Unified training loop across all graph modes and tasks."""

        def __init__(
            self,
            backend: GraphBackend,
            task_adapter: TaskAdapter,
            state_manager: StateManager,
            model: nn.Module,
        ):
            self.backend = backend
            self.task_adapter = task_adapter
            self.state_manager = state_manager
            self.model = model

        def run_epoch(self, split: str, batch_size: int) -> tuple[List[float], Dict[str, float]]:
            """Run full epoch and return (losses, avg_metrics)."""
            losses = []
            metrics_accum = defaultdict(list)

            for batch_idx, result in enumerate(self.iter_batches_with_step(split, batch_size)):
                losses.append(result["loss"].item())
                for metric_name, metric_val in result["metrics"].items():
                    metrics_accum[metric_name].append(metric_val)

            avg_metrics = {k: mean(v) for k, v in metrics_accum.items()}
            return losses, avg_metrics

        def iter_batches_with_step(self, split: str, batch_size: int):
            """Yield per-batch results (loss, metrics)."""
            for batch in self.backend.iter_batches(split, batch_size):
                # Prepare state
                state = self.state_manager.prepare(batch.node_ids, batch.timestamps)

                # Forward pass
                with torch.no_grad():
                    output = self.model(batch, state=state)

                # Loss and metrics
                loss = self.task_adapter.compute_loss(output, batch)
                metrics = self.task_adapter.compute_metrics(output, batch)

                # Update state
                state_update = getattr(self.model, 'compute_state_update', lambda *a: None)(batch, output)
                if state_update:
                    self.state_manager.update(batch.node_ids, state_update)

                yield {
                    "batch": batch,
                    "output": output,
                    "loss": loss,
                    "metrics": metrics,
                }

Dispatch Flow
~~~~~~~~~~~~~

The dispatch happens **once** at initialization:

.. code-block:: python

    # Determine graph mode from config
    graph_mode = config.data.get('graph_mode') or infer_from_model(config)

    # Build backend
    if graph_mode == "ctdg":
        backend = CTDGGraphBackend(ctdg_session)
    elif graph_mode == "dtdg":
        backend = FlareGraphBackend(flare_loader)
    elif graph_mode == "chunk":
        backend = ChunkGraphBackend(chunk_loader)

    # Get task adapter
    task_adapter = task_registry.get(config.task.task_type)

    # Build state manager
    state_manager = RNNStateManager() or DummyStateManager()

    # Create engine (no more mode-specific branching!)
    engine = PipelineEngine(backend, task_adapter, state_manager, model)
    engine.run_epoch("train", batch_size=32)

Design Benefits
~~~~~~~~~~~~~~~

1. **No Mode-Specific Code in Training Loop**

   - Before: If/else branches for CTDG/DTDG/Chunk in train_epoch()
   - After: Single generic run_epoch() works for all modes

2. **Composable Components**

   - Swap backends: CTDG → DTDG without changing loop
   - Swap tasks: EdgePredict → NodeRegress without changing loop
   - Swap models: TGN → MPNN without changing loop

3. **Extensibility**

   - New graph mode? Implement GraphBackend protocol
   - New task? Implement TaskAdapter protocol
   - New state manager? Implement StateManager protocol
   - No changes to PipelineEngine

4. **Testability**

   - Use DummyGraphBackend, DummyStateManager for unit tests
   - Mock any protocol for isolation testing

Example: Custom Training Loop
-----------------------------

For research, you can extend PipelineEngine or use iter_batches_with_step():

.. code-block:: python

    engine = PipelineEngine(backend, task_adapter, state_manager, model)

    for batch_idx, result in enumerate(engine.iter_batches_with_step("train", batch_size=32)):
        loss = result["loss"]
        output = result["output"]
        batch = result["batch"]
        metrics = result["metrics"]

        # Custom gradient accumulation
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Custom logging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss={loss:.4f}, {metrics}")

Integration with SchedulerSession
----------------------------------

The unified entry point (SchedulerSession) builds PipelineEngine:

.. code-block:: python

    class SchedulerSession:
        """Unified training orchestrator."""

        def build_pipeline_engine(self, model: nn.Module) -> PipelineEngine:
            """Build PipelineEngine from prepared artifacts.

            Args:
                model: TemporalModel instance

            Returns:
                PipelineEngine ready for run_epoch()
            """
            # Dispatch based on graph_mode
            backend = self._build_backend()  # Returns GraphBackend
            adapter = task_registry.get(self.config.task.task_type)
            state_mgr = self._build_state_manager()

            return PipelineEngine(backend, adapter, state_mgr, model)

        def run_epoch(self, split, batch_size, model):
            """High-level epoch runner."""
            engine = self.build_pipeline_engine(model)
            losses, metrics = engine.run_epoch(split, batch_size)
            return losses, metrics

Usage Example
-------------

Complete training loop:

.. code-block:: python

    from starry_unigraph import SchedulerSession
    from starry_unigraph.models import WrappedModel

    # Initialize
    session = SchedulerSession.from_config("config.yaml")
    session.prepare_data()

    # Build model
    model = WrappedModel(
        backbone=GCNStack(input_size=64, hidden_size=128),
        head=EdgePredictHead(hidden_size=128, output_dim=1)
    )
    model = model.to("cuda:0")

    # Build engine
    engine = session.build_pipeline_engine(model)

    # Training loop
    for epoch in range(num_epochs):
        train_losses, train_metrics = engine.run_epoch("train", batch_size=32)
        val_losses, val_metrics = engine.run_epoch("val", batch_size=128)

        print(f"Epoch {epoch}: train_loss={mean(train_losses):.4f}, "
              f"val_metrics={val_metrics}")

Multi-Mode Training Comparison
-------------------------------

The same code works for all modes:

.. code-block:: python

    # CTDG (online)
    config_ctdg = load_config("configs/ctdg.yaml")
    session_ctdg = SchedulerSession.from_config(config_ctdg)
    engine_ctdg = session_ctdg.build_pipeline_engine(model)

    # DTDG (snapshot pipeline)
    config_dtdg = load_config("configs/dtdg.yaml")
    session_dtdg = SchedulerSession.from_config(config_dtdg)
    engine_dtdg = session_dtdg.build_pipeline_engine(model)

    # Chunk (experimental)
    config_chunk = load_config("configs/chunk.yaml")
    session_chunk = SchedulerSession.from_config(config_chunk)
    engine_chunk = session_chunk.build_pipeline_engine(model)

    # All three use identical training code!
    for engine in [engine_ctdg, engine_dtdg, engine_chunk]:
        losses, metrics = engine.run_epoch("train", batch_size=32)

See Also
--------

- :doc:`data_layer` — BatchData structure
- :doc:`protocols` — All protocol interfaces
- Source: ``runtime/engine.py``, ``session.py``, ``runtime/backend_adapters.py``
