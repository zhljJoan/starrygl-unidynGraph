Unified Training Pipeline
=========================

The unified pipeline is an experimental refactoring path that tries to express
CTDG, DTDG, and Chunk execution through the same high-level loop. The core
entry points are:

- ``starry_unigraph.runtime.engine.PipelineEngine``
- ``starry_unigraph.runtime.backend`` protocol definitions
- ``starry_unigraph.runtime.backend_adapters`` runtime adapters
- ``starry_unigraph.session.SchedulerSession.build_pipeline_engine()``

This path exists in the current codebase and is usable, but it is still more
prototype-like than the backend-native runtime flows.

Current Flow
------------

At a high level, the current engine composes the following pieces:

.. code-block:: text

    GraphBackend.iter_batches()
        -> ChunkAtomic
        -> TaskAdapter.build_sample_config()
        -> PipelineEngine._materialize_batch()
        -> BatchData
        -> StateManager.prepare()
        -> model.predict(state, batch)
        -> TaskAdapter.compute_loss() / compute_metrics()
        -> StateManager.update(model_output, chunk)

Two details matter for understanding the real implementation:

- ``GraphBackend`` currently yields ``ChunkAtomic`` objects, not final
  ``BatchData``.
- ``PipelineEngine._materialize_batch()`` is currently a placeholder that builds
  a minimal ``BatchData`` from the chunk and sample config.

``PipelineEngine`` in ``runtime/engine.py``
-------------------------------------------

The current class signature is:

.. code-block:: python

    class PipelineEngine:
        def __init__(
            self,
            backend: GraphBackend,
            state_manager: StateManager,
            model: nn.Module,
            task_adapter: TaskAdapter,
            device: str = "cpu",
        ):
            ...

Its main public methods are:

.. code-block:: python

    def run_epoch(
        self,
        split: str = "train",
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        ...

    def iter_batches_with_step(
        self,
        split: str = "train",
        batch_size: int = 64,
    ) -> Iterator[Dict[str, Any]]:
        ...

``run_epoch()`` currently returns a dictionary with epoch-level summary data:

.. code-block:: python

    {
        "split": split,
        "loss": avg_loss,
        "num_batches": len(outputs),
        "metrics": avg_metrics,
        "outputs": outputs,
    }

``iter_batches_with_step()`` yields per-step dictionaries:

.. code-block:: python

    {
        "loss": float | None,
        "metrics": Dict[str, float],
        "batch_idx": int,
        "output": Dict[str, Any],
    }

Mode Adapters
-------------

The current unified path adapts existing runtimes rather than replacing them.

``CTDGGraphBackend``
~~~~~~~~~~~~~~~~~~~~

- wraps ``CTDGSession``
- uses ``iter_train()`` / ``iter_eval()``
- converts runtime batches into placeholder ``ChunkAtomic`` objects

``FlareGraphBackend``
~~~~~~~~~~~~~~~~~~~~~

- wraps ``FlareRuntimeLoader``
- uses ``iter_train()`` / ``iter_eval()``
- converts ``STGraphBlob`` windows into placeholder ``ChunkAtomic`` objects

``ChunkGraphBackend``
~~~~~~~~~~~~~~~~~~~~~

- wraps ``ChunkRuntimeLoader``
- forwards chunk iterators directly

This means the unified pipeline currently sits on top of the existing backend
implementations instead of replacing their internal loaders and samplers.

Integration with ``SchedulerSession``
-------------------------------------

``SchedulerSession.build_pipeline_engine()`` is the session-level hook that
builds the optional unified engine after artifact loading.

At a high level, it:

1. loads prepared artifacts if needed
2. detects ``graph_mode`` from the prepared metadata
3. constructs the matching backend adapter
4. creates a placeholder ``DummyStateManager``
5. returns ``PipelineEngine(...)``

This is separate from the stable backend-native train/eval/predict paths in
``SchedulerSession.build_runtime()`` and the mode-specific helper functions.

Usage Example
-------------

.. code-block:: python

    from starry_unigraph.session import SchedulerSession

    session = SchedulerSession.from_config("config.yaml")
    session.prepare_data()
    session.build_runtime()

    engine = session.build_pipeline_engine(model=my_model)
    epoch_result = engine.run_epoch(split="train", batch_size=32)
    print(epoch_result["loss"], epoch_result["metrics"])

Current Limitations
-------------------

- Batch materialization is still a placeholder in ``PipelineEngine``.
- The default state manager is ``DummyStateManager``.
- Backend adapters use minimal chunk conversion shims.
- The production CTDG and DTDG runtimes remain the more complete execution
  paths for real training.

So this layer is best read as the current unification direction, not as the
only stable runtime contract.

See Also
--------

- :doc:`protocols` — The interfaces composed by this engine
- :doc:`data_layer` — ``ChunkAtomic``, ``BatchData``, and related containers
- Source: ``runtime/engine.py``, ``runtime/backend.py``,
  ``runtime/backend_adapters.py``, ``session.py``
