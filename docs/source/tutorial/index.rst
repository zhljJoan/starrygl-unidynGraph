Tutorials & Architecture
========================

.. toctree::
   :maxdepth: 1

High-level Guides
-----------------

.. toctree::
   :maxdepth: 1

   intro
   UniTraining
   CTDGTraining
   DTDGTraining
   ChunkTraining

Deep Technical Reference
------------------------

For implementation details, architectural decisions, and extensibility,
see the :doc:`../architecture/__init__` documentation.

Key architecture documents:

- :doc:`../architecture/data_layer` — Data structures (BatchData, RouteData, SampleConfig)
- :doc:`../architecture/route_layer` — Distributed feature exchange patterns
- :doc:`../architecture/preprocess_layer` — Preprocessing pipelines per graph mode
- :doc:`../architecture/protocols` — Protocol definitions (GraphBackend, TaskAdapter)
- :doc:`../architecture/unified_pipeline` — PipelineEngine training loop
- :doc:`../architecture/artifact_format` — Artifact files and serialization
- :doc:`../architecture/model_modules` — Reusable model components
