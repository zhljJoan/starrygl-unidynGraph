Architecture & Layer Reference
=====================================

Deep technical documentation of StarryUniGraph internal architecture.

These documents provide detailed reference on how data flows through the system,
how protocols abstract over graph modes (CTDG/DTDG/Chunk), and how to extend the
library with new backends, tasks, or components.

For high-level tutorials and quick starts, see :doc:`../tutorial/index`.

.. toctree::
   :maxdepth: 2

   data_layer
   route_layer
   preprocess_layer
   protocols
   artifact_format
   unified_pipeline
   model_modules

Key Concepts
============

**Three Independent Paths**:

- **CTDG** (Continuous-Time): Online event-driven processing with temporal neighbors
- **DTDG** (Discrete-Time with Flare): Snapshot-based processing with multi-GPU support
- **Chunk** (Future): Time-window and node-cluster based processing

**Unified Protocols**:

- Data structures (BatchData, SampleConfig) represent graph modes uniformly
- Task adapters (TaskAdapter) decouple task logic from graph modes
- Backend abstractions (GraphBackend, StateManager) enable composable training loops
- Routes (Route, RouteData) abstract feature exchange across partitions

**Library Stability**:

The stable public API (v0.1.0+) is exported via ``starry_unigraph.lib_stable``.
This includes all core data structures, protocols, models, and training components.
See the module docstring for stability guarantees and explicit export list.

.. note::

   These layers are continuously evolving. For the current development version
   and the latest changes, see the project README and ``CLAUDE.md``.

