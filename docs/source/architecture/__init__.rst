Architecture & Layer Reference
==============================

This section documents the internal architecture of StarryUniGraph: how raw
temporal data is preprocessed, how runtime artifacts are laid out, how routing
and data movement are represented, and which reusable model modules are shared
across graph modes.

For user-facing workflow guides, start with :doc:`../rel_0_1_1/intro` and the
matching pages under :doc:`../rel_0_1_1/training/index`.

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

**Current execution paths**

- **CTDG**: event-driven processing over continuous-time interaction streams
- **DTDG**: snapshot-driven processing over discrete-time graph sequences
- **Chunk**: an experimental unified abstraction that maps both modes into
  schedulable spatio-temporal chunks

**Shared architectural layers**

- **Data structures** such as batch containers, partition metadata, and chunk
  descriptors
- **Routing and communication plans** for cross-partition feature and state
  exchange
- **Preprocessing contracts** that transform raw graph data into runtime-ready
  artifacts
- **Reusable modules** for time encoding, graph convolution, recurrent state
  updates, and temporal attention

.. note::

   The architecture pages focus on the current repository layout and the
   interfaces exposed by the active codebase. Experimental components are
   labeled explicitly where appropriate.
