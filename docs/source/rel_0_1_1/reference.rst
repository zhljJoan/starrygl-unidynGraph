Reference and Architecture
==========================

Module Reference
----------------

The StarryGL repository is organized around a small number of core
responsibilities:

- ``starry_unigraph/session.py``: :class:`SchedulerSession` coordinates
  preprocessing, runtime creation, training, evaluation, and prediction.
- ``starry_unigraph/preprocess/``: Converts raw graph data into runtime-ready
  artifacts.
- ``starry_unigraph/backends/dtdg/``: Implements the snapshot-oriented runtime
  used by **DTDG** mode.
- ``starry_unigraph/backends/ctdg/``: Contains event-driven runtime components
  used by **CTDG** mode.
- ``starry_unigraph/preprocess/chunk.py``: Contains the experimental chunk
  preprocessing path and related data compilation logic.

Code Structure
--------------

.. code-block:: text

   StarryUniGraph/
   ├── configs/                   # Ready-to-use YAML configuration templates
   ├── docs/                      # Documentation source files
   ├── tests/                     # Test suite
   ├── train_tgn.py
   ├── train_tgn_dist.py
   ├── train_mpnn_lstm_4gpu.py
   ├── run_tgn_ctdg_flare.sh
   ├── run_mpnn_lstm_4gpu.sh
   └── starry_unigraph/          # Core framework source code
       ├── __main__.py           # Unified CLI entry point
       ├── session.py            # Unified lifecycle API
       ├── distributed.py        # Distributed communication primitives
       ├── preprocess/
       ├── models/
       ├── tasks/
       ├── registry/
       ├── runtime/
       ├── backends/
       │   ├── dtdg/
       │   ├── ctdg/
       │   └── flare/            # Backward-compatibility exports
       ├── native/
       ├── lib/
       └── vendor/

Deep Dive: Mathematical Intuition
---------------------------------

**Temporal Decay in DTDG**

The Flare backend applies a conceptual decay mechanism to reduce the
representation cost of older temporal blocks. If :math:`\mathcal{N}_t` is the
number of chunks in the block at time :math:`t`, then the next older block
retains:

.. math::

   \mathcal{N}_{t-1} = \lfloor \beta \cdot \mathcal{N}_t \rfloor

This decay is implemented through batching policy rather than as a user-managed
formula.

**Memory Updates in CTDG**

When an interaction event connects node :math:`u` and :math:`v` at time
:math:`t`, a message is generated and memory is updated:

.. math::

   m_e^{uv} = \text{MSG}(s(u,t^-) \,\|\, s(v,t^-) \,\|\, e(u,v,t) \,\|\, \phi(t-t^-))

.. math::

   s(u,t) = \text{UPDATE}(s(u,t^-), m_e^{uv})

Frequently accessed nodes create heavy synchronization pressure. StarryGL's
Hotspot Memory Sharing optimization smooths these updates across partitions
without breaking the exact state transition logic.

**Communication-Computation Overlap**

By explicitly mapping structural and temporal dependencies, the runtime can
issue network transfers for the next batch while the GPU is executing the
current batch, hiding communication latency.

Publications and Citation
-------------------------

StarryGL is built on research and system optimizations developed at Zhejiang
University. If you use StarryGL or its methodologies in your research, please
consider citing the core papers:

**For DTDG and sliding-window optimizations (Flare backend):**

   Wenjie Huang, Rui Wang, Jing Cao, Tongya Zheng, Xinyu Wang, Mingli Song,
   Sai Wu, and Chun Chen. "FlareDTDG: Harnessing Temporal Recency for Scalable
   Discrete-Time Dynamic Graph Training." *Proceedings of the VLDB Endowment
   (PVLDB)*, 2025.

**For CTDG and event-driven optimizations (MemShare backend):**

   Longjiao Zhang, Rui Wang, Tongya Zheng, Ziqi Huang, Wenjie Huang, Xinyu
   Wang, Can Wang, Mingli Song, Sai Wu, and Shuibing He. "Effective and
   Efficient Distributed Temporal Graph Learning through Hotspot Memory
   Sharing." *Proceedings of the VLDB Endowment (PVLDB)*, 18(9): 3093-3105,
   2025. doi:10.14778/3746405.3746430.
