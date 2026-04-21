Installation & Quick Start
===========================

This document provides installation instructions and quick start examples for StarryUniGraph.

Prerequisites
=============

Before starting the installation, ensure you meet the following requirements:

- **Operating System**: Linux (tested on Ubuntu 18.04+)
- **Python Version**: Python 3.9+ (tested with 3.10)
- **CUDA Toolkit**: CUDA 11.8+ (for GPU support)
- **Package Manager**: pip or conda

Installation
============

1. Clone the Repository
-----------------------

Clone the StarryUniGraph repository from GitHub:

.. code-block:: bash

    git clone https://github.com/zhljJoan/starrygl-unidynGraph.git
    cd starrygl-unidynGraph

2. Install Dependencies
-----------------------

Install required Python packages:

.. code-block:: bash

    pip install -r requirements.txt

Common dependencies include:

- PyTorch 2.0+
- DGL (Deep Graph Library)
- NumPy, SciPy
- PyYAML (for config files)
- tqdm (progress bars)

3. Install StarryUniGraph
-------------------------

Install the package in development mode:

.. code-block:: bash

    pip install -e .

Or install in production mode:

.. code-block:: bash

    python setup.py install

4. Verify Installation
----------------------

Verify the installation:

.. code-block:: bash

    python -c "import starry_unigraph; print(starry_unigraph.__version__)"

You should see the version number printed.

Quick Start
===========

Single GPU Training (CTDG)
--------------------------

Train on a single GPU using CTDG (Continuous-Time Dynamic Graphs):

.. code-block:: bash

    python -m starry_unigraph --config configs/tgn_wiki.yaml --phase all

Output:
- Preprocessed artifacts in ``output/artifacts/``
- Trained model checkpoint in ``output/checkpoint/``
- Training logs in ``output/logs/``

Multi-GPU Training (Single Machine)
------------------------------------

Train using multiple GPUs on a single machine with CTDG:

.. code-block:: bash

    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/tgn_wiki_multigpu.yaml \
        --phase all

CTDG Multi-GPU support:
- ✅ Single-machine multi-GPU via partitioning + NCCL all-to-all
- ✅ Event-based partitioning (round-robin or SPEED for load balancing)
- ✅ Memory bank synchronization across GPUs
- ✅ Async communication for latency hiding

**Characteristics**:
- **Communication**: NCCL collective all-to-all (~100-500 µs latency)
- **Scalability**: 2-8 GPUs on same machine (PCIe/NVLink fabric)
- **Memory**: Partitioned storage (each GPU stores ~1/N nodes)
- **Best for**: Dense neighborhoods, large temporal windows, social networks

Multi-Machine Distributed Training (CTDG via NCCL)
---------------------------------------------------

Train using multiple machines with CTDG using NCCL over TCP/IP:

.. code-block:: bash

    # On master node (rank=0)
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/tgn_wiki_distributed.yaml \
        --phase all

    # On worker node (rank=1)
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/tgn_wiki_distributed.yaml \
        --phase all

Configuration for multi-machine CTDG:

.. code-block:: yaml

    data:
      graph_mode: ctdg
      source: data/my_events.csv
      num_partitions: 8  # Multiple partitions (2 machines × 4 GPUs)
      partition: speed  # or round_robin

    training:
      batch_size: 32
      num_epochs: 100

CTDG Multi-Machine Support:
- ✅ Arbitrary number of machines via NCCL
- ✅ NCCL over TCP/IP for cross-machine communication
- ✅ Same communication pattern as single-machine (all-to-all_single)
- ✅ Node partitioning (SPEED or round-robin) for load balancing

**How It Works**:

CTDG uses **the same NCCL all-to-all_single collective** for both single-machine and multi-machine:

1. **Single-Machine**: NCCL uses NVLINK/PCIe → ~100-500 µs
2. **Multi-Machine**: NCCL uses TCP/IP → ~10-50 ms (network layer difference only)

The communication **pattern is identical** — only the underlying network transport changes.

**Characteristics**:
- **Communication**: NCCL all-to-all_single (~100-500 µs local, ~10-50 ms over network)
- **Scalability**: 2+ machines with arbitrary GPUs per machine
- **Memory**: Partitioned storage (each rank stores ~1/(machines×GPUs) nodes)
- **Best for**: Large-scale temporal graphs, distributed training with NCCL infrastructure

**Comparison: CTDG Single-Machine vs Multi-Machine**:

.. list-table::
   :header-rows: 1

   * - Feature
     - Single-Machine
     - Multi-Machine

   * - Backend
     - NCCL (NVLINK/PCIe)
     - NCCL (TCP/IP)

   * - Latency
     - ~100-500 µs
     - ~10-50 ms

   * - Bandwidth
     - ~100-1000 GB/s
     - ~1-100 GB/s (network)

   * - Communication Pattern
     - all-to-all_single
     - all-to-all_single (same!)

   * - Typical Setup
     - 2-8 GPUs on 1 machine
     - 2+ machines with 1-4 GPUs each

   * - Network Protocol
     - PCIe/NVLINK fabric
     - Ethernet/InfiniBand over TCP

Multi-GPU Training (Distributed - DTDG)
----------------------------------------

Train using multiple GPUs with DTDG (Discrete-Time Dynamic Graphs):

.. code-block:: bash

    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_4gpu.yaml \
        --phase all

DTDG supports:
- ✅ Single-machine multi-GPU (Flare architecture)
- ✅ Multi-machine via snapshot partitioning (with torch.distributed)
- ✅ Mixed precision training
- ✅ Gradient accumulation

Prepare Your Data
=================

Data Format
-----------

StarryUniGraph expects temporal graphs in CSV format:

.. code-block:: text

    timestamp, source, destination, edge_feature_1, ..., edge_feature_n

Example:

.. code-block:: text

    1000.5, 10, 25, 0.1, 0.2, 0.3
    1001.2, 15, 30, 0.15, 0.25, 0.35
    1002.1, 20, 35, 0.2, 0.3, 0.4

Node and edge features should be in separate files:

- ``node_features.pt``: PyTorch tensor [num_nodes, feature_dim]
- ``edge_features.pt``: PyTorch tensor [num_edges, feature_dim]

Configuration
-------------

Create a YAML configuration file ``configs/my_dataset.yaml``:

.. code-block:: yaml

    data:
      graph_mode: ctdg  # or "dtdg" for discrete-time
      source: data/my_events.csv
      num_partitions: 4
      partition: speed  # or "round_robin"
      time_window: 3600  # seconds per snapshot (DTDG)

    model:
      family: tgn  # or "dyrep" for CTDG; "mpnn_lstm" for DTDG
      hidden_dim: 128
      edge_feat_dim: 16
      num_neighbors: 10

    task:
      task_type: edge_predict  # or "node_regression", "node_classify"

    training:
      batch_size: 32
      learning_rate: 0.0001
      num_epochs: 100

See ``docs/source/architecture/`` for detailed protocol documentation.

Creating Temporal GNN Models
=============================

Using the Unified Pipeline
---------------------------

StarryUniGraph provides a unified training pipeline that works for all graph modes and tasks:

.. code-block:: python

    from starry_unigraph import SchedulerSession, lib_stable
    from starry_unigraph.models import WrappedModel
    from starry_unigraph.runtime.modules import TimeEncode, GCNStack
    import torch.nn as nn

    # Load and preprocess data
    session = SchedulerSession.from_config("configs/my_config.yaml")
    session.prepare_data()

    # Build model
    backbone = GCNStack(input_size=64, hidden_size=128, num_layers=2)
    head = lib_stable.EdgePredictHead(hidden_size=128, output_dim=1)
    model = WrappedModel(backbone, head)
    model = model.to("cuda:0")

    # Build training engine
    engine = session.build_pipeline_engine(model)

    # Train
    for epoch in range(100):
        train_losses, train_metrics = engine.run_epoch("train", batch_size=32)
        val_losses, val_metrics = engine.run_epoch("val", batch_size=128)
        print(f"Epoch {epoch}: train_loss={sum(train_losses)/len(train_losses):.4f}")

Custom Models
--------------

Implement custom temporal GNN models:

.. code-block:: python

    import torch
    import torch.nn as nn
    from starry_unigraph.models import TemporalModel
    from starry_unigraph.runtime.modules import TimeEncode

    class MyTemporalGNN(nn.Module):
        """Custom temporal GNN model."""

        def __init__(self, input_dim: int, hidden_dim: int, time_dim: int):
            super().__init__()
            self.time_enc = TimeEncode(dim=time_dim)
            self.gnn = nn.Linear(input_dim + time_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, batch, state=None):
            # batch.node_ids [B]
            # batch.timestamps [B]
            # batch.edges [2, E]

            # Encode times
            time_feat = self.time_enc(batch.timestamps)

            # GNN forward
            h = self.gnn(time_feat)
            h = torch.relu(h)

            # Output layer
            output = self.output(h)
            return output

        def compute_state_update(self, batch, output):
            # Optional: return updated state for stateful models
            return None

Distributed Training
====================

**Summary Table: Training Modes**

.. list-table::
   :header-rows: 1

   * - Mode
     - Single-Machine Multi-GPU
     - Multi-Machine
     - Use Case

   * - **CTDG**
     - ✅ NCCL all-to-all
     - ✅ RPC (async)
     - Temporal state + memory update

   * - **DTDG**
     - ✅ Per-snapshot routing
     - ✅ Snapshot-based boundaries
     - Discrete snapshots + sparse sampling

   * - **Chunk**
     - ✅ (development)
     - ✅ (development)
     - Time-windowed + node clustering

Single Machine, Multiple GPUs (CTDG)
------------------------------------

CTDG on multi-GPU (same machine) with NCCL collective communication:

.. code-block:: bash

    # Edit config
    cat > configs/tgn_multigpu.yaml << 'EOF'
    data:
      graph_mode: ctdg
      source: data/wiki.csv
      num_partitions: 4
      partition: speed

    training:
      batch_size: 32
      num_epochs: 100
    EOF

    # Launch with torchrun
    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/tgn_multigpu.yaml \
        --phase train

**How CTDG Single-Machine Multi-GPU Works**:

1. **Event Partitioning**: Nodes partitioned by rank (SPEED or round-robin)
2. **Memory Bank**: Each GPU maintains partition of memory bank (1/N nodes)
3. **Feature Exchange**: NCCL all-to-all for cross-partition neighbor features
4. **Synchronization**: Async sync for remote memory updates
5. **Compute**: Forward pass with local + fetched features

**Performance Characteristics**:
- **Latency**: ~100-500 µs per all-to-all
- **Bandwidth**: ~100-1000 GB/s (PCIe/NVLink)
- **Scalability**: 2-8 GPUs optimal
- **Memory**: ~1/N per GPU (N = number of GPUs)

Multi-Machine (CTDG via RPC)
-----------------------------

CTDG on multiple machines with RPC-based remote memory access:

.. code-block:: bash

    # On master node (rank=0)
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/tgn_distributed.yaml \
        --phase train

    # On worker node (rank=1)
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/tgn_distributed.yaml \
        --phase train

**Configuration for multi-machine CTDG**:

.. code-block:: yaml

    data:
      graph_mode: ctdg
      source: data/my_events.csv
      num_partitions: 8  # 2 machines × 4 GPUs
      partition: speed

    distributed:
      backend: nccl
      use_rpc: true  # Enable RPC for multi-machine

    training:
      batch_size: 32
      num_epochs: 100

**How CTDG Multi-Machine Works**:

1. **Event Partitioning**: Nodes partitioned across all ranks (global partition map)
2. **Memory Bank**: Each rank stores ~1/(machines×GPUs) nodes
3. **Local Access**: All local nodes in GPU memory (fast)
4. **Remote Access**: RPC calls to other ranks (async, non-blocking)
5. **Async Batching**: Multiple RPC requests sent before waiting

**Performance Characteristics**:
- **Latency**: ~1-10 ms per RPC call (multi-machine)
- **Bandwidth**: ~1-100 GB/s (network dependent)
- **Scalability**: 2+ machines
- **Memory**: ~1/(machines×GPUs) per rank
- **Communication**: Asynchronous, overlappable with compute

Single Machine, Multiple GPUs (DTDG)
-------------------------------------

DTDG on multi-GPU (same machine):

.. code-block:: bash

    cat > configs/mpnn_lstm_multigpu.yaml << 'EOF'
    data:
      graph_mode: dtdg
      source: data/wiki.csv
      time_window: 3600
      num_partitions: 4
      partition_method: metis

    training:
      batch_size: 32
      num_epochs: 100
    EOF

    torchrun --nproc_per_node=4 -m starry_unigraph \
        --config configs/mpnn_lstm_multigpu.yaml \
        --phase train

Advantages:
- ✅ Supports both single-machine and multi-machine
- ✅ Per-snapshot partitioning enables flexible scaling
- ✅ Proven with Flare architecture

Multi-Machine (DTDG with torch.distributed)
-------------------------

To train across multiple machines, use DTDG with torch.distributed launcher:

.. code-block:: bash

    # On master node
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/mpnn_lstm_dist.yaml \
        --phase train

    # On worker node
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=<master_ip> \
        --master_port=29500 \
        -m starry_unigraph \
        --config configs/mpnn_lstm_dist.yaml \
        --phase train

Configuration for multi-machine:

.. code-block:: yaml

    data:
      graph_mode: dtdg
      source: data/wiki.csv
      time_window: 3600
      num_partitions: 8  # Increase for multi-machine
      partition_method: metis

    training:
      batch_size: 64    # Larger batches for distributed
      num_epochs: 100

Resources
=========

- **Documentation**: https://github.com/zhljJoan/starrygl-unidynGraph/tree/main/docs
- **Architecture Guide**: ``docs/source/architecture/``
- **Example Configs**: ``configs/``
- **GitHub Issues**: https://github.com/zhljJoan/starrygl-unidynGraph/issues

Troubleshooting
===============

**ImportError: No module named 'starry_unigraph'**

Install in development mode:

.. code-block:: bash

    pip install -e .

**CUDA Out of Memory**

Reduce batch size or use gradient accumulation:

.. code-block:: yaml

    training:
      batch_size: 16
      gradient_accumulation_steps: 2

**Distributed Training Hangs**

Check network connectivity and NCCL debug mode:

.. code-block:: bash

    export NCCL_DEBUG=INFO
    torchrun --nproc_per_node=4 -m starry_unigraph --config config.yaml

**Performance Issues**

Profile with torch.profiler or check I/O bottlenecks. See ``docs/source/architecture/unified_pipeline.rst`` for optimization tips.

See Also
========

- :doc:`../tutorial/UniTraining` — Unified training entry point
- :doc:`../tutorial/CTDGTraining` — CTDG-specific training
- :doc:`../tutorial/DTDGTraining` — DTDG distributed training
- :doc:`../architecture/__init__` — Architecture reference
- :doc:`../architecture/model_modules` — Reusable model components
