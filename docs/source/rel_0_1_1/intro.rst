Introduction
============

Background
----------

Many real-world systems can be formulated as temporal interaction graphs,
including social networks, citation graphs, transaction networks, and
recommender systems. In these settings, nodes represent entities and edges
represent interactions. Those interactions are often timestamped, so the graph
structure and its semantics evolve over time rather than staying fixed.

Temporal Graph Neural Networks (temporal GNNs) are designed for this setting.
They model both structural dependencies and temporal dependencies, allowing the
system to learn dynamic node or edge representations as the graph changes.
Depending on how time is represented, temporal GNN workloads are usually
organized in two ways:

1. **Continuous-time temporal graphs**, where the graph is treated as an
   ordered stream of timestamped interactions.
2. **Discrete-time temporal graphs**, where the graph is treated as an ordered
   sequence of snapshots.

As temporal graphs scale to millions of nodes and billions of interactions,
training becomes substantially harder. Sampling historical neighborhoods,
tracking chronological state, and coordinating computation across long temporal
ranges all become expensive. This is the setting StarryGL is built for.

Dynamic graphs are more than just large graphs. Their structure, features, and
temporal dependencies evolve together over time. Social interactions, payment
networks, recommender systems, and citation graphs all fall into this
category. In these workloads, the real challenge is not merely running a
Graph Neural Network (GNN), but doing so while graph topology and node states
continue to change.

To handle industrial-scale datasets with hundreds of millions of nodes and
billions of temporal events, training must scale beyond a single device.
Distributed training becomes essential, partitioning graph data and
computational workloads across multiple GPUs and machines. However,
distributing a dynamic graph introduces significant complexity. Unlike static
graphs, whose cross-partition boundaries remain fixed, temporal graphs require
continuous synchronization of evolving node states and chronological
dependencies across network connections.

Why StarryGL?
-------------

Scaling temporal graph training across a distributed cluster exposes three
severe bottlenecks that traditional pipelines struggle to handle:

1. **Memory Explosion**. A realistic distributed training job requires
   historical snapshots, sampled temporal neighborhoods, dynamic node states,
   and intermediate buffers to coexist in memory. On large datasets, this
   temporal history can easily exceed the memory capacity of even high-end
   multi-GPU systems.
2. **Communication Limits**. In distributed training, GPUs must constantly
   exchange features, states, and routing metadata. For temporal workloads,
   this overhead is amplified because the system must preserve strict
   chronological dependencies in addition to structural dependencies.
3. **Workload Imbalance**. Real-world dynamic graphs are highly skewed both
   structurally, for example by power-law degree distributions, and
   temporally, for example by bursty events and hot-spot nodes. When these
   workloads are partitioned across multiple GPUs, the skew naturally leads to
   severe load imbalance. Fast workers idle at synchronization barriers while
   overloaded workers finish dense temporal blocks, significantly degrading
   overall cluster throughput.

**StarryGL** is built from the ground up to address these bottlenecks. It
provides a highly optimized, unified runtime for large-scale dynamic graph
training with four core capabilities:

1. **Scalable Distributed Training** across multiple GPUs and nodes.
2. **Adaptive Communication-Computation Overlap** to hide network latency and
   reduce workload imbalance across devices.
3. **Mode-Specific System Optimizations** such as Hotspot Memory Sharing and
   Temporal Decay for both discrete and continuous temporal workloads.
4. **A Chunk-Centric Execution Abstraction** that reduces memory
   fragmentation and enables fine-grained load distribution across the training
   pipeline.

Two Graph Modes
---------------

StarryGL supports the two mainstream formulations of temporal graphs. Choose
the graph mode that matches your data and model semantics, and StarryGL will
load the corresponding optimized backend automatically.

**DTDG (Discrete-Time Dynamic Graph)**
   The graph is represented as a sequence of snapshots. This mode is a natural
   fit when you care about graph states over defined time windows, such as
   daily structural summaries or rolling interaction networks.

**CTDG (Continuous-Time Dynamic Graph)**
   The graph is represented as a stream of timestamped events. This mode is
   ideal when exact event ordering and fine-grained temporal resolution
   matter, such as in temporal link prediction, fraud detection, and real-time
   interaction modeling.

Chunk-Based Unified System Architecture (Roadmap)
-------------------------------------------------

Historically, temporal graph training systems evolved into two disconnected
worlds: one pipeline specialized for snapshots (DTDG) and another for event
streams (CTDG). That separation made it difficult to reuse scheduling logic,
communication primitives, and preprocessing artifacts across graph modes.

StarryGL is moving toward a cleaner, unified architecture. While the current
**DTDG** and **CTDG** pipelines are fully supported and stable for production
use, we are also developing a shared **chunk-centric execution model** as the
next-generation backend.

This chunk-centric runtime is currently **experimental**. It is designed to
standardize how data flows through the distributed cluster. In future
releases, this unified logic will gradually replace the separate training
pipelines, reducing redundant execution paths. This transition will make the
system easier to understand, simpler to extend, and more efficient to schedule
on parallel hardware.

How To Read This Documentation
------------------------------

If you are new to the project, read the pages in the following order:

1. Start with the training guide that matches your workload. These guides
   describe the stable, production-ready pipelines:

   - :doc:`training/dtdg` for snapshot-based temporal training
   - :doc:`training/ctdg` for event-driven temporal training

2. Read :doc:`training/chunk` if you want to understand the experimental
   chunk-centric abstraction that aims to unify discrete and continuous
   temporal workloads under one execution model:

   - :doc:`training/chunk` for the experimental chunk-centric execution path

3. Use :doc:`reference` when you need the module map, code structure, and
   deeper implementation notes.
