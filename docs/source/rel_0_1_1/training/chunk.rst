CHUNK Training Logic (Experimental)
===================================

The DTDG (snapshots) and CTDG (event streams) paths solve fundamentally
different modeling problems. Historically, graph learning systems have
implemented them as completely disconnected pipelines.

If you force an event stream (CTDG) into a snapshot abstraction, you lose
fine-grained temporal dependencies between individual interactions. If you
force snapshots (DTDG) into an event-stream abstraction, you lose the holistic
structural topology that exists within a defined time bucket. This
fragmentation forces systems to maintain two incompatible execution chains: one
scheduler and caching strategy for snapshots, and another entirely different
one for events.

StarryGL bridges this divide with its next-generation chunk-centric execution
model.

The Atomic CHUNK Abstraction
----------------------------

To eliminate pipeline fragmentation, StarryGL introduces a unified execution
and scheduling unit: the **CHUNK**.

A CHUNK is neither merely a snapshot nor merely an event batch. Instead, it is
a standardized, mode-agnostic unit that encapsulates:

- a bounded time range
- a bounded topology region, including local nodes and edges
- the metadata required for distributed routing
- the dependency information required for memory prefetching and synchronization

By compiling both discrete snapshots and continuous events down into these
standardized CHUNKs, StarryGL preserves high-level graph semantics while
feeding a single, highly optimized runtime underneath.

How DTDG and CTDG Map to CHUNKs
-------------------------------

To understand the value of this unified abstraction, it helps to see how the
user-facing graph modes are compiled into CHUNKs internally. The translation
focuses on mapping different graph semantics into the same spatio-temporal
coordinates.

Mapping DTDG (Snapshots) to CHUNKs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In DTDG mode, the translation is primarily structural.

- A single graph snapshot naturally maps to one time slice in the unified
  pipeline.
- A graph partitioning algorithm, such as METIS, splits the topology of that
  snapshot into multiple structural CHUNKs.
- Dependencies between CHUNKs are relatively straightforward: spatial
  dependencies resolve cross-partition edges, while temporal dependencies link
  a CHUNK to its corresponding spatial twin in the previous snapshot so it can
  inherit temporal state such as RNN hidden states.

Mapping CTDG (Event Streams) to CHUNKs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In CTDG mode, the translation is primarily chronological.

- A sequential batch of events, for example 200 timestamped interactions, is
  grouped to form a continuous time slice.
- Within that slice, the active nodes, their interactions, and the required
  historical neighbors are grouped into CHUNKs.
- Dependencies between CHUNKs are more dynamic. Instead of broadcasting node
  memories globally, the runtime inspects CHUNK metadata to establish targeted
  peer-to-peer memory synchronization routes before computation begins.

By expressing both structural snapshots and sequential events in the same
CHUNK vocabulary, StarryGL can reuse shared scheduling, pinned-memory
management, and distributed communication patterns across very different
temporal workloads.

How CHUNK Organizes Data
------------------------

The chunk pipeline reorganizes a large temporal graph in two orthogonal
dimensions:

**Temporal Slicing**
   The workload is first split along the time axis. This creates bounded
   temporal segments so chronological processing can be scheduled and prefetched
   predictably.

**Spatial Partitioning**
   Inside each temporal segment, the graph structure is partitioned into
   topology-aware regions such as clusters or local subgraphs.

The intersection of these two slices forms a CHUNK: a compact, independently
schedulable execution unit containing local data, time bounds, and explicit
references for remote cross-partition access.

Unified Dependency Model and Overlap
------------------------------------

Once the training workload is expressed as CHUNKs, distributed communication
becomes easier to reason about. Chunk-to-chunk dependencies are abstracted into
two dimensions:

**Spatial Dependencies**
   Data required from remote partitions because of graph topology, such as
   cross-GPU edges.

**Temporal Dependencies**
   State required from previous time steps, such as historical node memory or
   recurrent hidden states.

This abstraction is the basis for StarryGL's communication-computation overlap.
Because dependencies are declared at the CHUNK level, the runtime can look
ahead:

1. It identifies which remote features or memory states a future CHUNK will
   require.
2. It issues asynchronous prefetch requests during compute gaps in the current
   CHUNK.
3. It prunes unnecessary network transfers when the dependency graph shows that
   the next CHUNK does not require them.

End-to-End Pipeline
-------------------

For system developers targeting the experimental CHUNK API, the execution
pipeline follows six standardized steps:

1. **Preprocess**: Compile the raw temporal graph into partition-aware
   artifacts.
2. **Chunkify**: Convert snapshots or event streams into standardized CHUNK
   units based on time and topology boundaries.
3. **Map Dependencies**: Build the spatial and temporal dependency graph
   between generated CHUNKs.
4. **Dispatch**: Assign CHUNKs to distributed workers based on load balancing
   and communication locality.
5. **Execute and Overlap**: Run the training loop while overlapping network
   prefetching, data routing, and GPU computation asynchronously.
6. **Commit**: Commit node memory or state updates and advance to the next
   CHUNK.

Why This Matters
----------------

For most users, the key takeaway is simple.

First, you still choose **DTDG** or **CTDG** according to your data and model.

Next, the CHUNK abstraction provides a shared internal coordinate system for
time, topology, and dependency management.

Finally, this unified abstraction creates more room for load-aware scheduling,
asynchronous prefetch, and reusable distributed execution logic across graph
modes.

Config Example
--------------

The repository includes a template config at ``configs/chunk_default.yaml``.

.. code-block:: yaml

   data:
     graph_mode: chunk

   chunk:
     time_slices: 100
     node_clusters: 8

This file is best understood as a systems template rather than a single fixed
training recipe. Its purpose is to show how chunk-based execution can be
configured independently from the user-facing DTDG and CTDG guides.
