Route Layer Reference
=====================

The route layer abstracts distributed feature exchange across partitions
in multi-GPU/multi-machine training. It decouples batch materialization
(how we gather data) from physical data movement (how we move it in a cluster).

Overview
--------

Routes handle the communication pattern for gathering temporal neighbors:

- In **DTDG** (snapshots): Routes specify which nodes need features from remote GPUs
  per snapshot, enabling all-to-all collectives
- In **CTDG** (events): Routing is determined by node ownership metadata and
  executed through runtime exchange helpers for distributed memory / feature sync
- In **Chunk**: There may be extra cross-partition forwarding logic, but the exact
  interface is backend-dependent and is not stable enough to treat as a core route API

Both DTDG and CTDG can overlap communication with computation. In DTDG this is
implemented through the route runtime's asynchronous send/recv path and is
typically completed with ``await`` at the window level. In CTDG, distributed
memory / mailbox synchronization can likewise be submitted asynchronously and
hidden behind later compute.

Two-Level Route Abstraction
---------------------------

At the conceptual level, the route layer can be viewed as two linked stages:

- a requirement stage that determines what remote data the current batch needs
- an execution stage that packs local outputs and issues the actual communication

The current persisted ``RouteData`` artifact corresponds to the DTDG execution
stage. CTDG uses runtime route helpers rather than a persisted route artifact.

``RouteData`` — Execution Exchange Metadata
--------------------------------------------

The current dataclass used by DTDG artifacts describes the metadata needed for
explicit all-to-all feature exchange:

.. code-block:: python

    @dataclass
    class RouteData:
        """Execution-facing distributed feature exchange metadata."""

        # How many items are exchanged with each peer for every snapshot
        send_sizes: List[List[int]]
        recv_sizes: List[List[int]]

        # Packed local indices used to gather send buffers
        send_index_ind: Tensor | None
        send_index_ptr: List[int] | None

Construction:

.. code-block:: python

    routes = RouteData(
        send_sizes=[[2, 2], [1, 3]],
        recv_sizes=[[3, 1], [0, 4]],
        send_index_ind=torch.tensor([8, 3, 5, 1, 2, 7, 6, 4]),
        send_index_ptr=[0, 4, 8],
    )

For one concrete snapshot, the runtime slices ``send_index_ind`` with
``send_index_ptr`` to recover that snapshot's ``send_index`` and then gathers
the local GNN outputs in the exact order required by ``send_sizes``.

This means the stored artifact is already the **post-materialization execution
plan**, not the earlier sampling-time requirement description.

Mode-Specific Usage
-------------------

**DTDG: Per-Snapshot Routes**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTDG creates one execution ``Route`` per snapshot, describing how features move:

.. code-block:: python

    class Route:
        """One snapshot's feature routing (DTDG)."""

        @property
        def send_index(self) -> Tensor | None:
            """Indices of local outputs packed into the all-to-all send buffer."""
            ...

        def forward(self, features: Tensor) -> Tensor:
            """Execute all-to-all exchange. [N_local, F] → [N_fetched, F]"""
            ...

        async def async_forward(self, features: Tensor) -> Tensor:
            """Async route exchange used to overlap communication with compute."""
            ...

Integration with DGL blocks:

.. code-block:: python

    block = dgl.create_block(...)
    block.route = route_for_this_snapshot

    # Inside GCN layer
    def _gcn_message_pass(graph, x):
        if graph.is_block and hasattr(graph, 'route'):
            src_x = graph.route.forward(x)  # Fetch remote features
        else:
            src_x = x  # Local only
        # ... continue message passing with src_x

The DTDG runtime also supports an asynchronous path. A typical pattern is to
submit route communication for snapshots in a window, continue local work, and
then ``await`` completion when the window needs the exchanged features.

DTDG's all-to-all uses ``torch.distributed`` collectives directly:

.. code-block:: python

    def route_forward(features: Tensor, send_index: Tensor):
        """All-to-all single for distributed training.

        send_index selects and orders local outputs for the send buffer.
        Returns: stacked features received from peer ranks.
        """
        # Typical usage:
        # input_buffer = features[send_index]
        # send_count then splits input_buffer by destination rank

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        input_buffer = features[send_index]
        dist.all_to_all_single(
            output=output_buffer,  # Pre-allocated for recv_count total features
            input=input_buffer,    # Packed local outputs in all-to-all order
            output_split_sizes=recv_count,
            input_split_sizes=send_count,
        )
        return output_buffer

**CTDG: Ownership-Based Runtime Exchange**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CTDG does have a concrete route implementation in the current codebase, but it
is runtime-only rather than a persisted ``RouteData`` artifact.

Feature exchange is handled by ``CTDGFeatureRoute``:

.. code-block:: python

    @dataclass
    class CTDGFeatureRoute:
        route_type: str
        world_size: int
        replicated_memory: bool = True

        def exchange(
            self,
            ctx: DistributedContext,
            node_ids: Tensor,
            values: Tensor,
            async_op: bool = False,
        ) -> tuple[Tensor, Tensor] | AsyncExchangeHandle:
            ...

The route exchanges ``(node_id, feature_vector)`` pairs across ranks. In the
fast path it packs each record as ``[node_id | values...]`` and uses
``dist.all_to_all_single``. When ``async_op=True`` it returns an
``AsyncExchangeHandle`` whose ``wait()`` method reconstructs the merged
``(ids, values)`` tensors.

Memory and mailbox synchronization are handled in
``backends/ctdg/runtime/memory.py`` using the same ownership-based routing
idea:

.. code-block:: python

    def submit_async_memory_sync(ctx, node_ids, values, timestamps) -> None:
        # 1. determine owner rank from node_parts or modulo fallback
        # 2. keep only remote-owned updates
        # 3. pack by owner and exchange ids + payload with all_to_all_single
        # 4. optionally defer completion until wait_pending_syncs()
        ...

    def submit_async_mail_sync(ctx, node_ids, mail_slots, mail_ts) -> None:
        ...

So the CTDG route layer is real and stable, but its unit of abstraction is
runtime exchange by node ownership, not per-snapshot graph-route artifacts.

**Chunk: Backend-Specific Forwarding Interface**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chunk-related forwarding exists in some pipelines, but the concrete interface is
not yet stable. It is therefore documented only as a backend-specific extension,
not as part of the stable route-layer contract.

.. code-block:: python

    class ChunkRoute(Protocol):
        """Possible forwarding interface for chunk-style pipelines."""

        def request_remote(self, cluster_id: int, payload: dict) -> None:
            ...

        def flush(self) -> None:
            ...

See Also
--------

- :doc:`data_layer` — Data access abstractions related to routing
- :doc:`artifact_format` — Serialization of routes to disk
- :doc:`unified_pipeline` — How backends integrate routes into training
- Source: ``backends/dtdg/runtime/route.py``, ``backends/ctdg/runtime/route.py``,
  ``backends/ctdg/runtime/memory.py``
