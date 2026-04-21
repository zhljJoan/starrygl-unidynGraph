Reusable Model Modules
======================

The ``starry_unigraph.runtime.modules`` package contains stable, backend-agnostic
neural network components that can be reused across CTDG, DTDG, and Chunk backends.

Modules are extracted from backend implementations to avoid duplication and
enable composable model design.

Overview
--------

Reusable modules:

- **TimeEncode** — Learnable time encoding (all temporal modes)
- **GCNStack** — Multi-layer GCN message passing (DTDG, Chunk)
- **MatGRUCell** — Weight-evolving GRU (DTDG EvolveGCN)
- **_LSTMCell** — Custom LSTM cell (DTDG MPNN-LSTM)

Backend-specific (not reusable):

- **TemporalTransformerConv** — CTDG multi-head attention
- **CTDGMemoryUpdater** — CTDG GRU memory bank
- **CTDGLinkPredictor** — CTDG full link prediction
- See ``backends/ctdg/runtime/models.py`` and ``backends/dtdg/runtime/models.py``

TimeEncode
----------

Learnable time encoding using cosine basis functions.

**Location**: ``starry_unigraph.runtime.modules.time_encode``

**Import**:

.. code-block:: python

    from starry_unigraph.runtime.modules import TimeEncode

**API**:

.. code-block:: python

    class TimeEncode(nn.Module):
        def __init__(self, dim: int) -> None:
            """
            Args:
                dim: Output embedding dimension (64-128 typical)
            """

        def forward(self, t: Tensor) -> Tensor:
            """
            Args:
                t: Time values, any shape

            Returns:
                Time embeddings [*original_shape, dim]
            """

**Example**:

.. code-block:: python

    import torch
    from starry_unigraph.runtime.modules import TimeEncode

    # Create encoder
    time_enc = TimeEncode(dim=100)

    # Encode scalar times
    t = torch.tensor([0.5, 1.2, 2.0])            # [3]
    h_t = time_enc(t)                             # [3, 100]

    # Use in attention
    batch_size = 32
    dt = torch.randn(batch_size)
    time_feat = time_enc(dt)                      # [32, 100]
    query = torch.cat([node_feat, time_feat], dim=1)

**Mathematical Form**:

.. math::

    \text{TimeEncode}(t) = \cos(W \cdot t)

where :math:`W = \text{diag}(1/10^{[0,1,...,\text{dim}-1]})`.

This captures multiple frequency scales for flexible time representation.

GCNStack
--------

Multi-layer GCN message passing on DGL graphs.

**Location**: ``starry_unigraph.runtime.modules.gcn_layers``

**Import**:

.. code-block:: python

    from starry_unigraph.runtime.modules import GCNStack

**API**:

.. code-block:: python

    class GCNStack(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 2
        ) -> None:
            """
            Args:
                input_size: Node feature dimension
                hidden_size: Hidden and output dimension for all layers
                num_layers: Number of GCN layers (default 2)
            """

        def forward_graph(self, graph: DGLGraph, x: Tensor | None = None) -> Tensor:
            """Apply GCN to single graph.

            Returns:
                Node embeddings [N, hidden_size]
            """

        def layerwise(self, blob_or_graph) -> List[Tensor]:
            """Apply GCN to sequence of graphs (STGraphBlob).

            Returns:
                List of node embeddings, one per graph
            """

**Example**:

.. code-block:: python

    import torch
    import dgl
    from starry_unigraph.runtime.modules import GCNStack

    # Create GCN stack
    gcn = GCNStack(input_size=64, hidden_size=128, num_layers=2)

    # Create sample graph
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    graph = dgl.graph((edges[0], edges[1]))
    graph.ndata['x'] = torch.randn(3, 64)

    # Forward pass
    h = gcn.forward_graph(graph)                  # [3, 128]

    # Multi-frame (temporal sequence)
    graphs = [graph for _ in range(5)]
    h_list = gcn.layerwise(graphs)                # List of 5 [3, 128] tensors

**Integration with Flare**:

.. code-block:: python

    # Inside FlareTGCN
    class FlareTGCN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.gcn = GCNStack(input_size, hidden_size * 3, num_layers=2)
            # ...

        def forward(self, blob_or_graph, state=None):
            # GCN per snapshot
            outputs = self.gcn.layerwise(blob_or_graph)
            # ... apply GRU to process outputs

**Message Passing Details**:

GCN layer applies: :math:`h' = \sigma(W \cdot (\text{GCN\_norm} \odot h))` per message pass.

- Handles both full graphs and DGL bipartite blocks
- Supports distributed training via route() if attached to block
- ReLU activation between layers

MatGRUCell
----------

GRU cell for evolving weight matrices (EvolveGCN).

**Location**: ``starry_unigraph.runtime.modules.rnn_cells``

**Import**:

.. code-block:: python

    from starry_unigraph.runtime.modules import MatGRUCell

**API**:

.. code-block:: python

    class MatGRUCell(nn.Module):
        def __init__(self, in_feats: int, out_feats: int) -> None:
            """
            Args:
                in_feats: Context feature dimension
                out_feats: Output (matrix) dimension
            """

        def forward(self, prev_w: Tensor, inputs: Tensor) -> Tensor:
            """Evolve weight matrix.

            Args:
                prev_w: Previous weight [out_feats, ...]
                inputs: Context input [in_feats]

            Returns:
                New weight [out_feats, ...]
            """

**Example**:

.. code-block:: python

    import torch
    from starry_unigraph.runtime.modules import MatGRUCell

    # Create cell
    mat_gru = MatGRUCell(in_feats=64, out_feats=256)

    # Initialize weight matrix
    w = torch.randn(256, 256)

    # Get context (e.g., graph pooling)
    ctx = torch.randn(64)

    # Evolve weight
    w_new = mat_gru(w, ctx)                       # [256, 256]

**Integration with EvolveGCN**:

.. code-block:: python

    # Inside FlareEvolveGCN
    class FlareEvolveGCN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.mat_gru = MatGRUCell(in_feats=input_size, out_feats=hidden_size)
            # ...

        def forward(self, blob_or_graph, state=None):
            weights = self.initial_weight if state is None else state
            for graph in _graph_sequence(blob_or_graph):
                ctx = self._pool_graph_context(graph)
                weights = self.mat_gru(weights, ctx)
                # ... use weights in GCN

_LSTMCell
---------

Custom LSTM cell with tuple state support.

**Location**: ``starry_unigraph.runtime.modules.rnn_cells``

**Import**:

.. code-block:: python

    from starry_unigraph.runtime.modules import _LSTMCell

**API**:

.. code-block:: python

    class _LSTMCell(nn.Module):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            """
            Args:
                input_size: Input feature dimension
                hidden_size: Hidden state dimension
            """

        def forward(
            self,
            x: Tensor,
            state: tuple[Tensor, Tensor] | None = None
        ) -> tuple[Tensor, Tensor]:
            """LSTM cell forward.

            Args:
                x: Input [batch, input_size]
                state: Tuple (h, c) or None

            Returns:
                Tuple (h_new, c_new)
            """

**Example**:

.. code-block:: python

    import torch
    from starry_unigraph.runtime.modules import _LSTMCell

    # Create LSTM cell
    lstm_cell = _LSTMCell(input_size=64, hidden_size=128)

    # Process sequence
    seq_len = 10
    batch_size = 32
    state = None

    for t in range(seq_len):
        x_t = torch.randn(batch_size, 64)
        h_t, c_t = lstm_cell(x_t, state)
        state = (h_t, c_t)

    # Multi-layer stacking (tuple concatenation)
    lstm1 = _LSTMCell(64, 128)
    lstm2 = _LSTMCell(128, 128)
    state = None

    for t in range(seq_len):
        x_t = torch.randn(batch_size, 64)

        # Layer 1
        s1 = None if state is None else state[0:2]
        x_t, _ = s1 = lstm1(x_t, s1)

        # Layer 2
        s2 = None if state is None else state[2:4]
        x_t, _ = s2 = lstm2(x_t, s2)

        # Combine states via tuple concatenation
        state = s1 + s2                           # (h1, c1, h2, c2)

**Integration with MPNN-LSTM**:

.. code-block:: python

    # Inside FlareMPNNLSTM
    class FlareMPNNLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.gcn = GCNStack(input_size, hidden_size, num_layers=2)
            self.rnn = nn.ModuleList([
                _LSTMCell(hidden_size, hidden_size),
                _LSTMCell(hidden_size, hidden_size),
            ])
            # ...

        def forward(self, blob_or_graph, state=None):
            h = state
            for graph, x in zip(_graph_sequence(blob_or_graph), self.gcn.layerwise(blob_or_graph)):
                s1 = None if h is None else h[0:2]
                s2 = None if h is None else h[2:4]

                x, _ = s1 = self.rnn[0](x, s1)
                x, _ = s2 = self.rnn[1](x, s2)

                h = s1 + s2                       # Combined state

Using Modules in Custom Models
-------------------------------

Build task-specific heads:

.. code-block:: python

    from starry_unigraph.runtime.modules import TimeEncode, GCNStack
    from starry_unigraph.models import EdgePredictHead, WrappedModel

    # Backbone combining reusable modules
    class MyTemporalGCN(nn.Module):
        def __init__(self, input_size, hidden_size, time_dim):
            super().__init__()
            self.time_enc = TimeEncode(time_dim)
            self.gcn = GCNStack(input_size + time_dim, hidden_size, num_layers=2)

        def forward(self, batch, state=None):
            # Encode times
            time_feat = self.time_enc(batch.timestamps)

            # Augment node features
            x = torch.cat([batch.node_feat, time_feat], dim=1)

            # Apply GCN
            h = self.gcn.forward_graph(batch.graph, x)
            return h

    # Wrap with task head
    backbone = MyTemporalGCN(input_size=64, hidden_size=128, time_dim=100)
    head = EdgePredictHead(hidden_size=128)
    model = WrappedModel(backbone, head)

Stability & Versioning
----------------------

All modules in ``runtime.modules`` are marked as **Stable API - v0.1.0+**.

This means:

- API won't break in patch versions (0.1.0 → 0.1.1)
- Behavior is tested and documented
- Safe for external use

See Also
--------

- :doc:`protocols` — Model protocols (TemporalModel)
- :doc:`data_layer` — Data structures (BatchData, RouteData)
- Source: ``starry_unigraph/runtime/modules/``
- Consumers: ``backends/ctdg/runtime/models.py``, ``backends/dtdg/runtime/models.py``
