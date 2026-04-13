Distributed Partition Parallel
==============================

The **distributed partition parallel** refers to the process of partitioning a large graph into
multiple partitions and distributing them to different workers. Each worker is responsible for training
a specific partition in parallel. The data is exchanged across different partitions.

Starygl provide a very simple way to implement partition parallel with Route class. The Route class can effectively manage
the data exchange between different partitions during training. It plays a crucial role in facilitating communication and synchronization
between workers responsible for training specific graph partitions. The Route class provides a mechanism for efficiently routing and exchanging data,
enabling seamless parallel training across distributed systems

Here we provide an example to show how to use Route to implement partition parallel.
Before you start with partition parallel. First you should decide how to partition your graph data. Starygl provide serveral
partition algorithms

- lgd
- metis
- multi-constraint metis

In the following code, we first partition graph with specific algorithm.Then we save the node and edge feature with the
correspond graph partition together.

.. code-block:: python

    def partition_graph(self,
        root: str,
        num_parts: int,
        node_weight: Optional[str] = None,
        edge_weight: Optional[str] = None,
        algorithm: str = "metis",
        partition_kwargs = None,):

        assert not self.is_heterogeneous, "only support homomorphic graph"

        num_nodes: int = self.node().num_nodes
        edge_index: Tensor = self.edge_index()

        logging.info(f"running partition aglorithm: {algorithm}")
        partition_kwargs = partition_kwargs or {}

        not_self_loop = (edge_index[0] != edge_index[1])

        if node_weight is not None:
            node_weight = self.node()[node_weight]

        if edge_weight is not None:
            edge_weight = self.edge()[edge_weight]
            edge_weight = edge_weight[not_self_loop]

        # partition graph
        node_parts = metis_partition(
            edge_index[:,not_self_loop],
            num_nodes, num_parts,
            node_weight=node_weight,
            edge_weight=edge_weight,
            **partition_kwargs,


        root_path = Path(root).expanduser().resolve()
        base_path = root_path / f"{algorithm}_{num_parts}"

        # handle each partition
        for i in range(num_parts):
            npart_mask = node_parts == i
            epart_mask = npart_mask[edge_index[1]]

            raw_dst_ids: Tensor = torch.where(npart_mask)[0]
            local_edges = edge_index[:, epart_mask]

            raw_src_ids, local_edges = init_vc_edge_index(
                raw_dst_ids, local_edges, bipartite=True,
            )

            # get GraphData obj
            g = GraphData.from_bipartite(
                local_edges,
                raw_src_ids=raw_src_ids,
                raw_dst_ids=raw_dst_ids,
            )

            # handle feature data
            # ......

            logging.info(f"saving partition data: {i+1}/{num_parts}")
            # save each partition
            torch.save(g, (base_path / f"{i:03d}").__str__())

Next we will deal with our model.With Route, developers just need to change few lines of code to implement partition parallel
In the example below, we just need to add one line code in the forward function. And the Route will help us manage the feature exchange.

.. code-block:: python

    class SimpleConv(pyg_nn.MessagePassing):
        def __init__(self, in_feats: int, out_feats: int):
            super().__init__(aggr="mean")
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, x: Tensor, edge_index: Tensor, route: Route):
            dst_len = x.size(0)
            x = route.apply(x) # exchange features
            return self.propagate(edge_index, x=x)[:dst_len]

        def message(self, x_j: Tensor):
            return x_j

        def update(self, x: Tensor):
            return F.relu(self.linear(x))

