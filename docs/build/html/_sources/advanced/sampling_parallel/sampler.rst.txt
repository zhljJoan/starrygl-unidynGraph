Distributed Temporal Sampling
=============================

In this tutorial, we will explore the concept of parallel sampling in the context of large-scale graph data. We'll discuss the benefits of parallel sampling, the hybrid CPU-GPU approach we adopt, and how to use the provided functions for parallel sampling.

Introduction
------------
Parallel sampling plays a crucial role in training models on large amounts of data. Traditional serial sampling methods can be inefficient and waste computing and storage resources when dealing with complex graph data. Parallel sampling, on the other hand, improves efficiency and overall computational speed by simultaneously sampling from multiple nodes or neighbors. This approach accelerates the training and inference process of the model, making it more scalable and practical for large-scale graph data.

Hybrid CPU-GPU Approach
-----------------------
Our parallel sampling approach combines the power of both CPUs and GPUs. The entire graph structure is stored on the CPU, and the graph structure is sampled on the CPU before being uploaded to the GPU. Each trainer has a separate sampler for parallel training, ensuring efficient utilization of computing resources.

Using the Parallel Sampler
--------------------------
To easily use the parallel sampler, follow these steps:

1. Import the required Python packages::

        from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler

2. Initialize the parallel sampler with the desired parameters::

        sampler = NeighborSampler(num_nodes=num_nodes, num_layers=num_layers, fanout=fanout, graph_data=graph_data,
                             workers=workers, is_distinct=is_distinct, policy=policy, edge_weight=edge_weight,
                             graph_name=graph_name)

   In the code snippet above, we import the ``NeighborSampler`` class from the ``starrygl.sample.sample_core`` module. We then create an instance of the ``NeighborSampler`` class, providing the necessary parameters such as the number of nodes, the number of layers to be sampled, the fanout (the maximum number of neighbors chosen for each layer), the graph data to be sampled, the number of workers (threads), the distinct multi-edge flag, the sampling policy, the initial weights of edges, and the graph name.

3. Perform the parallel sampling::

        # Perform parallel sampling
        sampler.sample()

   After initializing the sampler, you can call the ``sample()`` method to perform the parallel sampling. This method internally handles the sampling process, leveraging the hybrid CPU-GPU approach. The sampled data can then be used for further training or analysis.

Directly Calling Parallel Sampling Functions
--------------------------------------------
If you prefer to directly call the parallel sampling functions, you can use the following methods:

1. Import the required Python package::

   from starrygl.lib.libstarrygl_sampler import ParallelSampler, get_neighbors

2. Retrieve neighbor information and create a neighbor information table::


        # Get neighbor information table
        tnb = get_neighbors(graph_name, row.contiguous(), col.contiguous(), num_nodes, is_distinct, graph_data.eid,
                       edge_weight, timestamp)

   The ``get_neighbors`` function retrieves the neighbor information table based on the provided parameters, such as the graph name, the row and column indices (from ``graph_data.edge_index``), the number of nodes, the distinct multi-edge flag, the edge IDs, the edge weights, and the timestamp.

3. Call the parallel sampler::


        # Call parallel sampler
        p_sampler = ParallelSampler(tnb, num_nodes, graph_data.num_edges, workers, fanout, num_layers, policy)

   The ``ParallelSampler`` class is used to perform the parallel sampling. It takes the neighbor information table (``tnb``) and other parameters, such as the number of nodes, the number of edges, the number of workers, the fanout, the number of layers, and the sampling policy.

Additional Resources
--------------------
For complete usage details and more information, please refer to the ``starrygl.sample.sample_core.neighbor_sampler`` module.

I hope this tutorial provides a comprehensive understanding of distributed temporal sampling and how to use the provided functions for parallel sampling. If you have any further questions or need additional assistance, please don't hesitate to ask.
