Distributed Feature Fetching
============================

Introduction
------------
In this tutorial, we will explore how to perform feature fetching in the data loader using StarryGL. StarryGL provides convenient methods for fetching node or edge features during the data loading process. We will demonstrate how to define a data loader and utilize StarryGL's features to fetch the required features.

Defining the Data Loader
------------------------
To use feature fetching in the data loader, we need to define a data loader and configure it with the necessary parameters. We can use the `DistributedDataLoader` class from the `starrygl.sample.data_loader` module. 

Here is an example of how to define a data loader for feature fetching:

    .. code-block:: python  

        from starrygl.sample.data_loader import DistributedDataLoader

        # Define the data loader
        trainloader = DistributedDataLoader(graph, data, sampler=sampler, sampler_fn=sampler_fn, 
                                   neg_sampler=neg_sampler, batch_size=batch_size, mailbox=mailbox)

In the code snippet above, we import the `DistributedDataLoader` class and initialize it with the following parameters:
- `graph`: The distributed graph store.

- `data`: The graph data.

- `sampler`: A parallel sampler, such as the `NeighborSampler`.

- `sampler_fn`: The sample type.

- `neg_sampler`: The negative sampler.

- `batch_size`: The batch size.

- `mailbox`: The mailbox used for communication and memory sharing.

Examples:

    .. code-block:: python

        import torch

        from starrygl.sample.data_loader import DistributedDataLoader
        from starrygl.sample.part_utils.partition_tgnn import partition_load
        from starrygl.sample.graph_core import DataSet, DistributedGraphStore, TemporalNeighborSampleGraph
        from starrygl.sample.memory.shared_mailbox import SharedMailBox
        from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler
        from starrygl.sample.sample_core.base import NegativeSampling
        from starrygl.sample.batch_data import SAMPLE_TYPE

        pdata = partition_load("PATH/{}".format(dataname), algo="metis_for_tgnn")    
        graph = DistributedGraphStore(pdata = pdata, uvm_edge = False, uvm_node = False)
        sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat=pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
        sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=1, fanout=[10], graph_data=sample_graph, workers=15,policy = 'recent',graph_name = "wiki_train")
        neg_sampler = NegativeSampling('triplet')
        train_data = torch.masked_select(graph.edge_index, pdata.train_mask.to(graph.edge_index.device)).reshape(2, -1)
        trainloader = DistributedDataLoader(graph, train_data, sampler=sampler, sampler_fn=SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler, batch_size=1000, shuffle=False, drop_last=True, chunk_size = None,
                                            train=True, queue_size=1000, mailbox=mailbox )

In the data loader, we will call the `graph_sample`, sourced from `starrygl.sample.batch_data`.

And the `to_block` function in the `graph_sample` will implement feature fetching.
If cache is not used, we will directly fetch node or edge features from the graph data, 
otherwise we will call `starrgl.sample.cache.FetchFeatureCache` for feature fetching.
