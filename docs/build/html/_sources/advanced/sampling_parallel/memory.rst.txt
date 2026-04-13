Distributed Memory Updater
==========================

Introduction
------------
In this tutorial, we will explore the concept of a distributed memory updater in the context of StarryGL. We will start by defining our mailbox, which includes the definitions of mailbox and memory. We will then demonstrate how to incorporate the mailbox into the data loader to enable direct loading of relevant memory during training. Finally, we will discuss the process of updating the relevant storage using the `get_update_memory` and `get_update_mail` functions.

Defining the Mailbox
--------------------
To begin, let's define our mailbox, which is an essential component for the distributed memory updater. We will use the `SharedMailBox` class from the `starrygl.sample.memory.shared_mailbox` module.

Here is an example of how to define the mailbox:

   .. code-block:: python
    
        from starrygl.sample.memory.shared_mailbox import SharedMailBox

        # Define the mailbox
        mailbox = SharedMailBox(num_nodes=num_nodes, memory_param=memory_param, dim_edge_feat=dim_edge_feat)

In the code snippet above, we import the `SharedMailBox` class and initialize it with the following parameters:
- `num_nodes`: The number of nodes in the graph.

- `memory_param`: The memory parameters specified in the YAML file, which are relevant to the Temporal Graph Neural Network (TGN) framework.

- `dim_edge_feat`: The dimension of the edge feature.

Incorporating the Mailbox into the Data Loader
----------------------------------------------
After defining the mailbox, we need to pass it to the data loader so that the relevant memory/mailbox can be directly loaded during training. This ensures efficient access to the required memory for updating.

Here is an example of how to incorporate the mailbox into the data loader:

    .. code-block:: python
        
        from starrygl.sample.part_utils.partition_tgnn import partition_load
        from starrygl.sample.memory.shared_mailbox import SharedMailBox

        # Load the partitioned data
        pdata = partition_load("PATH/{}".format(dataname), algo="metis_for_tgnn")

        # Initialize the mailbox with the required parameters
        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat=pdata.edge_attr.shape[1] if pdata. edge_attr is not None else 0)

In the code snippet above, we import the necessary modules and load the partitioned data using the `partition_load` function. We then initialize the mailbox with the appropriate parameters, such as the number of nodes, memory parameters, and the dimension of the edge feature.

Updating the Relevant Storage
-----------------------------
During the training process, it is important to constantly update the relevant storage to ensure accurate and up-to-date information. In StarryGL, this is achieved by calling the `get_update_memory` and `get_update_mail` functions.

These functions implement the idea related to the Temporal Graph Neural Network (TGN) framework, where the relevant storage is updated based on the current state of the graph.

Conclusion
----------
In this tutorial, we explored the concept of a distributed memory updater in StarryGL. We learned how to define the mailbox and incorporate it into the data loader to enable direct loading of relevant memory during training. We also discussed the process of updating the relevant storage using the `get_update_memory` and `get_update_mail` functions.

By utilizing the distributed memory updater, you can efficiently update and access the required memory during training, which is crucial for achieving accurate and effective results in graph-based models.

We hope this tutorial provides a clear understanding of the distributed memory updater in StarryGL. If you have any further questions or need additional assistance, please don't hesitate to ask.

Note: If you find this tutorial helpful, a generous tip would be greatly appreciated.