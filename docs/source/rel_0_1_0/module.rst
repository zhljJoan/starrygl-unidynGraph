Creating Temporal GNN Models
============================

1. Continuous-time Temporal GNN Models
--------------------------------------

To create a continuous-time temporal GNN model, we first need to define a configuration file with the suffix yml to specify the model structures and parameters. Here we use the configuration file :code:`TGN.yml` for TGN model as an example:

.. code-block:: yaml

    sampling:
      - layer: 1
        neighbor: 
          - 10
        strategy: 'recent'
        prop_time: False
        history: 1
        duration: 0
        num_thread: 32
    memory: 
      - type: 'node'
        dim_time: 100
        deliver_to: 'self'
        mail_combine: 'last'
        memory_update: 'gru'
        mailbox_size: 1
        combine_node_feature: True
        dim_out: 100
    gnn:
      - arch: 'transformer_attention'
        use_src_emb: False
        use_dst_emb: False
        layer: 1
        att_head: 2
        dim_time: 100
        dim_out: 100
    train:
      - epoch: 20
        batch_size: 200
        # reorder: 16
        lr: 0.0001
        dropout: 0.2
        att_dropout: 0.2
        all_on_gpu: True

The configuration file is composed of four parts: :code:`sampling`, :code:`memory`, :code:`gnn` and :code:`train`. Here are their meanings:

- :code:`sampling`: This part specifies the sampling strategy for the temporal graph. :code:`layer` field specifies the number of layers in the sampling strategy. The :code:`neighbor` field specifies the number of neighbors to sample for each layer. The :code:`strategy` field specifies the sampling strategy(recent or uniform). The :code:`prop_time` field specifies whether to propagate the time information. The :code:`history` field specifies the number of historical timestamps to use. The :code:`duration` field specifies the duration of the time window. The :code:`num_thread` field specifies the number of threads to use for sampling.
- :code:`memory`: This part specifies the memory module. :code:`type` field specifies the type of memory module(node or none). :code:`dim_time` field specifies the dimension of the time embedding. :code:`deliver_to` field specifies the destination of the message. :code:`mail_combine` field specifies the way to combine the messages. :code:`memory_update` field specifies the way to update the memory. :code:`mailbox_size` field specifies the size of the mailbox. :code:`combine_node_feature` field specifies whether to combine the node features. :code:`dim_out` field specifies the dimension of the output.
- :code:`gnn`: This part specifies the GNN module. :code:`arch` field specifies the architecture of the GNN module. :code:`use_src_emb` field specifies whether to use the source embedding. :code:`use_dst_emb` field specifies whether to use the destination embedding. :code:`layer` field specifies the number of layers in the GNN module. :code:`att_head` field specifies the number of attention heads. :code:`dim_time` field specifies the dimension of the time embedding. :code:`dim_out` field specifies the dimension of the output.
- :code:`train`: This part specifies the training parameters. :code:`epoch` field specifies the number of epochs. :code:`batch_size` field specifies the batch size. :code:`lr` field specifies the learning rate. :code:`dropout` field specifies the dropout rate. :code:`att_dropout` field specifies the attention dropout rate. :code:`all_on_gpu` field specifies whether to put all the data on GPU.

After defining the configuration file, we can firstly read the parameters from the configuration file and create the model by constructing a :code:`General Model` object:

.. code-block:: python

    def parse_config(f):
        conf = yaml.safe_load(open(f, 'r'))
        sample_param = conf['sampling'][0]
        memory_param = conf['memory'][0]
        gnn_param = conf['gnn'][0]
        train_param = conf['train'][0]
        return sample_param, memory_param, gnn_param, train_param
    
    sample_param, memory_param, gnn_param, train_param = parse_config('./config/{}.yml'.format(args.model))
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param).cuda()
    model = DDP(model)

Then a :code:`GeneralModel` object is created. If needed, we can adjust the model's parameters by modifying the contents of the configuration file. Here we provide 5 models for continuous-time temporal GNNs:

- :code:`TGN`: The TGN model proposed in `Temporal Graph Networks for Deep Learning on Dynamic Graphs <https://arxiv.org/abs/2006.10637>`__.
- :code:`DyRep`: The DyRep model proposed in `Representation Learning and Reasoning on Temporal Knowledge Graphs <https://arxiv.org/abs/1803.04051>`__.
- :code:`TIGER`: The TIGER model proposed in `TIGER: A Transformer-Based Framework for Temporal Knowledge Graph Completion <https://arxiv.org/abs/2302.06057>`__.
- :code:`Jodie`: The Jodie model proposed in `JODIE: Joint Optimization of Dynamics and Importance for Online Embedding <https://arxiv.org/abs/1908.01207>`__.
- :code:`TGAT`: The TGAT model proposed in `Temporal Graph Attention for Deep Temporal Modeling <https://arxiv.org/abs/2002.07962>`__.

2. Discrete-time Temporal GNN Models
------------------------------------

To create a discrete-time temporal GNN model, we first need to define a configuration file with the suffix yml to specify the model structures and parameters. Here we use the configuration file :code:`parameters_elliptic_egcn_o.yaml` for egcn_o model as an example:

.. code-block:: yaml

    dataset_args:
        data: elliptic_temporal
        elliptic_args:
          folder: ./data/elliptic_temporal
          tar_file: elliptic_bitcoin_dataset_cont.tar.gz
          feats_file: elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv
          edges_file: elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv
          classes_file: elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv
          times_file: elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv
          aggr_time: 1

    train:
        use_cuda: True
        use_logfile: True
        model: egcn_o
        task: node_cls

        class_weights: [ 0.35, 0.65]
        use_2_hot_node_feats: False
        use_1_hot_node_feats: False
        save_node_embeddings: True

        train_proportion: 0.65
        dev_proportion: 0.1
        num_epochs: 800
        steps_accum_gradients: 1
        learning_rate: 0.001
        learning_rate_min: 0.001
        learning_rate_max: 0.02
        negative_mult_training: 20
        negative_mult_test: 100
        smart_neg_sampling: False
        seed: 1234
        target_measure: F1
        target_class: 1
        early_stop_patience: 100

        eval_after_epochs: 5
        adj_mat_time_window: 1
        adj_mat_time_window_min: 1
        adj_mat_time_window_max: 10
        num_hist_steps: 5 # number of previous steps used for prediction
        num_hist_steps_min: 3 # only used if num_hist_steps: None
        num_hist_steps_max: 10 # only used if num_hist_steps: None
        data_loading_params:
          batch_size: 1
          num_workers: 6

    gcn_parameters:
      feats_per_node: 50
      feats_per_node_min: 30
      feats_per_node_max: 312
      layer_1_feats: 256
      layer_1_feats_min: 30
      layer_1_feats_max: 500
      layer_2_feats: None
      layer_2_feats_same_as_l1: True
      k_top_grcu: 200
      num_layers: 2
      lstm_l1_layers: 125
      lstm_l1_feats: 100
      lstm_l1_feats_min: 50
      lstm_l1_feats_max: 500
      lstm_l2_layers: 1
      lstm_l2_feats: 400
      lstm_l2_feats_same_as_l1: True
      cls_feats: 307
      cls_feats_min: 100
      cls_feats_max: 700

The configuration file is composed of three parts: :code:`dataset_args`, :code:`train` and :code:`gcn_parameters`. Here are their meanings:


- :code:`dataset_args`: This part specifies some configurations for the dataset used. :code:`data` specifies the name of the dataset. :code:`elliptic_args` contains parameters related to the dataset, including the folder location of the dataset, the name of the data file, and so on.
- :code:`train`: This part specifies the training parameters. :code:`use_cuda` indicates whether cuda is used for computation. :code:`use_logfile` indicates whether log files are used to record running processes. :code:`model` indicates the model name to use. :code:`task` indicates the type of the task. :code:`class_weights` is the class weight, which deals with class imbalance. :code:`use_2_hot_node_feats` and :code:`use_2_hot_node_feats` indicates whether one-hot encoding is used. :code:`save_node_embeddings` indicates whether the node embedding is saved. :code:`train_proportion` and :code:`dev_proportion` indicates the ratio of training set and validation set. :code:`num_epochs` indicates the total number of rounds of training. :code:`steps_accum_gradients` indicates the number of steps for gradient accumulation, which is used to implement gradient accumulation. :code:`learning_rate` indicates learning rate. :code:`negative_mult_training` and :code:`negative_mult_test` indicates the multiple of negative sampling at training and test time. :code:`smart_neg_sampling` indicates whether to use negative-only sampling. :code:`seed` indicates the random number seed. :code:`target_measure` and :code:`target_class` denote the target evaluation metric and target category respectively. :code:`early_stop_patience` is the patience value of early stopping, stopping the training if the performance on the validation set does not improve within a certain number of rounds. :code:`eval_after_epochs` indicates how many rounds the evaluation should be performed. :code:`adj_mat_time_window` indicates the time window of the adjacency matrix. :code:`num_hist_steps` indicates the number of historical steps used for prediction. :code:`data_loading_params` indicates data loading parameters, including batch size and number of worker threads.
- :code:`gcn_parameters`: This part specifies the GCN module.These include the number of features per node, the number of features per layer, and the parameters of the LSTM layer. Notice that there are parameters with the suffixes min and max, which means that the parameter values will be randomly generated between min and max based on the random number seed.

After defining the configuration file, we can firstly read the parameters from the configuration file and create a GNN model that supports partition parallelism and use it for later training:

.. code-block:: python

    def create_parser():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--config_file', default='experiments/parameters_elliptic_egcn_o.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')

        return parser

    def parse_args(parser):
        args = parser.parse_args()
        if args.config_file:
            data = yaml.load(args.config_file, Loader=yaml.FullLoader)
            delattr(args, 'config_file')
            arg_dict = args.__dict__
            for key, value in data.items():
                arg_dict[key] = value
        if args.model in ['dysat', 'gcrn']:
            return args

        args.learning_rate = random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
        args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
        args.gcn_parameters['feats_per_node'] = random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
        args.gcn_parameters['layer_1_feats'] = random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
        if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
            args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
        else:
            args.gcn_parameters['layer_2_feats'] = random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
        args.gcn_parameters['lstm_l1_feats'] = random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
        if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
            args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
        else:
            args.gcn_parameters['lstm_l2_feats'] = random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
        args.gcn_parameters['cls_feats'] = random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')

        return args

        parser = u.create_parser()
        args = u.parse_args(parser)

        sync_gnn = build_model(args, graph=g, group=group)

Then a :code:`build_model` object is created. If needed, we can adjust the model's parameters by modifying the contents of the configuration file. Here we provide 4 models for discrete-time temporal GNNs:

- :code:`EloveGCN`: The EGCN model proposed in `Evolving graph convolutional networks for dynamic graphs <https://ojs.aaai.org/index.php/AAAI/article/download/5984/5840>`__.
- :code:`DySAT`: The DySAT model proposed in `Deep neural representation learning on dynamic graphs via self-attention networks <https://sci-hub.yncjkj.com/10.1145/3336191.3371845>`__.
- :code:`GCRN`: The GCRN model proposed in `Structured sequence modeling with graph convolutional recurrent networks <https://arxiv.dosf.top/pdf/1612.07659.pdf>`__.
- :code:`TGCN`: The TGCN model proposed in `Tag graph convolutional network for tag-aware recommendation <https://xinxin-me.github.io/papers/TGCN.pdf>`__.
