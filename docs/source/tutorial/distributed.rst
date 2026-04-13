Distributed Training
====================

1. Preparation distributed environment for CTDG
-----------------------------------------------

Before start training, we need to prepare the environment for distributed training, inluding the following steps:

1. Initialize the Distributed context

    .. code-block:: python

        ctx = DistributedContext.init(backend="nccl", use_gpu=True) 

2. Load the partitioned dataset

    .. code-block:: python

        pdata = partition_load("/mnt/data/part_data/dataset/here/{}".format(args.dataname), algo="metis_for_tgnn")    
        graph = DistributedGraphStore(pdata = pdata)
        sample_graph = TemporalNeighborSampleGraph(sample_graph = pdata.sample_graph,mode = 'full')
        train_data = torch.masked_select(graph.edge_index,pdata.train_mask.to(graph.edge_index.device)).reshape(2,-1)
        train_ts = torch.masked_select(graph.edge_ts,pdata.train_mask.to(graph.edge_index.device))
        val_data = torch.masked_select(graph.edge_index,pdata.val_mask.to(graph.edge_index.device)).reshape(2,-1)
        val_ts = torch.masked_select(graph.edge_ts,pdata.val_mask.to(graph.edge_index.device))
        test_data = torch.masked_select(graph.edge_index,pdata.test_mask.to(graph.edge_index.device)).reshape(2,-1)
        test_ts = torch.masked_select(graph.edge_ts,pdata.test_mask.to(graph.edge_index.device)) 
        train_data = DataSet(edges = train_data,ts =train_ts,eids = torch.nonzero(pdata.train_mask).view(-1))
        test_data = DataSet(edges = test_data,ts =test_ts,eids = torch.nonzero(pdata.test_mask).view(-1))
        val_data = DataSet(edges = val_data,ts = val_ts,eids = torch.nonzero(pdata.val_mask).view(-1))
        train_stream = torch.cuda.Stream()
        send_stream = torch.cuda.Stream()
        scatter_stream = torch.cuda.Stream()

3. Construct Mailbox and sampler

    .. code-block:: python

        mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat = pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)
        sampler = NeighborSampler(num_nodes=graph.num_nodes, num_layers=num_layers, fanout=fanout,graph_data=sample_graph, workers=10,policy = policy, graph_name = "wiki_train")
        neg_sampler = NegativeSampling('triplet')

4. Construct the DataLoader

    .. code-block:: python

        trainloader = DistributedDataLoader(graph,train_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=True,
                                            chunk_size = None,
                                            train=True,
                                            queue_size = 1000,
                                            mailbox = mailbox)
        testloader = DistributedDataLoader(graph,test_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=False,
                                            chunk_size = None,
                                            train=False,
                                            queue_size = 100,
                                            mailbox = mailbox)
        valloader = DistributedDataLoader(graph,val_data,sampler = sampler,
                                            sampler_fn = SAMPLE_TYPE.SAMPLE_FROM_TEMPORAL_EDGES,
                                            neg_sampler=neg_sampler,
                                            batch_size = train_param['batch_size'],
                                            shuffle=False,
                                            drop_last=False,
                                            chunk_size = None,
                                            train=False,
                                            queue_size = 100,
                                            mailbox = mailbox)

5. `Create the Model <module.rst>`_

6. Construct the optimizer, early stopper and creterion

    .. code-block:: python

        creterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
        early_stopper = EarlyStopMonitor(max_round=args.patience)

7. Start Training

    .. code-block:: python

         for e in range(train_param['epoch']):
            torch.cuda.synchronize()
            model.train()
            if mailbox is not None:
                mailbox.reset()
                model.module.memory_updater.last_updated_nid = None
                model.module.memory_updater.last_updated_memory = None
                model.module.memory_updater.last_updated_ts = None
            for roots,mfgs,metadata,sample_time in trainloader:
                with torch.cuda.stream(train_stream):
                    optimizer.zero_grad()
                    pred_pos, pred_neg = model(mfgs,metadata)
                    loss = creterion(pred_pos, torch.ones_like(pred_pos))
                    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                    total_loss += float(loss)
                    loss.backward()
                    optimizer.step()
                    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                    train_aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
                    if mailbox is not None:
                        src = metadata['src_pos_index']
                        dst = metadata['dst_pos_index']
                        ts = roots.ts
                        if graph.edge_attr is None:
                            edge_feats = None
                        elif(graph.edge_attr.device == torch.device('cpu')):
                            edge_feats = graph.edge_attr[roots.eids.to('cpu')].to('cuda')
                        else:
                            edge_feats = graph.edge_attr[roots.eids] 
                        dist_index_mapper = mfgs[0][0].srcdata['ID']
                        root_index = torch.cat((src,dst))
                        last_updated_nid = model.module.memory_updater.last_updated_nid[root_index]
                        last_updated_memory = model.module.memory_updater.last_updated_memory[root_index]
                        last_updated_ts=model.module.memory_updater.last_updated_ts[root_index]
                        index, memory, memory_ts = mailbox.get_update_memory(last_updated_nid,
                                                                        last_updated_memory,
                                                                        last_updated_ts)
                        index, mail, mail_ts = mailbox.get_update_mail(dist_index_mapper,
                                                    src,dst,ts,edge_feats,
                                                    model.module.memory_updater.last_updated_memory, 
                                                    model.module.embedding,use_src_emb,use_dst_emb,
                                                    )
                        mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')
            train_ap = float(torch.tensor(train_aps).mean())    
            ap, auc = eval('val')
            print('\ttrain loss:{:.4f}  train ap:{:4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss,train_ap, ap, auc))

8. Deifine the Evaluation function
   
   .. code-block:: python

        def eval(mode='val'):
            model.eval()
            aps = list()
            aucs_mrrs = list()
            if mode == 'val':
                loader = valloader
            elif mode == 'test':
                loader = testloader
            elif mode == 'train':
                loader = trainloader
            with torch.no_grad():
                total_loss = 0
                for roots,mfgs,metadata,sample_time in loader:
                    
                    pred_pos, pred_neg = model(mfgs,metadata)
                    total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                    total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                    aps.append(average_precision_score(y_true, y_pred.detach().numpy()))
                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                    if mailbox is not None:
                        src = metadata['src_pos_index']
                        dst = metadata['dst_pos_index']
                        
                        ts = roots.ts
                        if graph.edge_attr is None:
                            edge_feats = None
                        elif(graph.edge_attr.device == torch.device('cpu')):
                            edge_feats = graph.edge_attr[roots.eids.to('cpu')].to('cuda')
                        else:
                            edge_feats = graph.edge_attr[roots.eids] 
                        dist_index_mapper = mfgs[0][0].srcdata['ID']
                        root_index = torch.cat((src,dst))
                        last_updated_nid = model.module.memory_updater.last_updated_nid[root_index]
                        last_updated_memory = model.module.memory_updater.last_updated_memory[root_index]
                        last_updated_ts=model.module.memory_updater.last_updated_ts[root_index]
                        index, memory, memory_ts = mailbox.get_update_memory(last_updated_nid,
                                                                        last_updated_memory,
                                                                        last_updated_ts)
                        
                        index, mail, mail_ts = mailbox.get_update_mail(dist_index_mapper,
                                                    src,dst,ts,edge_feats,
                                                    model.module.memory_updater.last_updated_memory,
                                                    model.module.embedding,use_src_emb,use_dst_emb,
                                                    )
                        mailbox.set_mailbox_all_to_all(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max')

9. Start Evaluation

    .. code-block::python

        if mailbox is not None:
            mailbox.reset()
            model.module.memory_updater.last_updated_nid = None
            print("Train eval:", eval('train'))
            print("Val eval:", eval('test'))
        ap, auc = eval('val')
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))

2. Preparation distributed environment for DTDG
-----------------------------------------------

Before start training, we need to prepare the environment for distributed training, including the following steps:

1. Initialize the Distributed context

    .. code-block:: python

        ctx = DistributedContext.init(backend="nccl", use_gpu=True)
        group = ctx.get_default_group()

2. Import the partitioned dataset using the wrapped function, and let the main process (ctx.rank=0) do the data preparation

    .. code-block:: python

        data_root = "./dataset"
        dataset = build_dataset(args)
        if ctx.rank == 0:
            graph, dataset = prepare_data(args, data_root, dist.get_world_size(group), dataset)
        dist.barrier()
        g = get_graph(data_root, group).to(ctx.device)

        def prepare_data(root: str, num_parts):
            dataset = TwitterTennisDatasetLoader().get_dataset()

            x = []
            y = []
            edge_index = []
            edge_times = []
            edge_attr = []
            snapshot_count = 0
            for i, data in enumerate(dataset):
                x.append(data.x[:,None,:])
                y.append(data.y[:,None])
                edge_index.append(data.edge_index)
                print(data.edge_index.shape)
                exit(0)
                edge_times.append(torch.full_like(data.edge_index[0], i))
                edge_attr.append(data.edge_attr)
                snapshot_count += 1
            x = torch.cat(x, dim=1)
            y = torch.cat(y, dim=1)
            edge_index = torch.cat(edge_index, dim=1)
            edge_times = torch.cat(edge_times, dim=0)
            edge_attr = torch.cat(edge_attr, dim=0)

            g = GraphData(edge_index, num_nodes=x.size(0))
            g.node()["x"] = x
            g.node()["y"] = y
            g.edge()["time"] = edge_times
            g.edge()["attr"] = edge_attr
            g.meta()["num_nodes"] = x.size(0)
            g.meta()["num_snapshots"] = snapshot_count

            logging.info(f"GraphData.meta().keys(): {g.meta().keys()}")
            logging.info(f"GraphData.node().keys(): {g.node().keys()}")
            logging.info(f"GraphData.edge().keys(): {g.edge().keys()}")

            g.save_partition(root, num_parts, algorithm="random")
            return g

3. Creating a partitioned parallel-based GNN model :code:`sync_gnn`, and create a classifier and a splitter

    .. code-block:: python

        sync_gnn = build_model(args, graph=g, group=group)
        sync_gnn = sync_gnn.to(ctx.device)

        classifier = Classifier(args.hidden_dim, args.hidden_dim)
        classifier = classifier.to(ctx.device)
        spl = splitter(args, min_time, max_time)

4.Start to train our model

    .. code-block:: python

        trainer = Trainer(args, spl, sync_gnn, classifier, dataset, ctx)
        trainer.train()

        class Trainer():
            def __init__(self, args, splitter, gcn, classifier, dataset, ctx):
                self.args = args
                self.splitter = splitter
                self.gcn = gcn
                self.classifier = classifier
                self.comp_loss = nn.BCELoss()
                self.group = self.gcn.group
                self.graph = self.gcn.graph
                self.ctx = ctx

                self.logger = logger.Logger(args, 1)

                self.num_nodes = dataset.num_nodes
                self.data = dataset
                self.time = {'TRAIN': [], 'VALID': [], 'TEST':[]}

                self.init_optimizers(args)

            def init_optimizers(self, args):
                params = self.gcn.parameters()
                self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
                params = self.classifier.parameters()
                self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
                self.gcn_opt.zero_grad()
                self.classifier_opt.zero_grad()

            def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
                torch.save(state, filename)

            def load_checkpoint(self, filename, model):
                if os.path.isfile(filename):
                    print("=> loading checkpoint '{}'".format(filename))
                    checkpoint = torch.load(filename)
                    epoch = checkpoint['epoch']
                    self.gcn.load_state_dict(checkpoint['gcn_dict'])
                    self.classifier.load_state_dict(checkpoint['classifier_dict'])
                    self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
                    self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
                    self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
                    return epoch
                else:
                    self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
                    return 0

            def train(self):
                self.tr_step = 0
                best_eval_valid = 0
                eval_valid = 0
                epochs_without_impr = 0

                for e in range(self.args.num_epochs):
                    eval_train = self.run_epoch(self.splitter.train, e, 'TRAIN', grad=True)
                    if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                        eval_valid = self.run_epoch(self.splitter.dev, e, 'VALID', grad=False)
                        eval_test = self.run_epoch(self.splitter.test, e, 'TEST', grad=False)
                        if eval_valid > best_eval_valid:
                            best_eval_valid = eval_valid
                            best_test = eval_test
                            epochs_without_impr = 0

                for tmp in self.time.keys():
                    self.ctx.sync_print(tmp, np.mean(self.time[tmp]))
                print(eval_test)

            def run_epoch(self, split, epoch, set_name, grad):
                t0 = time.time()
                log_interval = 1
                if set_name == 'TEST':
                    log_interval = 1
                self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

                torch.set_grad_enabled(grad)

                for s in split:
                    hist_snap_ids = s['hist_ts']
                    label_snap_id = s['label_ts']
                    predictions, labels, label_edge = self.predict(hist_snap_ids, label_snap_id, set_name)

                    loss = self.comp_loss(predictions, labels)
                    if set_name == 'TRAIN':
                        loss.backward()

                        all_reduce_gradients(self.gcn)
                        all_reduce_buffers(self.gcn)
                        all_reduce_gradients(self.classifier)
                        all_reduce_buffers(self.classifier)

                        self.gcn_opt.step()
                        self.classifier_opt.step()

                        self.gcn_opt.zero_grad()
                        self.classifier_opt.zero_grad()

                    if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                        self.logger.log_minibatch(predictions, labels, loss.detach(), adj=label_edge)
                        dist.barrier()
                    else:
                        self.logger.log_minibatch(predictions, labels, loss.detach())

                torch.set_grad_enabled(True)
                eval_measure = self.logger.log_epoch_done()
                t1 = time.time()
                self.time[set_name].append(t1-t0)

                return eval_measure

            def predict(self, hist_snap_ids, label_snap_id, set_name):
                nodes_embs_dst = self.gcn(hist_snap_ids)
                num_dst = nodes_embs_dst.shape[0]
                nodes_embs_src = self.gcn.route.apply(nodes_embs_dst)
                num_src = nodes_embs_src.shape[0]
                num_nodes, x, pos_edge_index, edge_attr = self.gcn.get_snapshot(label_snap_id)
                neg_edge_index = self.negative_sampling(num_src, num_dst, edge_attr.shape[0], set_name)

                pos_cls_input = self.gather_node_embs(nodes_embs_src, pos_edge_index, nodes_embs_dst)
                neg_cls_input = self.gather_node_embs(nodes_embs_src, neg_edge_index, nodes_embs_dst)

                pos_predictions = self.classifier(pos_cls_input)
                neg_predictions = self.classifier(neg_cls_input)
                pos_label = torch.ones_like(pos_predictions)
                neg_label = torch.zeros_like(neg_predictions)

                pred = torch.cat([pos_predictions, neg_predictions], dim=0)
                label = torch.cat([pos_label, neg_label], dim=0)

                label_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                return pred.sigmoid(), label, label_edge

            def gather_node_embs(self, nodes_embs_src, node_indices, nodes_embs_dist):
                return torch.cat([nodes_embs_src[node_indices[0,:]], nodes_embs_dist[node_indices[1,:]]], dim=1)

            def optim_step(self, loss):
                self.tr_step += 1
                loss.backward()

                if self.tr_step % self.args.steps_accum_gradients == 0:
                    self.gcn_opt.step()
                    self.classifier_opt.step()

                    self.gcn_opt.zero_grad()
                    self.classifier_opt.zero_grad()

            def negative_sampling(self, num_src, num_dst, num_edge, set_name):
                if set_name == 'TRAIN':
                    num_sample = num_edge * self.args.negative_mult_training
                else:
                    num_sample = num_edge * self.args.negative_mult_test

                src = torch.randint(low=0, high=num_src, size=(num_sample,))
                dst = torch.randint(low=0, high=num_dst, size=(num_sample,))

                return torch.vstack([src, dst]).to(self.ctx.device)
