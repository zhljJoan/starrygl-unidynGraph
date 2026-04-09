#pragma once
#include <head.h>
#include <neighbors.h>
# include <output.h>

class ParallelSampler
{
    public:
        TemporalNeighborBlock& tnb;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int threads;
        vector<int> fanouts;
        // vector<NodeIDType> part_ptr;
        // int pid;
        vector<int>node_part;
        int local_part = -1;
        int num_layers;
        string policy;
        std::vector<TemporalGraphBlock> ret;
        TemporalNeighborList block;
        vector<unsigned int> loc_seeds;
        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, int _num_layers, string _policy, int _local_part, th::Tensor _node_part) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), num_layers(_num_layers), policy(_policy), local_part(_local_part)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret.resize(_num_layers);
            if(local_part != -1){
                int *part_ptr = _node_part.data_ptr<int>();
                node_part.assign(part_ptr, part_ptr + _node_part.numel());
            }
            for(int i = 0; i < threads; i++){
                loc_seeds.push_back(i);
            }
        }

        void reset()
        {
            ret.clear();
            ret.resize(num_layers);
        }

        void neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique);
        void neighbor_sample_from_nodes_static(th::Tensor nodes, bool part_unique);
        void neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer, bool part_unique);
        void neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_dynamic_nodes(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer(th::Tensor nodes, th::Tensor root_ts, int cur_layer);
        template<typename T>
        void union_to_vector(vector<T> *p, vector<T> &to_vec);
};



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique)
{
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform")
        ;
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
        neighbor_sample_from_dynamic_nodes(nodes,root_ts.value());
        //neighbor_sample_from_nodes_with_before(nodes, root_ts.value());
    }
    else{
        bool flag = part_unique.has_value() ? part_unique.value() : true;
        neighbor_sample_from_nodes_static(nodes, flag);
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer, bool part_unique){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();

    TemporalGraphBlock tgb = TemporalGraphBlock();
    int fanout = fanouts[cur_layer];
    ret[cur_layer] = TemporalGraphBlock();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes); 
    vector<phmap::parallel_flat_hash_set<NodeIDType>> node_s_threads(threads);
    vector<vector<NodeIDType>> node_threads(threads);
    phmap::parallel_flat_hash_set<NodeIDType> node_s;
    vector<vector<NodeIDType>> eid_threads(threads);
    vector<vector<NodeIDType>> src_index_threads(threads);
    AT_ASSERTM(tnb.with_eid, "Tnb has no eid infomation! We need eid!");
    
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(nodes.size(0) / threads)) * fanout;
    
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    eid_threads[tid].reserve(reserve_capacity);
    src_index_threads[tid].reserve(reserve_capacity);
    if(!part_unique)
        node_threads[tid].reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil((static_cast<float>(nodes.size(0)) / threads))))
    for(int64_t i=0; i<nodes.size(0); i++){
        NodeIDType node = nodes_data[i];
        vector<NodeIDType>& nei = tnb.neighbors[node];
        vector<EdgeIDType> edge;
        edge = tnb.eid[node];

        double s_start_time = omp_get_wtime();
        if(tnb.deg[node]>fanout){
            phmap::flat_hash_set<NodeIDType> temp_s;
            default_random_engine e(8);//(time(0));
            // uniform_int_distribution<> u(0, tnb.deg[node]-1);            
            // while(temp_s.size()!=fanout && temp_s.size()<tnb.neighbors_set[node].size()){
            for(int i=0;i<fanout;i++){
                //循环选择fanout个邻居
                NodeIDType indice;
                if(policy == "weighted"){//考虑边权重信息
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    indice = sample_multinomial(ew, e);
                }
                else if(policy == "uniform"){//均匀采样
                    // indice = u(e);
                    indice = rand_r(&loc_seed) % (nei.size());
                }
                auto chosen_n_iter = nei.begin() + indice;
                auto chosen_e_iter = edge.begin() + indice;
                if(part_unique){
                    auto rst = temp_s.insert(*chosen_n_iter);
                    if(rst.second){ //不重复
                        eid_threads[tid].emplace_back(*chosen_e_iter);
                        node_s_threads[tid].insert(*chosen_n_iter);
                        if(!tnb.neighbors_set.empty() && temp_s.size()<fanout && temp_s.size()<tnb.neighbors_set[node].size()) fanout++;
                    }
                }
                else{
                    eid_threads[tid].emplace_back(*chosen_e_iter);
                    node_threads[tid].emplace_back(*chosen_n_iter);
                }
            }
            if(part_unique)
                src_index_threads[tid].insert(src_index_threads[tid].end(), temp_s.size(), i);
            else
                src_index_threads[tid].insert(src_index_threads[tid].end(), fanout, i);
        }
        else{
            src_index_threads[tid].insert(src_index_threads[tid].end(), tnb.deg[node], i);
            if(part_unique)
                node_s_threads[tid].insert(nei.begin(), nei.end());
            else
                node_threads[tid].insert(node_threads[tid].end(), nei.begin(), nei.end());
            eid_threads[tid].insert(eid_threads[tid].end(),edge.begin(), edge.end());
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = eid_threads[i].size();
        each_begin[i]=size;
        size += s;
    }
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].src_index.resize(size);
#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        copy(eid_threads[i].begin(), eid_threads[i].end(), ret[cur_layer].eid.begin()+each_begin[i]);
        if(!part_unique)
            copy(node_threads[i].begin(), node_threads[i].end(), ret[cur_layer].sample_nodes.begin()+each_begin[i]);
        copy(src_index_threads[i].begin(), src_index_threads[i].end(), ret[cur_layer].src_index.begin()+each_begin[i]);
    }
    if(part_unique){
        for(int i = 0; i<threads; i++)
            node_s.insert(node_s_threads[i].begin(), node_s_threads[i].end());
        ret[cur_layer].sample_nodes.assign(node_s.begin(), node_s.end());
    }

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_static(th::Tensor nodes, bool part_unique){
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_static_layer(nodes, i, part_unique);
        else neighbor_sample_from_nodes_static_layer(vecToTensor<NodeIDType>(ret[i-1].sample_nodes), i, part_unique);
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer(
        th::Tensor nodes, th::Tensor root_ts, int cur_layer){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    ret[cur_layer] = TemporalGraphBlock();
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    int fanout = fanouts[cur_layer];
    // HashT<pair<NodeIDType,TimeStampType> > node_s;
    vector<TemporalGraphBlock> tgb_i(threads);
    
    default_random_engine e(8);//(time(0));
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(nodes.size(0) / threads)) * fanout;
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    tgb_i[tid].sample_nodes.reserve(reserve_capacity);
    tgb_i[tid].sample_nodes_ts.reserve(reserve_capacity);
    tgb_i[tid].delta_ts.reserve(reserve_capacity);
    tgb_i[tid].eid.reserve(reserve_capacity);
    tgb_i[tid].src_index.reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil((static_cast<float>(nodes.size(0)) / threads))))
    for(int64_t i=0; i<nodes.size(0); i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        if(local_part > -1 && node_part[node] != local_part){
            continue;
        }
        TimeStampType rtts = ts_data[i];
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
        // cout<<node<<" "<<end_index<<" "<<tnb.deg[node]<<endl;

        double s_start_time = omp_get_wtime();
        if ((policy == "recent") || (end_index <= fanout)){
            int start_index = max(0, end_index-fanout);
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), tnb.neighbors[node].begin()+start_index, tnb.neighbors[node].begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), tnb.timestamp[node].begin()+start_index, tnb.timestamp[node].begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), tnb.eid[node].begin()+start_index, tnb.eid[node].begin()+end_index);
            for(int cid = start_index; cid < end_index;cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
            }
        }
        else{
            //可选邻居边大于扇出的话需要随机选择fanout个邻居
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), fanout, i);
            uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int i=0; i<fanout;i++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = rand_r(&loc_seed) % (end_index);
                else if(policy == "weighted"){
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, e);
                }
                tgb_i[tid].sample_nodes.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
            }
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    // start_time = omp_get_wtime();

    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = tgb_i[i].eid.size();
        each_begin[i]=size;
        size += s;
    }
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].src_index.resize(size);
    ret[cur_layer].delta_ts.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].sample_nodes_ts.resize(size);
#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        copy(tgb_i[i].eid.begin(), tgb_i[i].eid.end(), ret[cur_layer].eid.begin()+each_begin[i]);
        copy(tgb_i[i].src_index.begin(), tgb_i[i].src_index.end(), ret[cur_layer].src_index.begin()+each_begin[i]);
        copy(tgb_i[i].delta_ts.begin(), tgb_i[i].delta_ts.end(), ret[cur_layer].delta_ts.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes.begin(), tgb_i[i].sample_nodes.end(), ret[cur_layer].sample_nodes.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes_ts.begin(), tgb_i[i].sample_nodes_ts.end(), ret[cur_layer].sample_nodes_ts.begin()+each_begin[i]);
    }

    // end_time = omp_get_wtime();
    // cout<<"end union consume: "<<end_time-start_time<<"s"<<endl;

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts){
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_with_before_layer(nodes, root_ts, i);
        else neighbor_sample_from_nodes_with_before_layer(vecToTensor<NodeIDType>(ret[i-1].sample_nodes), 
                                                          vecToTensor<TimeStampType>(ret[i-1].sample_nodes_ts), i);
    }
}

template <typename T>
void ParallelSampler::union_to_vector(vector<T> *p, vector<T> &to_vec){
    int sz = 0;
    for(int i=0 ;i<threads; i++){
        sz+=p[i].size();
    }
    to_vec.resize(sz);
    sz = 0;
    for(int i=0;i<threads;i++){
        copy(p[i].begin(),p[i].end(),to_vec.begin()+sz);
        sz+=p[i].size();
    }
}

void ParallelSampler :: neighbor_sample_from_dynamic_nodes(th::Tensor nodes, th::Tensor root_ts){
    clock_t start_time = clock();
    int work_thread = threads;
    HashM <pair<NodeIDType,TimeStampType>,int> unq_id;
    HashM <EdgeIDType,int> unq_eid;
    atomic<int> id_cnt(0);
    atomic<int> eid_cnt(0);
    auto node_ptr = get_data_ptr<NodeIDType>(nodes);
    auto ts_ptr = get_data_ptr<TimeStampType>(root_ts);
    vector<int> uid_vec_i[work_thread];
    std::mutex mtx;
    std::mutex emtx;
    #pragma omp parallel for num_threads(work_thread)
    for(int i = 0 ; i< nodes.size(0);i++){
        int tid = omp_get_thread_num();
        pair<NodeIDType,TimeStampType>pr = make_pair(node_ptr[i],ts_ptr[i]);
        //if(unq_id.find(pr)== unq_id.end()){
        //unq_id[pr] = unq_id.find(pr) == unq_id.end() ? id_cnt++ : unq_id[pr];
        //}
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto res = unq_id.insert({pr,0});
            if(res.second){
                unq_id[pr] = id_cnt.fetch_add(1);
            //printf("root unq_id %lld %f %d\n",pr.first,pr.second,unq_id[pr]);
            }
        }
        //uid_vec_i[tid].push_back(unq_id[pr]);
    }   
    #pragma omp parallel for num_threads(work_thread)
    for(int i = 0 ; i< nodes.size(0);i++){
        int tid = omp_get_thread_num();
        pair<NodeIDType,TimeStampType>pr = make_pair(node_ptr[i],ts_ptr[i]);
        uid_vec_i[tid].push_back(unq_id[pr]);
    }   
    vector<int> uid_vec;
    union_to_vector(uid_vec_i,uid_vec);
    auto options_int = th::TensorOptions().dtype(torch::kInt32);
    auto options_float = th::TensorOptions().dtype(torch::kFloat32);
    auto options_long = th::TensorOptions().dtype(torch::kInt64);
    vector<th::Tensor>().swap(block.uid);
    vector<th::Tensor>().swap(block.ueid);
    vector<th::Tensor>().swap(block.src_index);
    block.uid.push_back(th::from_blob(uid_vec.data(),th::IntArrayRef(uid_vec.size()),options_int).clone());
    default_random_engine e(8);

    for(int i = 0; i<num_layers;i++){
        auto node_ptr = get_data_ptr<NodeIDType>(nodes);
        auto ts_ptr = get_data_ptr<TimeStampType>(root_ts);
        for(int j = 0;j<work_thread;j++)    
            vector<int>().swap(uid_vec_i[j]);
        vector<NodeIDType>next_root_i[work_thread];
        vector<TimeStampType>next_ts_i[work_thread];
        vector<int> eid_vec_i[work_thread];
        vector<int> src_index_i[work_thread];
        int fanout = fanouts[i];
        #pragma omp parallel for num_threads(work_thread)
        for(int j = 0 ; j< nodes.size(0);j++){
            int tid = omp_get_thread_num();
            NodeIDType node = node_ptr[j];
            TimeStampType rtts = ts_ptr[j];
            int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
            src_index_i[tid].insert(src_index_i[tid].end(), min(fanout,end_index), unq_id[make_pair(node,rtts)]);
            uniform_int_distribution<> u(0, end_index-1);
            int start_index = max(0, end_index - fanout);
            for(int t=0; t<min(end_index,fanout);t++){
                int cid;
                if(policy == "recent" || start_index == 0){
                    cid = start_index + t;
                }
                else if(policy == "uniform"){
                    cid = rand_r(&loc_seeds[tid]) % (end_index);
                }
                else if(policy == "weighted"){
                    default_random_engine e(8);
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, e);
                }
                pair<NodeIDType,TimeStampType> pr0 = make_pair(tnb.neighbors[node][cid],tnb.timestamp[node][cid]);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    auto res0 = unq_id.insert({pr0,0});
                    if(res0.second){
                        unq_id[pr0] = id_cnt.fetch_add(1);
                        //printf("unq_id %lld %f %d\n",pr0.first,pr0.second,unq_id[pr0],id_cnt.load());
                    }
                }
                int eid = tnb.eid[node][cid];
                {
                    std::lock_guard<std::mutex> lock(emtx);
                    auto res1 = unq_eid.insert({eid,0});
                    if(res1.second){
                        unq_eid[eid] = eid_cnt.fetch_add(1);
                    }
                }
                int eid_v = eid;//unq_eid[eid];
                next_root_i[tid].emplace_back(tnb.neighbors[node][cid]);
                next_ts_i[tid].emplace_back(tnb.timestamp[node][cid]);
                //uid_vec_i[tid].emplace_back(uid_v);
                eid_vec_i[tid].emplace_back(eid_v);
            }
        }   
        #pragma omp parallel for num_threads(work_thread)
        for(int tid = 0;tid<work_thread;tid++){
            for(int j=0;j<eid_vec_i[tid].size();j++){
                pair<NodeIDType,TimeStampType> pr0 = make_pair(next_root_i[tid][j],next_ts_i[tid][j]);
                uid_vec_i[tid].emplace_back(unq_id[pr0]);
                eid_vec_i[tid][j] = unq_eid[eid_vec_i[tid][j]];
            }
        }
        vector<NodeIDType>next_root;
        vector<TimeStampType>next_ts;
        vector<int> eid_vec;
        vector<int> src_index;
        vector<int> uid_vec_layer;
        #pragma omp parallel sections
        {
            #pragma omp section
                {
                    
                    union_to_vector(next_root_i,next_root);
                }
                #pragma omp section
                {
                    
                    union_to_vector(next_ts_i,next_ts);
                }
                #pragma omp section
                {
                    
                    union_to_vector(eid_vec_i,eid_vec);
                }
                #pragma omp section
                {
                    union_to_vector(src_index_i,src_index);
                }
                #pragma omp section
                {
                    union_to_vector(uid_vec_i,uid_vec_layer);
                }
        }
        nodes = th::from_blob(next_root.data(),th::IntArrayRef(next_root.size()),options_long);
        root_ts = th::from_blob(next_ts.data(),th::IntArrayRef(next_ts.size()),options_float);
        block.uid.push_back(th::from_blob(uid_vec_layer.data(),th::IntArrayRef(uid_vec_layer.size()),options_int).clone());
        auto ptr = get_data_ptr<int>(block.uid[1]);
        block.ueid.push_back(th::from_blob(eid_vec.data(),th::IntArrayRef(eid_vec.size()),options_int).clone());
        block.src_index.push_back(th::from_blob(src_index.data(),th::IntArrayRef(src_index.size()),options_int).clone());

    }
    block.nids = th::empty({id_cnt},options_long);
    block.eids = th::empty({eid_cnt},options_long);
    block.ts = th::empty({id_cnt},options_float);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for(auto & kv: unq_id){
                
                block.nids[kv.second] = kv.first.first;
                block.ts[kv.second] = kv.first.second;
              //  cout<<kv.second<<" "<<kv.first.first<<" "<<kv.first.second<<" "<<block.nids[kv.second]<<endl;
            }
            //cout<<"block size is "<<block.nids.size(0)<<endl;
            //auto ptr = get_data_ptr<NodeIDType>(block.nids);
           // for(int i = 0 ; i < block.nids.size(0);i++){
              //  cout<<i<<" "<<ptr[i]<<endl;
           // }
            //cout<<endl;
        }
        #pragma omp section
        {
            for(auto & kv: unq_eid){
                block.eids[kv.second] = kv.first;
              //  cout<<kv.second<<" "<<kv.first<<" "<<block.eids[kv.second]<<endl;
            }
        }
    }
    clock_t end_time = clock();
    //std::cout<<double(start_time-end_time)/CLOCKS_PER_SEC<<endl;
}

