#pragma once
#include <head.h>
#include <neighbors.h>
# include <output.h>

struct NodeInstanceKey
{
    NodeIDType node;
    TimeStampType ts;

    bool operator==(const NodeInstanceKey& other) const
    {
        return node == other.node && ts == other.ts;
    }
};

struct NodeInstanceKeyHash
{
    size_t operator()(const NodeInstanceKey& key) const
    {
        size_t h1 = std::hash<NodeIDType>{}(key.node);
        size_t h2 = std::hash<TimeStampType>{}(key.ts);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

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
        vector<int>part;
        vector<int>node_part;
        int local_part = -1;
        int num_layers;
        string policy;
        std::vector<TemporalGraphBlock> ret;
        th::Tensor dist_nid;
        th::Tensor dist_eid;
        th::Tensor block_node_list;
        th::Tensor eid_inv;
        th::Tensor unq_id;
        th::Tensor first_block_id;
        double boundery_probility;
        vector<unsigned int> loc_seeds;
        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, int _num_layers, string _policy, int _local_part, th::Tensor _part, th::Tensor _node_part,double _p) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), num_layers(_num_layers), policy(_policy), local_part(_local_part), boundery_probility(_p)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret.resize(_num_layers);
            if(local_part != -1){
                int *part_ptr = _part.data_ptr<int>();
                part.assign(part_ptr, part_ptr + _part.numel());
                part_ptr = _node_part.data_ptr<int>();
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
        std::vector<NativeSamplingOutput> sample_dtdg_uniform(th::Tensor root_nodes, int64_t t_now, int64_t num_hist);
        template<typename T>
        void union_to_vector(vector<T> *p, vector<T> &to_vec);
        void sample_unique(     th::Tensor seed, th::Tensor seed_ts,
                                th::Tensor nid_mapper, th::Tensor eid_mapper,string out_device);
        NativeSamplingOutput get_sampling_output(th::Tensor root_nodes, optional<th::Tensor> root_ts);
        NativeSamplingOutput get_sampling_output_parallel(th::Tensor root_nodes, optional<th::Tensor> root_ts);
};



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique)
{
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform"|| policy == "dtdg_uniform" || policy =="boundery_recent_decay"||policy=="boundery_recent_uniform"||policy=="boundery_uniform");
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
        //neighbor_sample_from_dynamic_nodes(nodes,root_ts.value());
        neighbor_sample_from_nodes_with_before(nodes, root_ts.value());
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
                    indice = rand_r(&loc_seeds[tid]) % (nei.size());
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
    tgb_i[tid].col.reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil((static_cast<float>(nodes.size(0)) / threads))))
    for(int64_t i=0; i<nodes.size(0); i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        int end_index = lower_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
        int start_index = 0;
        if(policy == "dtdg_uniform"){
            start_index = end_index;
            end_index = upper_bound(tnb.timestamp[node].begin(), tnb.timestamp[node].end(), rtts)-tnb.timestamp[node].begin();
        }

        double s_start_time = omp_get_wtime();
        
        if(policy == "dtdg_uniform" && end_index <= start_index){
            continue;
        }
        else if(policy == "dtdg_uniform" && (end_index - start_index) <= fanout){
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), tnb.neighbors[node].begin()+start_index, tnb.neighbors[node].begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), tnb.timestamp[node].begin()+start_index, tnb.timestamp[node].begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), tnb.eid[node].begin()+start_index, tnb.eid[node].begin()+end_index);
            for(int cid = start_index; cid < end_index; cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].sample_weight.emplace_back(cid-start_index);
            }
        }
        else if ((policy == "recent") || (end_index <= fanout)&& policy.substr(0,8) != "boundery" ){
            int cnt  = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cnt++;
                if(cnt>fanout)break;
            }
            int start_index = max(0, end_index-fanout);
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), tnb.neighbors[node].begin()+start_index, tnb.neighbors[node].begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), tnb.timestamp[node].begin()+start_index, tnb.timestamp[node].begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), tnb.eid[node].begin()+start_index, tnb.eid[node].begin()+end_index);
            for(int cid = start_index; cid < end_index;cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].sample_weight.emplace_back(cid-start_index);
            }
        }
        else if(policy == "boundery_recent_uniform"){
            int cnt = 0;
            int cal_cnt = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt > fanout)break;
                int eid = tnb.eid[node][cid];
                 if(part[tnb.eid[node][cid]] != local_part|| node_part[tnb.neighbors[node][cid]]!= local_part){
                    double p0 = (double)rand_r(&loc_seeds[tid]) / (RAND_MAX + 1.0);
                    if(p0 > boundery_probility)continue;
                }
                tgb_i[tid].src_index.emplace_back(i);
                tgb_i[tid].sample_nodes.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
                cnt++;
                if(cnt > fanout)break;
            
            }

        }
        else if(policy == "boundery_recent_decay"){
            int cnt = 0;
            int cal_cnt = 0;
            double sum_p = 0;
            int sum_1 = 0;
            vector<double>pr(2*fanout);
            TimeStampType delta = end_index-1>=0?(rtts - tnb.timestamp[node][end_index-1])*fanout:0; 
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt>fanout)break;
                if(part[tnb.eid[node][cid]] != local_part|| node_part[tnb.neighbors[node][cid]]!= local_part){
                    double ep = exp((double)(tnb.timestamp[node][cid]-rtts)/(delta));
                    sum_p+=ep;pr[cal_cnt-1]=ep;
                    sum_1++;
                }
            }
            if(sum_p<1e-6)sum_p=1;
            cal_cnt = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                //if(cal_cnt > fanout)break;
                int eid = tnb.eid[node][cid];
                
                if((part[tnb.eid[node][cid]] != local_part|| node_part[tnb.neighbors[node][cid]]!= local_part)){
                    if(cal_cnt<=fanout){
                        double p0 = (double)rand_r(&loc_seeds[tid]) / (RAND_MAX + 1.0);
                        double ep = boundery_probility*pr[cal_cnt-1]/sum_p*sum_1;
                        if(p0 > ep)continue;
                        //tgb_i[tid].sample_weight.emplace_back((float)ep);
                    }
                    else continue;
                    //cout<<"in"<<endl;
                    
                }
                else{
                    //tgb_i[tid].sample_weight.emplace_back((float)1.0);
                }
                tgb_i[tid].src_index.emplace_back(i);
                tgb_i[tid].sample_nodes.emplace_back(tnb.neighbors[node][cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(tnb.timestamp[node][cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-tnb.timestamp[node][cid]);
                tgb_i[tid].eid.emplace_back(tnb.eid[node][cid]);
                cnt++;
                if(cnt > fanout)break;
            }
        }
        else{
            //可选邻居边大于扇出的话需要随机选择fanout个邻居
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), fanout, i);
            uniform_int_distribution<> u(start_index, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int i=0; i<fanout;i++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = rand_r(&loc_seeds[tid]) % (end_index);
                else if(policy == "dtdg_uniform")
                    cid = start_index + (rand_r(&loc_seeds[tid]) % (end_index - start_index));
                else if(policy == "weighted"|| policy == "boundery_uniform"){
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
    //if(policy == "boundery_recent_decay")
    ret[cur_layer].sample_weight.resize(size);
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].src_index.resize(size);
    ret[cur_layer].delta_ts.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].sample_nodes_ts.resize(size);

#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        //if(policy == "boundery_recent_decay")
        copy(tgb_i[i].sample_weight.begin(), tgb_i[i].sample_weight.end(), ret[cur_layer].sample_weight.begin()+each_begin[i]);
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

std::vector<NativeSamplingOutput> ParallelSampler::sample_dtdg_uniform(th::Tensor root_nodes, int64_t t_now, int64_t num_hist)
{
    AT_ASSERTM(tnb.with_timestamp, "DTDG uniform sampling requires timestamped/time-slice edges");
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    AT_ASSERTM(num_hist >= 1, "num_hist must be >= 1");

    std::vector<NativeSamplingOutput> outputs;
    outputs.reserve(static_cast<size_t>(num_hist));
    const std::string old_policy = policy;
    policy = "dtdg_uniform";

    int64_t start_slice = t_now - num_hist + 1;
    for(int64_t slice = start_slice; slice <= t_now; slice++){
        reset();
        th::Tensor root_ts = th::full(
            {root_nodes.size(0)},
            slice,
            th::TensorOptions().dtype(th::kInt64).device(root_nodes.device())
        ).contiguous();
        neighbor_sample_from_nodes_with_before(root_nodes, root_ts);
        NativeSamplingOutput out = get_sampling_output(root_nodes, root_ts);
        outputs.emplace_back(std::move(out));
    }

    policy = old_policy;
    return outputs;
}


NativeSamplingOutput ParallelSampler::get_sampling_output(th::Tensor root_nodes, optional<th::Tensor> root_ts)
{
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    if(root_ts.has_value()){
        AT_ASSERTM(root_ts.value().is_contiguous(), "root_ts must be contiguous");
        AT_ASSERTM(root_ts.value().dim() == 1, "root_ts must be one-dimensional");
        AT_ASSERTM(root_ts.value().size(0) == root_nodes.size(0), "root_ts size must match root_nodes");
    }

    py::gil_scoped_release release;

    NativeSamplingOutput out;
    out.mfgs.resize(ret.size());
    out.node_layer_ptr.reserve(ret.size() + 2);
    out.edge_layer_ptr.reserve(ret.size() + 1);
    out.node_layer_ptr.emplace_back(0);
    out.edge_layer_ptr.emplace_back(0);

    size_t estimated_nodes = static_cast<size_t>(root_nodes.size(0));
    size_t estimated_edges = 0;
    for(const TemporalGraphBlock& block : ret){
        estimated_nodes += block.sample_nodes.size();
        estimated_edges += block.eid.size();
    }
    out.node_gids.reserve(estimated_nodes);
    out.node_ts.reserve(estimated_nodes);
    out.edge_gids.reserve(estimated_edges);
    out.edge_ts.reserve(estimated_edges);

    phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> node_lid;
    phmap::flat_hash_map<EdgeIDType, int64_t> edge_lid;
    node_lid.reserve(estimated_nodes);
    edge_lid.reserve(estimated_edges);

    auto add_node = [&](NodeIDType gid, TimeStampType ts) -> int64_t {
        NodeInstanceKey key{gid, ts};
        auto it = node_lid.find(key);
        if(it != node_lid.end()) return it->second;
        int64_t lid = static_cast<int64_t>(out.node_gids.size());
        node_lid.emplace(key, lid);
        out.node_gids.emplace_back(gid);
        out.node_ts.emplace_back(ts);
        return lid;
    };

    auto add_edge = [&](EdgeIDType gid, TimeStampType ts) -> int64_t {
        auto it = edge_lid.find(gid);
        if(it != edge_lid.end()) return it->second;
        int64_t lid = static_cast<int64_t>(out.edge_gids.size());
        edge_lid.emplace(gid, lid);
        out.edge_gids.emplace_back(gid);
        out.edge_ts.emplace_back(ts);
        return lid;
    };

    auto root_nodes_data = get_data_ptr<NodeIDType>(root_nodes);
    TimeStampType* root_ts_data = nullptr;
    if(root_ts.has_value()){
        root_ts_data = get_data_ptr<TimeStampType>(root_ts.value());
    }

    out.root_gids.reserve(root_nodes.size(0));
    out.root_ts.reserve(root_nodes.size(0));
    out.root_lids.reserve(root_nodes.size(0));
    vector<int64_t> frontier_lids;
    frontier_lids.reserve(root_nodes.size(0));

    for(int64_t i = 0; i < root_nodes.size(0); i++){
        NodeIDType gid = root_nodes_data[i];
        TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        int64_t lid = add_node(gid, ts);
        out.root_gids.emplace_back(gid);
        out.root_ts.emplace_back(ts);
        out.root_lids.emplace_back(lid);
        frontier_lids.emplace_back(lid);
    }
    out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));

    for(int64_t layer = 0; layer < static_cast<int64_t>(ret.size()); layer++){
        TemporalGraphBlock& block = ret[layer];
        NativeMFGBlock mfg;
        mfg.layer = layer;
        mfg.dst_begin = layer == 0 ? 0 : out.node_layer_ptr[layer];
        mfg.dst_end = out.node_layer_ptr[layer + 1];

        const int64_t num_dst = static_cast<int64_t>(frontier_lids.size());
        mfg.dst_lids.assign(frontier_lids.begin(), frontier_lids.end());

        mfg.csc_indptr.assign(num_dst + 1, 0);
        const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
        for(int64_t j = 0; j < edge_count; j++){
            int64_t dst_pos = block.src_index[j];
            if(dst_pos >= 0 && dst_pos < num_dst){
                mfg.csc_indptr[dst_pos + 1] += 1;
            }
        }
        for(int64_t i = 0; i < num_dst; i++){
            mfg.csc_indptr[i + 1] += mfg.csc_indptr[i];
        }

        vector<int64_t> cursor = mfg.csc_indptr;
        mfg.csc_indices.resize(edge_count);
        mfg.edge_lids.resize(edge_count);
        mfg.delta_t.resize(edge_count);
        vector<int64_t> next_frontier_lids(static_cast<size_t>(block.sample_nodes.size()), -1);

        phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> local_node_index;
        phmap::flat_hash_map<EdgeIDType, int64_t> local_edge_index;
        vector<NodeInstanceKey> local_nodes;
        vector<EdgeIDType> local_edges;
        vector<TimeStampType> local_edge_ts;
        vector<int64_t> edge_to_local_node(static_cast<size_t>(edge_count), -1);
        vector<int64_t> edge_to_local_edge(static_cast<size_t>(edge_count), -1);
        local_node_index.reserve(static_cast<size_t>(edge_count));
        local_edge_index.reserve(static_cast<size_t>(edge_count));
        local_nodes.reserve(static_cast<size_t>(edge_count));
        local_edges.reserve(static_cast<size_t>(edge_count));
        local_edge_ts.reserve(static_cast<size_t>(edge_count));

        for(int64_t j = 0; j < edge_count; j++){
            const NodeInstanceKey node_key{
                block.sample_nodes[j],
                block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j],
            };
            auto node_it = local_node_index.find(node_key);
            if(node_it == local_node_index.end()){
                int64_t idx = static_cast<int64_t>(local_nodes.size());
                local_node_index.emplace(node_key, idx);
                local_nodes.emplace_back(node_key);
                edge_to_local_node[static_cast<size_t>(j)] = idx;
            } else {
                edge_to_local_node[static_cast<size_t>(j)] = node_it->second;
            }

            const EdgeIDType edge_gid = block.eid[j];
            auto edge_it = local_edge_index.find(edge_gid);
            if(edge_it == local_edge_index.end()){
                int64_t idx = static_cast<int64_t>(local_edges.size());
                local_edge_index.emplace(edge_gid, idx);
                local_edges.emplace_back(edge_gid);
                local_edge_ts.emplace_back(node_key.ts);
                edge_to_local_edge[static_cast<size_t>(j)] = idx;
            } else {
                edge_to_local_edge[static_cast<size_t>(j)] = edge_it->second;
            }
        }

        vector<int64_t> local_node_lids(local_nodes.size(), -1);
        for(int64_t i = 0; i < static_cast<int64_t>(local_nodes.size()); i++){
            local_node_lids[static_cast<size_t>(i)] = add_node(local_nodes[static_cast<size_t>(i)].node, local_nodes[static_cast<size_t>(i)].ts);
        }
        vector<int64_t> local_edge_lids(local_edges.size(), -1);
        for(int64_t i = 0; i < static_cast<int64_t>(local_edges.size()); i++){
            local_edge_lids[static_cast<size_t>(i)] = add_edge(local_edges[static_cast<size_t>(i)], local_edge_ts[static_cast<size_t>(i)]);
        }

        for(int64_t j = 0; j < edge_count; j++){
            int64_t src_lid = local_node_lids[static_cast<size_t>(edge_to_local_node[static_cast<size_t>(j)])];
            next_frontier_lids[static_cast<size_t>(j)] = src_lid;
            int64_t dst_pos = block.src_index[j];
            if(dst_pos < 0 || dst_pos >= num_dst) continue;
            int64_t offset = cursor[dst_pos]++;
            mfg.csc_indices[offset] = src_lid;
            mfg.edge_lids[offset] = local_edge_lids[static_cast<size_t>(edge_to_local_edge[static_cast<size_t>(j)])];
            mfg.delta_t[offset] = block.delta_ts.empty() ? 0 : block.delta_ts[j];
        }

        mfg.src_begin = 0;
        mfg.src_end = static_cast<int64_t>(out.node_gids.size());
        mfg.src_lids.resize(mfg.src_end - mfg.src_begin);
        std::iota(mfg.src_lids.begin(), mfg.src_lids.end(), mfg.src_begin);
        out.mfgs[layer] = std::move(mfg);
        out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));
        out.edge_layer_ptr.emplace_back(static_cast<int64_t>(out.edge_gids.size()));
        frontier_lids.swap(next_frontier_lids);
    }
    return out;
}


NativeSamplingOutput ParallelSampler::get_sampling_output_parallel(th::Tensor root_nodes, optional<th::Tensor> root_ts)
{
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    if(root_ts.has_value()){
        AT_ASSERTM(root_ts.value().is_contiguous(), "root_ts must be contiguous");
        AT_ASSERTM(root_ts.value().dim() == 1, "root_ts must be one-dimensional");
        AT_ASSERTM(root_ts.value().size(0) == root_nodes.size(0), "root_ts size must match root_nodes");
    }

    py::gil_scoped_release release;

    struct FirstEdgeOccurrence
    {
        uint64_t pos;
        TimeStampType ts;
    };
    struct NodeOccurrenceEntry
    {
        NodeInstanceKey key;
        uint64_t pos;
    };
    struct EdgeOccurrenceEntry
    {
        EdgeIDType gid;
        uint64_t pos;
        TimeStampType ts;
    };

    auto update_node_first = [](phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash>& first,
                                const NodeInstanceKey& key,
                                uint64_t pos) {
        auto it = first.find(key);
        if(it == first.end()){
            first.emplace(key, pos);
        }
        else if(pos < it->second){
            it->second = pos;
        }
    };
    auto update_edge_first = [](phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence>& first,
                                EdgeIDType gid,
                                TimeStampType ts,
                                uint64_t pos) {
        auto it = first.find(gid);
        if(it == first.end()){
            first.emplace(gid, FirstEdgeOccurrence{pos, ts});
        }
        else if(pos < it->second.pos){
            it->second = FirstEdgeOccurrence{pos, ts};
        }
    };

    const int nthreads = std::max(1, threads);
    const int64_t num_roots = root_nodes.size(0);
    const int64_t num_layers_i64 = static_cast<int64_t>(ret.size());
    vector<uint64_t> layer_dst_base(ret.size(), 0);
    vector<uint64_t> layer_edge_base(ret.size(), 0);
    vector<uint64_t> layer_end_pos(ret.size(), static_cast<uint64_t>(num_roots));
    uint64_t next_pos = static_cast<uint64_t>(num_roots);
    size_t estimated_nodes = static_cast<size_t>(num_roots);
    size_t estimated_edges = 0;
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        const int64_t num_dst = layer == 0 ? num_roots : static_cast<int64_t>(ret[layer - 1].sample_nodes.size());
        layer_dst_base[layer] = next_pos;
        next_pos += static_cast<uint64_t>(num_dst);
        layer_edge_base[layer] = next_pos;
        next_pos += static_cast<uint64_t>(ret[layer].src_index.size());
        layer_end_pos[layer] = next_pos;
        estimated_nodes += ret[layer].sample_nodes.size();
        estimated_edges += ret[layer].eid.size();
    }

    auto root_nodes_data = get_data_ptr<NodeIDType>(root_nodes);
    TimeStampType* root_ts_data = nullptr;
    if(root_ts.has_value()){
        root_ts_data = get_data_ptr<TimeStampType>(root_ts.value());
    }

    phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash> global_node_first;
    phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence> global_edge_first;
    global_node_first.reserve(estimated_nodes);
    global_edge_first.reserve(estimated_edges);

    for(int64_t i = 0; i < num_roots; i++){
        TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        update_node_first(global_node_first, NodeInstanceKey{root_nodes_data[i], ts}, static_cast<uint64_t>(i));
    }

    vector<phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash>> local_node_first(nthreads);
    vector<phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence>> local_edge_first(nthreads);
    for(int tid = 0; tid < nthreads; tid++){
        local_node_first[tid].reserve(estimated_nodes / nthreads + 1);
        local_edge_first[tid].reserve(estimated_edges / nthreads + 1);
    }

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto& node_first = local_node_first[tid];
        auto& edge_first = local_edge_first[tid];
#pragma omp for schedule(static)
        for(int64_t layer = 0; layer < num_layers_i64; layer++){
            if(layer > 0){
                TemporalGraphBlock& prev = ret[layer - 1];
                const uint64_t dst_base = layer_dst_base[layer];
                for(int64_t i = 0; i < static_cast<int64_t>(prev.sample_nodes.size()); i++){
                    TimeStampType ts = prev.sample_nodes_ts.empty() ? 0 : prev.sample_nodes_ts[i];
                    update_node_first(node_first, NodeInstanceKey{prev.sample_nodes[i], ts}, dst_base + static_cast<uint64_t>(i));
                }
            }

            TemporalGraphBlock& block = ret[layer];
            const int64_t num_dst = layer == 0 ? num_roots : static_cast<int64_t>(ret[layer - 1].sample_nodes.size());
            const uint64_t edge_base = layer_edge_base[layer];
            const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
            for(int64_t j = 0; j < edge_count; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                const TimeStampType src_ts = block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j];
                const uint64_t pos = edge_base + static_cast<uint64_t>(j);
                update_node_first(node_first, NodeInstanceKey{block.sample_nodes[j], src_ts}, pos);
                update_edge_first(edge_first, block.eid[j], src_ts, pos);
            }
        }
    }

    for(int tid = 0; tid < nthreads; tid++){
        for(const auto& item : local_node_first[tid]){
            update_node_first(global_node_first, item.first, item.second);
        }
        for(const auto& item : local_edge_first[tid]){
            update_edge_first(global_edge_first, item.first, item.second.ts, item.second.pos);
        }
    }

    vector<NodeOccurrenceEntry> node_entries;
    node_entries.reserve(global_node_first.size());
    for(const auto& item : global_node_first){
        node_entries.push_back(NodeOccurrenceEntry{item.first, item.second});
    }
    std::sort(node_entries.begin(), node_entries.end(), [](const NodeOccurrenceEntry& a, const NodeOccurrenceEntry& b) {
        if(a.pos != b.pos) return a.pos < b.pos;
        if(a.key.node != b.key.node) return a.key.node < b.key.node;
        return a.key.ts < b.key.ts;
    });

    vector<EdgeOccurrenceEntry> edge_entries;
    edge_entries.reserve(global_edge_first.size());
    for(const auto& item : global_edge_first){
        edge_entries.push_back(EdgeOccurrenceEntry{item.first, item.second.pos, item.second.ts});
    }
    std::sort(edge_entries.begin(), edge_entries.end(), [](const EdgeOccurrenceEntry& a, const EdgeOccurrenceEntry& b) {
        if(a.pos != b.pos) return a.pos < b.pos;
        return a.gid < b.gid;
    });

    NativeSamplingOutput out;
    out.mfgs.resize(ret.size());
    out.node_gids.resize(node_entries.size());
    out.node_ts.resize(node_entries.size());
    out.edge_gids.resize(edge_entries.size());
    out.edge_ts.resize(edge_entries.size());
    out.node_layer_ptr.reserve(ret.size() + 2);
    out.edge_layer_ptr.reserve(ret.size() + 1);
    out.root_gids.reserve(num_roots);
    out.root_ts.reserve(num_roots);
    out.root_lids.reserve(num_roots);

    phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> node_lid;
    phmap::flat_hash_map<EdgeIDType, int64_t> edge_lid;
    node_lid.reserve(node_entries.size());
    edge_lid.reserve(edge_entries.size());
    for(int64_t i = 0; i < static_cast<int64_t>(node_entries.size()); i++){
        const NodeInstanceKey& key = node_entries[i].key;
        node_lid.emplace(key, i);
        out.node_gids[i] = key.node;
        out.node_ts[i] = key.ts;
    }
    for(int64_t i = 0; i < static_cast<int64_t>(edge_entries.size()); i++){
        edge_lid.emplace(edge_entries[i].gid, i);
        out.edge_gids[i] = edge_entries[i].gid;
        out.edge_ts[i] = edge_entries[i].ts;
    }

    for(int64_t i = 0; i < num_roots; i++){
        const TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        const NodeIDType gid = root_nodes_data[i];
        out.root_gids.emplace_back(gid);
        out.root_ts.emplace_back(ts);
        out.root_lids.emplace_back(node_lid.at(NodeInstanceKey{gid, ts}));
    }

    out.node_layer_ptr.emplace_back(0);
    int64_t node_cursor = 0;
    auto append_node_layer_ptr = [&](uint64_t end_pos) {
        while(node_cursor < static_cast<int64_t>(node_entries.size()) && node_entries[node_cursor].pos < end_pos){
            node_cursor++;
        }
        out.node_layer_ptr.emplace_back(node_cursor);
    };
    append_node_layer_ptr(static_cast<uint64_t>(num_roots));
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        append_node_layer_ptr(layer_end_pos[layer]);
    }

    out.edge_layer_ptr.emplace_back(0);
    int64_t edge_cursor = 0;
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        while(edge_cursor < static_cast<int64_t>(edge_entries.size()) && edge_entries[edge_cursor].pos < layer_end_pos[layer]){
            edge_cursor++;
        }
        out.edge_layer_ptr.emplace_back(edge_cursor);
    }

    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        TemporalGraphBlock& block = ret[layer];
        NativeMFGBlock mfg;
        mfg.layer = layer;
        mfg.dst_begin = layer == 0 ? 0 : out.node_layer_ptr[layer];
        mfg.dst_end = out.node_layer_ptr[layer + 1];

        const vector<NodeIDType>* frontier_nodes_ptr = nullptr;
        const vector<TimeStampType>* frontier_ts_ptr = nullptr;
        vector<NodeIDType> root_frontier_nodes;
        vector<TimeStampType> root_frontier_ts;
        if(layer == 0){
            root_frontier_nodes.assign(root_nodes_data, root_nodes_data + num_roots);
            root_frontier_ts.reserve(num_roots);
            for(int64_t i = 0; i < num_roots; i++){
                root_frontier_ts.emplace_back(root_ts_data == nullptr ? 0 : root_ts_data[i]);
            }
            frontier_nodes_ptr = &root_frontier_nodes;
            frontier_ts_ptr = &root_frontier_ts;
        }
        else{
            frontier_nodes_ptr = &ret[layer - 1].sample_nodes;
            frontier_ts_ptr = &ret[layer - 1].sample_nodes_ts;
        }
        const vector<NodeIDType>& frontier_nodes = *frontier_nodes_ptr;
        const vector<TimeStampType>& frontier_ts = *frontier_ts_ptr;
        const int64_t num_dst = static_cast<int64_t>(frontier_nodes.size());

        mfg.dst_lids.resize(num_dst);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for(int64_t i = 0; i < num_dst; i++){
            const TimeStampType ts = frontier_ts.empty() ? 0 : frontier_ts[i];
            mfg.dst_lids[i] = node_lid.at(NodeInstanceKey{frontier_nodes[i], ts});
        }

        const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
        mfg.csc_indptr.assign(num_dst + 1, 0);
        vector<vector<int64_t>> thread_dst_counts(nthreads, vector<int64_t>(num_dst, 0));
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            const int64_t begin = edge_count * tid / nthreads;
            const int64_t end = edge_count * (tid + 1) / nthreads;
            for(int64_t j = begin; j < end; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos >= 0 && dst_pos < num_dst){
                    thread_dst_counts[tid][dst_pos] += 1;
                }
            }
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            int64_t count = 0;
            for(int tid = 0; tid < nthreads; tid++){
                count += thread_dst_counts[tid][dst];
            }
            mfg.csc_indptr[dst + 1] = count;
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            mfg.csc_indptr[dst + 1] += mfg.csc_indptr[dst];
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            int64_t base = mfg.csc_indptr[dst];
            for(int tid = 0; tid < nthreads; tid++){
                const int64_t count = thread_dst_counts[tid][dst];
                thread_dst_counts[tid][dst] = base;
                base += count;
            }
        }

        mfg.csc_indices.resize(edge_count);
        mfg.edge_lids.resize(edge_count);
        mfg.delta_t.resize(edge_count);
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            const int64_t begin = edge_count * tid / nthreads;
            const int64_t end = edge_count * (tid + 1) / nthreads;
            for(int64_t j = begin; j < end; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                const TimeStampType src_ts = block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j];
                const int64_t offset = thread_dst_counts[tid][dst_pos]++;
                mfg.csc_indices[offset] = node_lid.at(NodeInstanceKey{block.sample_nodes[j], src_ts});
                mfg.edge_lids[offset] = edge_lid.at(block.eid[j]);
                mfg.delta_t[offset] = block.delta_ts.empty() ? 0 : block.delta_ts[j];
            }
        }

        mfg.src_begin = 0;
        mfg.src_end = static_cast<int64_t>(out.node_gids.size());
        mfg.src_lids.resize(mfg.src_end - mfg.src_begin);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for(int64_t lid = mfg.src_begin; lid < mfg.src_end; lid++){
            mfg.src_lids[lid - mfg.src_begin] = lid;
        }
        out.mfgs[layer] = std::move(mfg);
    }

    return out;
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


void ParallelSampler::sample_unique(th::Tensor seed, th::Tensor seed_ts,
                                th::Tensor nid_mapper, th::Tensor eid_mapper,string out_device){
    th::Device device(torch::kCPU);
    if(out_device == "cpu"){}
    else device = th::Device(torch::kCUDA, out_device[0]-'0');
    vector<th::Tensor> eid_all_vec;
    vector<th::Tensor> dst_all_vec;  
    vector<th::Tensor> dst_ts_all_vec;  
    int hop = ret.size();
    vector<int> len(hop);
    for(int l = 0; l<hop;l++){
        int llen = ret[l].eid.end()-ret[l].eid.begin();
        len[l] = llen;
        eid_all_vec.emplace_back(torch::from_blob(ret[l].eid.data(), llen, torch::kInt64));
        dst_all_vec.emplace_back(torch::from_blob(ret[l].sample_nodes.data(), llen, torch::kInt64));
        dst_ts_all_vec.emplace_back(torch::from_blob(ret[l].sample_nodes_ts.data(),llen, torch::kFloat32));
    }
    th::Tensor dst = th::cat(dst_all_vec,0);
    th::Tensor dst_ts = th::cat(dst_ts_all_vec,0);
    th::Tensor eid_tensor = th::cat(eid_all_vec,0).to(eid_mapper.device());
    dist_eid = eid_mapper.index_select(0,eid_tensor).to(device);
    auto result0 = th::unique_dim(dist_eid,0,false,true,false);
    dist_eid = std::get<0>(result0);
    eid_inv = std::get<1>(result0);
    th::Tensor src_node = dst.to(nid_mapper.device());
    th::Tensor src_ts = th::cat({seed_ts.to(device),dst_ts.to(device)});
    th::Tensor nid_tensor = th::cat({seed.to(nid_mapper.device()),src_node});
    dist_nid = nid_mapper.index_select(0,nid_tensor).to(device);
    auto result1 = th::unique_dim(dist_nid,0,false,true,false);
    dist_nid = std::get<0>(result1);
    th::Tensor nid_inv = std::get<1>(result1);
    auto result2 = th::unique_dim(th::stack({nid_inv,src_ts.to(nid_inv.dtype())}),1,false,true,false);
    block_node_list = std::get<0>(result2);
    unq_id = std::get<1>(result2);
    th::Tensor array = th::arange(unq_id.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    th::Tensor first_index = th::empty(block_node_list.size(1),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    first_index = th::scatter_reduce(first_index, 0, unq_id, array, "amin",false);
    th::Tensor first_mask = th::zeros(unq_id.size(0),torch::TensorOptions().dtype(th::kBool).device(device));
    first_mask.index_fill_(0, first_index, 1);
    first_index = unq_id.masked_select(first_mask);
    first_block_id = th::empty(first_mask.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    array = th::arange(first_index.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    first_block_id.index_copy_(0,first_index,array);
    first_block_id = first_block_id.index_select(0,unq_id).contiguous();
}
