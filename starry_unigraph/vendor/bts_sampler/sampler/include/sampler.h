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
        template<typename T>
        void union_to_vector(vector<T> *p, vector<T> &to_vec);
        void sample_unique(     th::Tensor seed, th::Tensor seed_ts,
                                th::Tensor nid_mapper, th::Tensor eid_mapper,string out_device);
};



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique)
{
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform"|| policy =="boundery_recent_decay"||policy=="boundery_recent_uniform"||policy=="boundery_uniform");
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

        double s_start_time = omp_get_wtime();
        
        if ((policy == "recent") || (end_index <= fanout)&& policy.substr(0,8) != "boundery" ){
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
            uniform_int_distribution<> u(0, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int i=0; i<fanout;i++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = rand_r(&loc_seeds[tid]) % (end_index);
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
