#pragma once
#include <head.h>

class TemporalGraphBlock
{
    public:
        vector<NodeIDType> row;
        vector<NodeIDType> col;
        vector<EdgeIDType> eid;
        vector<TimeStampType> delta_ts;
        vector<int64_t> src_index;
        vector<NodeIDType> sample_nodes;
        vector<TimeStampType> sample_nodes_ts;
        vector<float> sample_weight; 
        vector<WeightType> e_weights;
        double sample_time = 0;
        double tot_time = 0;
        int64_t sample_edge_num = 0;

        TemporalGraphBlock(){}
        // TemporalGraphBlock(const TemporalGraphBlock &tgb);
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes,
                           vector<TimeStampType> &_sample_nodes_ts):
                           row(_row), col(_col), sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts){}
};
