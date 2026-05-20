#include <speed_partition.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEps = 1e-8;

struct SpeedNode {
    int64_t id;
    int64_t degree;
    double importance;
};

struct SpeedState {
    int64_t num_nodes;
    int64_t num_edges;
    int64_t num_parts;
    int64_t edge_cap;
    std::vector<double> time_bal;
    std::vector<std::vector<int64_t>> node_parts;
    std::vector<std::vector<uint8_t>> is_in;
    std::vector<int64_t> degree;
    std::vector<int64_t> degree_in;
    std::vector<double> importance;
    std::vector<std::vector<int64_t>> edge_ids_by_part;
    std::vector<int64_t> dropped_edges;
    std::vector<int64_t> edge_load_by_part;
    std::vector<int64_t> node_load_by_part;
    std::vector<std::vector<double>> importance_by_part;
    std::vector<std::vector<double>> last_ts_by_part;

    SpeedState(int64_t n, int64_t e, int64_t p)
        : num_nodes(n),
          num_edges(e),
          num_parts(p),
          edge_cap(static_cast<int64_t>((1.0 + 0.1) * static_cast<double>(e) / std::max<int64_t>(p, 1))),
          time_bal(p, 0.0),
          node_parts(n),
          is_in(n, std::vector<uint8_t>(p, 0)),
          degree(n, 0),
          degree_in(n, 0),
          importance(n, 0.0),
          edge_ids_by_part(p),
          edge_load_by_part(p, 0),
          node_load_by_part(p, 0),
          importance_by_part(n, std::vector<double>(p, 0.0)),
          last_ts_by_part(n, std::vector<double>(p, 0.0)) {}

    void add_edge_to_part(int64_t eid, int64_t part, double ts) {
        edge_load_by_part[part] += 1;
        edge_ids_by_part[part].push_back(eid);
        time_bal[part] = std::max(time_bal[part], ts);
    }

    void add_node_to_part(int64_t nid, int64_t part, double weight, double ts, double max_time) {
        if (!is_in[nid][part]) {
            is_in[nid][part] = 1;
            node_parts[nid].push_back(part);
            node_load_by_part[part] += 1;
        }
        degree[nid] += 1;
        importance[nid] += weight;
        const double denom = std::max(max_time, 1e-9);
        importance_by_part[nid][part] *= std::exp((last_ts_by_part[nid][part] - ts) / denom);
        importance_by_part[nid][part] += weight;
        last_ts_by_part[nid][part] = ts;
    }

    int64_t max_edge_load() const {
        return *std::max_element(edge_load_by_part.begin(), edge_load_by_part.end());
    }

    int64_t min_edge_load() const {
        return *std::min_element(edge_load_by_part.begin(), edge_load_by_part.end());
    }

    int64_t max_node_load() const {
        return *std::max_element(node_load_by_part.begin(), node_load_by_part.end());
    }

    int64_t min_node_load() const {
        return *std::min_element(node_load_by_part.begin(), node_load_by_part.end());
    }

    double max_time_load() const {
        return *std::max_element(time_bal.begin(), time_bal.end());
    }

    double min_time_load() const {
        return *std::min_element(time_bal.begin(), time_bal.end());
    }
};

bool almost_equal(double a, double b) {
    return std::abs(a - b) < kEps;
}

torch::Tensor require_1d_cpu_contiguous(torch::Tensor t, const char* name, torch::ScalarType dtype) {
    TORCH_CHECK(t.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    if (t.scalar_type() != dtype) {
        t = t.to(dtype);
    }
    return t.contiguous();
}

int64_t tensor_num_nodes(const int64_t* src, const int64_t* dst, int64_t num_edges, int64_t requested) {
    int64_t n = requested;
    if (n > 0) {
        return n;
    }
    int64_t max_id = -1;
    for (int64_t i = 0; i < num_edges; ++i) {
        max_id = std::max(max_id, std::max(src[i], dst[i]));
    }
    return max_id + 1;
}

template <typename Ts>
std::vector<double> copy_ts(torch::Tensor ts) {
    const auto* ptr = ts.data_ptr<Ts>();
    std::vector<double> out(ts.numel());
    for (int64_t i = 0; i < ts.numel(); ++i) {
        out[i] = static_cast<double>(ptr[i]);
    }
    return out;
}

std::vector<double> copy_timestamps(torch::Tensor ts) {
    TORCH_CHECK(ts.dim() == 1, "ts must be 1-D");
    TORCH_CHECK(ts.device().is_cpu(), "ts must be a CPU tensor");
    ts = ts.contiguous();
    if (ts.scalar_type() == torch::kFloat32) {
        return copy_ts<float>(ts);
    }
    if (ts.scalar_type() == torch::kFloat64) {
        return copy_ts<double>(ts);
    }
    ts = ts.to(torch::kFloat64).contiguous();
    return copy_ts<double>(ts);
}

void perform_step(
    int64_t u,
    int64_t v,
    int64_t eid,
    double ts,
    const std::vector<uint8_t>& is_hot,
    SpeedState& state,
    const std::vector<std::vector<int64_t>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_ts,
    double max_time,
    double delta_ts,
    int64_t epsilon,
    int64_t epsilon_t) {
    std::vector<int64_t> candidates;
    double best_score = -1.0;

    if (is_hot[u] && !is_hot[v]) {
        candidates.insert(candidates.end(), state.node_parts[v].begin(), state.node_parts[v].end());
    } else if (is_hot[v] && !is_hot[u]) {
        candidates.insert(candidates.end(), state.node_parts[u].begin(), state.node_parts[u].end());
    } else if (!is_hot[u] && !is_hot[v] && !state.node_parts[u].empty() && !state.node_parts[v].empty()) {
        for (int64_t p = 0; p < state.num_parts; ++p) {
            if (state.is_in[u][p] && state.is_in[v][p]) {
                candidates.push_back(p);
            }
        }
    }

    if (candidates.empty()) {
        const int64_t max_edge_load = state.max_edge_load();
        const int64_t min_edge_load = state.min_edge_load();
        const int64_t max_node_load = state.max_node_load();
        const int64_t min_node_load = state.min_node_load();
        const double max_time_load = state.max_time_load();
        const double min_time_load = state.min_time_load();
        const double denom_time = std::max(max_time, 1e-9);

        for (int64_t p = 0; p < state.num_parts; ++p) {
            double score =
                state.importance_by_part[u][p] * std::exp((state.last_ts_by_part[u][p] - ts) / denom_time) +
                state.importance_by_part[v][p] * std::exp((state.last_ts_by_part[v][p] - ts) / denom_time) +
                1.0;

            for (int64_t j = state.degree[u]; j < static_cast<int64_t>(neighbor_ts[u].size()); ++j) {
                if (neighbor_ts[u][j] >= ts && neighbor_ts[u][j] - ts < delta_ts) {
                    const int64_t ni = neighbors[u][j];
                    if ((!state.node_parts[ni].empty() && state.node_parts[ni][0] == p) || is_hot[ni]) {
                        score += std::exp((ts - neighbor_ts[u][j]) / denom_time);
                    }
                }
            }
            for (int64_t j = state.degree[v]; j < static_cast<int64_t>(neighbor_ts[v].size()); ++j) {
                if (neighbor_ts[v][j] >= ts && neighbor_ts[v][j] - ts < delta_ts) {
                    const int64_t ni = neighbors[v][j];
                    if ((!state.node_parts[ni].empty() && state.node_parts[ni][0] == p) || is_hot[ni]) {
                        score += std::exp((ts - neighbor_ts[v][j]) / denom_time);
                    }
                }
            }

            const double node_bal =
                static_cast<double>(state.node_load_by_part[p] - min_node_load) /
                static_cast<double>(epsilon + max_node_load - min_node_load);
            const double edge_bal =
                static_cast<double>(state.edge_load_by_part[p] - min_edge_load) /
                static_cast<double>(epsilon + max_edge_load - min_edge_load);
            const double time_bal = std::exp(
                (min_time_load - state.time_bal[p]) /
                (max_time_load - min_time_load + 1.0 - static_cast<double>(epsilon_t)));

            score = score * (1.0 - node_bal) * (1.0 - edge_bal) * time_bal;
            if (score > best_score + kEps) {
                best_score = score;
                candidates.clear();
                candidates.push_back(p);
            } else if (almost_equal(score, best_score)) {
                candidates.push_back(p);
            }
        }
    }

    state.degree_in[u] += 1;
    state.degree_in[v] += 1;
    if (candidates.empty()) {
        state.dropped_edges.push_back(eid);
        return;
    }

    const int64_t part = candidates[static_cast<size_t>(eid % candidates.size())];
    constexpr double weight = 1.0;
    if (is_hot[u] || state.node_parts[u].empty() || state.node_parts[u][0] == part) {
        state.add_node_to_part(u, part, weight, ts, max_time);
    }
    if (is_hot[v] || state.node_parts[v].empty() || state.node_parts[v][0] == part) {
        state.add_node_to_part(v, part, weight, ts, max_time);
    }
    state.add_edge_to_part(eid, part, ts);
}

torch::Tensor vector_to_long_tensor(const std::vector<int64_t>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::dtype(torch::kInt64));
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), out.data_ptr<int64_t>());
    }
    return out;
}

int64_t first_local_part_or_fallback(
    const SpeedState& state,
    int64_t nid,
    int64_t fallback) {
    if (!state.node_parts[static_cast<size_t>(nid)].empty()) {
        return state.node_parts[static_cast<size_t>(nid)][0];
    }
    return fallback;
}

torch::Tensor choose_balanced_node_master(
    const SpeedState& state,
    const int64_t* src_ptr,
    const int64_t* dst_ptr,
    const int64_t* edge_owner_ptr,
    const std::vector<uint8_t>& is_hot,
    const std::vector<int64_t>& degree) {
    const int64_t n = state.num_nodes;
    const int64_t num_parts = state.num_parts;
    auto node_master = torch::full({n}, -1, torch::dtype(torch::kInt64));
    auto* node_master_ptr = node_master.data_ptr<int64_t>();

    std::vector<double> master_load(static_cast<size_t>(num_parts), 0.0);
    std::vector<std::vector<double>> affinity(
        static_cast<size_t>(n),
        std::vector<double>(static_cast<size_t>(num_parts), 0.0));

    for (int64_t eid = 0; eid < state.num_edges; ++eid) {
        const int64_t part = edge_owner_ptr[eid];
        if (part < 0 || part >= num_parts) {
            continue;
        }
        affinity[static_cast<size_t>(src_ptr[eid])][static_cast<size_t>(part)] += 1.0;
        affinity[static_cast<size_t>(dst_ptr[eid])][static_cast<size_t>(part)] += 1.0;
    }

    std::vector<int64_t> replica_nodes;
    replica_nodes.reserve(static_cast<size_t>(n));
    for (int64_t nid = 0; nid < n; ++nid) {
        const bool replica = is_hot[static_cast<size_t>(nid)] || state.node_parts[static_cast<size_t>(nid)].size() > 1;
        if (replica) {
            replica_nodes.push_back(nid);
            continue;
        }
        const int64_t fallback = nid % std::max<int64_t>(num_parts, 1);
        const int64_t master = first_local_part_or_fallback(state, nid, fallback);
        node_master_ptr[nid] = master;
        master_load[static_cast<size_t>(master)] += static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
    }

    std::sort(replica_nodes.begin(), replica_nodes.end(), [&](int64_t a, int64_t b) {
        if (degree[static_cast<size_t>(a)] != degree[static_cast<size_t>(b)]) {
            return degree[static_cast<size_t>(a)] > degree[static_cast<size_t>(b)];
        }
        return a < b;
    });

    double total_weight = 0.0;
    for (int64_t nid = 0; nid < n; ++nid) {
        total_weight += static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
    }
    const double target_load = std::max(total_weight / static_cast<double>(std::max<int64_t>(num_parts, 1)), 1.0);

    for (int64_t nid : replica_nodes) {
        const double node_weight = static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
        double affinity_sum = 0.0;
        for (int64_t part = 0; part < num_parts; ++part) {
            affinity_sum += affinity[static_cast<size_t>(nid)][static_cast<size_t>(part)];
        }
        const double load_penalty = std::max(1.0, affinity_sum);
        int64_t best_part = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int64_t part = 0; part < num_parts; ++part) {
            const bool candidate = is_hot[static_cast<size_t>(nid)] || state.is_in[static_cast<size_t>(nid)][static_cast<size_t>(part)];
            if (!candidate) {
                continue;
            }
            const double score =
                affinity[static_cast<size_t>(nid)][static_cast<size_t>(part)] -
                load_penalty * (master_load[static_cast<size_t>(part)] / target_load);
            if (score > best_score + kEps || (almost_equal(score, best_score) && part < best_part)) {
                best_score = score;
                best_part = part;
            }
        }
        node_master_ptr[nid] = best_part;
        master_load[static_cast<size_t>(best_part)] += node_weight;
    }

    return node_master;
}

}  // namespace

py::dict speed_partition(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor ts,
    int64_t num_nodes,
    int64_t num_parts,
    double beta,
    double topk_ratio,
    const std::string& topk_type) {
    TORCH_CHECK(num_parts > 0, "num_parts must be positive");
    src = require_1d_cpu_contiguous(src, "src", torch::kInt64);
    dst = require_1d_cpu_contiguous(dst, "dst", torch::kInt64);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same length");
    std::vector<double> ts_values = copy_timestamps(ts);
    TORCH_CHECK(static_cast<int64_t>(ts_values.size()) == src.numel(), "ts must match src/dst length");

    const auto* src_ptr = src.data_ptr<int64_t>();
    const auto* dst_ptr = dst.data_ptr<int64_t>();
    const int64_t num_edges = src.numel();
    const int64_t n = tensor_num_nodes(src_ptr, dst_ptr, num_edges, num_nodes);
    TORCH_CHECK(n >= 0, "num_nodes must be non-negative");

    double max_ts = 0.0;
    for (double t : ts_values) {
        max_ts = std::max(max_ts, t);
    }
    const double raw_max_time = std::max(max_ts, 1e-9);

    std::vector<int64_t> degree(static_cast<size_t>(n), 0);
    std::vector<double> importance(static_cast<size_t>(n), 0.0);
    std::vector<std::vector<int64_t>> neighbors(static_cast<size_t>(n));
    std::vector<std::vector<double>> neighbor_ts(static_cast<size_t>(n));
    double total_delta_t = 0.0;

    for (int64_t i = 0; i < num_edges; ++i) {
        const int64_t u = src_ptr[i];
        const int64_t v = dst_ptr[i];
        TORCH_CHECK(u >= 0 && u < n && v >= 0 && v < n, "src/dst contains node id outside [0, num_nodes)");
        const double t = ts_values[static_cast<size_t>(i)];
        total_delta_t += i > 0 ? t - ts_values[static_cast<size_t>(i - 1)] : 0.0;
        neighbors[u].push_back(v);
        neighbors[v].push_back(u);
        neighbor_ts[u].push_back(t);
        neighbor_ts[v].push_back(t);
        degree[u] += 1;
        degree[v] += 1;
        importance[u] += beta * (t / raw_max_time);
        importance[v] += beta * (t / raw_max_time);
    }

    std::vector<SpeedNode> nodes;
    nodes.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        nodes.push_back(SpeedNode{i, degree[i], importance[i]});
    }

    const int64_t topk = std::clamp<int64_t>(
        static_cast<int64_t>(std::floor(topk_ratio * static_cast<double>(n))),
        0,
        n);
    const bool use_importance = topk_type == "importance";
    std::sort(nodes.begin(), nodes.end(), [&](const SpeedNode& a, const SpeedNode& b) {
        if (use_importance && !almost_equal(a.importance, b.importance)) {
            return a.importance > b.importance;
        }
        if (a.degree != b.degree) {
            return a.degree > b.degree;
        }
        return a.id < b.id;
    });

    std::vector<uint8_t> is_hot(static_cast<size_t>(n), 0);
    std::vector<int64_t> hot_node_ids;
    hot_node_ids.reserve(static_cast<size_t>(topk));
    for (int64_t i = 0; i < topk; ++i) {
        is_hot[nodes[static_cast<size_t>(i)].id] = 1;
        hot_node_ids.push_back(nodes[static_cast<size_t>(i)].id);
    }

    const double avg_delta_t = num_edges > 1 ? total_delta_t / static_cast<double>(num_edges - 1) : 0.0;
    const int64_t epsilon_t = static_cast<int64_t>(avg_delta_t * total_delta_t * 2.0);
    const int64_t epsilon = 20;
    const double delta_ts = avg_delta_t * static_cast<double>(epsilon);
    const double partition_max_time = raw_max_time / 10.0;

    SpeedState state(n, num_edges, num_parts);
    for (int64_t i = 0; i < num_edges; ++i) {
        perform_step(
            src_ptr[i],
            dst_ptr[i],
            i,
            ts_values[static_cast<size_t>(i)],
            is_hot,
            state,
            neighbors,
            neighbor_ts,
            partition_max_time,
            delta_ts,
            epsilon,
            epsilon_t);
    }

    auto edge_owner = torch::full({num_edges}, -1, torch::dtype(torch::kInt64));
    auto edge_owner_ptr = edge_owner.data_ptr<int64_t>();
    std::vector<torch::Tensor> edge_ids_by_part;
    edge_ids_by_part.reserve(static_cast<size_t>(num_parts));
    for (int64_t p = 0; p < num_parts; ++p) {
        for (int64_t eid : state.edge_ids_by_part[p]) {
            edge_owner_ptr[eid] = p;
        }
        edge_ids_by_part.push_back(vector_to_long_tensor(state.edge_ids_by_part[p]));
    }
    for (int64_t eid = 0; eid < num_edges; ++eid) {
        if (edge_owner_ptr[eid] < 0) {
            edge_owner_ptr[eid] = first_local_part_or_fallback(state, dst_ptr[eid], dst_ptr[eid] % num_parts);
        }
    }

    auto node_master = choose_balanced_node_master(state, src_ptr, dst_ptr, edge_owner_ptr, is_hot, degree);
    auto node_master_ptr = node_master.data_ptr<int64_t>();
    auto replica_mask = torch::zeros({n}, torch::dtype(torch::kBool));
    auto replica_ptr = replica_mask.data_ptr<bool>();
    std::vector<torch::Tensor> local_node_ids_by_part;
    std::vector<torch::Tensor> owned_node_ids_by_part;
    std::vector<torch::Tensor> replica_node_ids_by_part;
    std::vector<torch::Tensor> shadow_node_ids_by_part;
    local_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    owned_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    replica_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    shadow_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    for (int64_t nid = 0; nid < n; ++nid) {
        if (is_hot[static_cast<size_t>(nid)] || state.node_parts[static_cast<size_t>(nid)].size() > 1) {
            replica_ptr[nid] = true;
        }
    }
    for (int64_t p = 0; p < num_parts; ++p) {
        std::vector<int64_t> replica_nodes;
        std::vector<int64_t> owned_nodes;
        std::vector<int64_t> shadow_nodes;
        for (int64_t nid = 0; nid < n; ++nid) {
            const bool replica = replica_ptr[nid];
            const bool held = state.is_in[static_cast<size_t>(nid)][static_cast<size_t>(p)] || replica;
            if (!held) {
                continue;
            }
            if (replica) {
                replica_nodes.push_back(nid);
            } else if (node_master_ptr[nid] == p) {
                owned_nodes.push_back(nid);
            } else {
                shadow_nodes.push_back(nid);
            }
        }
        std::vector<int64_t> local_nodes;
        local_nodes.reserve(replica_nodes.size() + owned_nodes.size() + shadow_nodes.size());
        local_nodes.insert(local_nodes.end(), replica_nodes.begin(), replica_nodes.end());
        local_nodes.insert(local_nodes.end(), owned_nodes.begin(), owned_nodes.end());
        local_nodes.insert(local_nodes.end(), shadow_nodes.begin(), shadow_nodes.end());
        local_node_ids_by_part.push_back(vector_to_long_tensor(local_nodes));
        owned_node_ids_by_part.push_back(vector_to_long_tensor(owned_nodes));
        replica_node_ids_by_part.push_back(vector_to_long_tensor(replica_nodes));
        shadow_node_ids_by_part.push_back(vector_to_long_tensor(shadow_nodes));
    }

    py::dict out;
    out["node_master"] = node_master;
    out["node_parts"] = node_master;
    out["node_owner"] = node_master;
    out["edge_owner"] = edge_owner;
    out["edge_parts"] = edge_owner;
    out["replica_mask"] = replica_mask;
    out["hot_node_ids"] = vector_to_long_tensor(hot_node_ids);
    out["local_node_ids_by_part"] = local_node_ids_by_part;
    out["owned_node_ids_by_part"] = owned_node_ids_by_part;
    out["replica_node_ids_by_part"] = replica_node_ids_by_part;
    out["shadow_node_ids_by_part"] = shadow_node_ids_by_part;
    out["edge_ids_by_part"] = edge_ids_by_part;
    out["dropped_edges"] = vector_to_long_tensor(state.dropped_edges);
    return out;
}
