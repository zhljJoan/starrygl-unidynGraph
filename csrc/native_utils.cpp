#include <torch/extension.h>

#include <parallel_hashmap/phmap.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct PairKey {
  int64_t value;
  int64_t ts;

  bool operator==(const PairKey& other) const {
    return value == other.value && ts == other.ts;
  }
};

struct PairHash {
  size_t operator()(const PairKey& key) const {
    const uint64_t a = static_cast<uint64_t>(key.value);
    const uint64_t b = static_cast<uint64_t>(key.ts);
    return static_cast<size_t>(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
  }
};

torch::Tensor contiguous_long_cpu(torch::Tensor x, const char* name) {
  if (x.device().is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CPU tensor");
  }
  return x.to(torch::kLong).contiguous().view({-1});
}

}  // namespace

std::vector<torch::Tensor> stable_unique(torch::Tensor values) {
  auto vals = contiguous_long_cpu(values, "values");
  const auto n = vals.numel();
  auto* val_ptr = vals.data_ptr<int64_t>();

  phmap::flat_hash_map<int64_t, int64_t> seen;
  seen.reserve(static_cast<size_t>(n));

  std::vector<int64_t> unique;
  unique.reserve(static_cast<size_t>(n));
  auto inverse = torch::empty({n}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto* inv_ptr = inverse.data_ptr<int64_t>();

  for (int64_t i = 0; i < n; ++i) {
    const int64_t value = val_ptr[i];
    auto it = seen.find(value);
    if (it == seen.end()) {
      const int64_t row = static_cast<int64_t>(unique.size());
      seen.emplace(value, row);
      unique.push_back(value);
      inv_ptr[i] = row;
    } else {
      inv_ptr[i] = it->second;
    }
  }

  auto out = torch::empty({static_cast<int64_t>(unique.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  if (!unique.empty()) {
    std::memcpy(out.data_ptr<int64_t>(), unique.data(), unique.size() * sizeof(int64_t));
  }
  return {out.to(values.scalar_type()), inverse};
}

std::vector<torch::Tensor> stable_unique_with_ts(torch::Tensor values, torch::Tensor ts) {
  auto vals = contiguous_long_cpu(values, "values");
  auto times = contiguous_long_cpu(ts, "ts");
  const auto n = vals.numel();
  if (times.numel() != n) {
    throw std::runtime_error("values and ts must have the same number of elements");
  }
  auto* val_ptr = vals.data_ptr<int64_t>();
  auto* ts_ptr = times.data_ptr<int64_t>();

  phmap::flat_hash_map<PairKey, int64_t, PairHash> seen;
  seen.reserve(static_cast<size_t>(n));

  std::vector<int64_t> unique;
  unique.reserve(static_cast<size_t>(n));
  std::vector<int64_t> first_ts;
  first_ts.reserve(static_cast<size_t>(n));
  auto inverse = torch::empty({n}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto* inv_ptr = inverse.data_ptr<int64_t>();

  for (int64_t i = 0; i < n; ++i) {
    const PairKey key{val_ptr[i], ts_ptr[i]};
    auto it = seen.find(key);
    if (it == seen.end()) {
      const int64_t row = static_cast<int64_t>(unique.size());
      seen.emplace(key, row);
      unique.push_back(key.value);
      first_ts.push_back(key.ts);
      inv_ptr[i] = row;
    } else {
      inv_ptr[i] = it->second;
    }
  }

  auto uniq = torch::empty({static_cast<int64_t>(unique.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto first = torch::empty({static_cast<int64_t>(first_ts.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  if (!unique.empty()) {
    std::memcpy(uniq.data_ptr<int64_t>(), unique.data(), unique.size() * sizeof(int64_t));
    std::memcpy(first.data_ptr<int64_t>(), first_ts.data(), first_ts.size() * sizeof(int64_t));
  }
  return {uniq.to(values.scalar_type()), inverse, first.to(ts.scalar_type())};
}

torch::Tensor first_ts_for_lids(torch::Tensor root_ts, torch::Tensor root_lids, int64_t size) {
  auto times = contiguous_long_cpu(root_ts, "root_ts");
  auto lids = contiguous_long_cpu(root_lids, "root_lids");
  const auto n = lids.numel();
  if (times.numel() != n) {
    throw std::runtime_error("root_ts and root_lids must have the same number of elements");
  }
  auto out = torch::zeros({size}, torch::dtype(torch::kLong).device(torch::kCPU));
  std::vector<uint8_t> filled(static_cast<size_t>(size), 0);
  auto* out_ptr = out.data_ptr<int64_t>();
  auto* ts_ptr = times.data_ptr<int64_t>();
  auto* lid_ptr = lids.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t lid = lid_ptr[i];
    if (lid < 0 || lid >= size) {
      throw std::runtime_error("root_lids contains an out-of-range row");
    }
    if (!filled[static_cast<size_t>(lid)]) {
      out_ptr[lid] = ts_ptr[i];
      filled[static_cast<size_t>(lid)] = 1;
    }
  }
  return out.to(root_ts.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stable_unique", &stable_unique, "Stable unique values with inverse (CPU)");
  m.def("stable_unique_with_ts", &stable_unique_with_ts, "Stable unique (value, ts) pairs with inverse and first ts (CPU)");
  m.def("first_ts_for_lids", &first_ts_for_lids, "First timestamp per local id (CPU)");
}
