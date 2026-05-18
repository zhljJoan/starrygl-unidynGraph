# Chunk Update Alignment Notes

## Current Chunk Native Smoke Results

Run date: 2026-05-18.

Dataset: WIKI from `/mnt/data/zlj/starrygl-data/ctdg/WIKI.pth`.

Shared config aligned with MemShare `TGN_large.yml` where applicable:

- fanout: 20
- batch size: 3000
- learning rate: 0.0004
- model family/task: TGN temporal link prediction
- native sampler: MemShare BTS sampler through `MemShareEventEngine`
- temporal index: prepare-time `sampling/temporal_index_part_<rank>.pth`
- distributed backend: NCCL for multi-GPU

### Chunk Native, 1 GPU

Command:

```bash
python examples/bench_chunk_native.py \
  --dataset WIKI \
  --mode all \
  --device cuda \
  --epochs 1 \
  --batch-size 3000 \
  --hidden-dim 32 \
  --num-windows 8 \
  --artifact-root /tmp/starry_chunk_bench_wiki_cuda1
```

Result:

```text
prepare_s: 0.3745
build_runtime_s: 6.9681
train.wall_s: 8.4242
train.steps: 318
train.native_sampling_steps: 318
```

### Chunk Native, 4 GPU NCCL

Command:

```bash
torchrun --nproc_per_node=4 examples/bench_chunk_native.py \
  --dataset WIKI \
  --mode all \
  --device cuda \
  --epochs 1 \
  --batch-size 3000 \
  --hidden-dim 32 \
  --num-windows 8 \
  --artifact-root /tmp/starry_chunk_bench_wiki_cuda4
```

Result:

```text
prepare_s: 0.5650
build_runtime_s: 0.2243
train.wall_s: 4.5728
train.steps: 594
rank0 local_steps: 144
rank0 native_sampling_steps: 144
```

The `libibverbs` warnings observed during the 4-GPU run did not prevent NCCL
initialization or training.

## Baseline Commands To Keep Comparable

MemShare reference config:

```text
/home/zlj/MemShare-public/MemShare/config/TGN_large.yml
```

Key parameters in that config:

```text
neighbor: [20]
batch_size: 3000
lr: 0.0004
dim_time: 100
dim_out: 100
mailbox_size: 1
```

MemShare's `examples/test_all.sh` currently contains unresolved conflict
markers, so use direct `torchrun` commands against `train_boundery.py` instead
of invoking that script as-is.

FlareDTDG reference entry points:

```text
/home/zlj/FlareDTDG/run_flare2.py
/home/zlj/FlareDTDG/run_flare2.sh
/home/zlj/FlareDTDG/run_dyna2.py
/home/zlj/FlareDTDG/run_dyna2.sh
```

FlareDTDG is DTDG-oriented; compare only when the task/model/data mode is
matched. For CTDG TGN comparison, MemShare is the direct baseline.

## Remaining Alignment Work

- Run WIKI/WikiTalk MemShare `TGN_large` baselines with the same fanout,
  batch size, hidden dimension, learning rate, and GPU count.
- Run WikiTalk chunk native 1-GPU and 4-GPU smoke once the longer dataset run
  budget is available.
- Move sampled MFG dedup and remote-read route construction fully into native
  code; current training uses the native sampler, while the Python side still
  materializes positive/negative edge tensors and metadata.
- Integrate a real TGN memory updater that emits `memory_update` so
  MemShare-style change-rate communication is exercised during normal model
  training, not only through the `submit_memory_update` API tests.

## 2026-05-18 Follow-up Alignment Notes

Implemented in this pass:

- CTDG sampled-unit base comm plans no longer attach prebuilt
  `spatial_routes`; fetch is represented only after sampling through
  `FetchPlan.remote_read_index`.
- `FetchPlan`, `SpatialRouteData`, and `MemoryRouteData` now carry optional
  packed DistIndex fields while keeping global-id fields for compatibility
  with existing artifacts.
- Runtime dynamic CTDG fetch plans are cached by packed `remote_read_index`.
- `edge_predict_mixed` negative sampling is wired with train local-biased
  destination pools, eval/test `global_average`, and optional `neg_weight`.
- DTDG `ChunkPropagationRoute` supports `append_recv` and `recv_src_rows`;
  unit tests cover remote-row reorder and backward scatter semantics.
- `CommPipeline.submit_*` now returns a `CommHandle`; callers can use
  `await_handle(handle)` while old channel-specific awaits remain available.

Verification:

```text
python -m pytest \
  starry_unigraph/backends/chunk/data/test_route.py \
  starry_unigraph/backends/chunk/data/test_edge_index_conversion.py \
  starry_unigraph/backends/chunk/data/test_propagation_route.py \
  starry_unigraph/backends/chunk/runtime/test_runtime.py -q

31 passed
```

WIKI CPU real-data smoke with `edge_predict_mixed`:

```text
artifact_root: /tmp/starry_chunk_bench_wiki_cpu_align
train.wall_s: 9.9096
train.steps: 318
train.native_sampling_steps: 318
```

CUDA/NCCL verification was not rerun in this pass because the required
escalated command was rejected by the user. The previously recorded 1-GPU and
4-GPU WIKI NCCL results above remain the latest successful GPU smoke in this
project note.
