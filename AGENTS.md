# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `src/atc_starrygl_lib`.
- Major domains:
  - `preprocess/`: dataset, partition, rank, feature, and pipeline artifact generation.
  - `ctdg/` and `dtdg/`: mode-specific runtime and preprocess integration.
  - `models/`: CTDG/DTDG/shared model components.
  - `sampling/`, `memory/`, `comm/`, `runtime/`, `tasks/`: execution path utilities.
- Native code is under `csrc/` (sampler/partition logic), with build config in `CMakeLists.txt`.
- Tests are in `tests/`, named `test_*.py`.

## Build, Test, and Development Commands
- Install editable package:
  - `pip install -e .`
- Run unit tests:
  - `python -m pytest tests`
- Run targeted tests during development:
  - `python -m pytest tests/test_preprocess_dataset.py`
- Run DTDG STGraphLoader tests:
  - `python -m pytest tests/test_dtdg_stgraph_loader.py`
- Run single-process DTDG STGraph smoke:
  - `python tools/dtdg_stgraph_smoke.py --artifact-root /tmp/atc_dtdg_stgraph_smoke --epochs 2`
- Run the unified config entry:
  - Single-machine preprocessing: `python tools/atc_run.py prepare --config configs/dtdg_tgcn_node_regression.json --artifact-root /shared/artifacts/wiki`
  - Multi-GPU train/eval: `torchrun --standalone --nproc_per_node=2 tools/atc_run.py run --no-prepare --config configs/dtdg_tgcn_node_regression.json --artifact-root /shared/artifacts/wiki --epochs 1`
- Compile-check Python modules:
  - `python -m py_compile src/atc_starrygl_lib/preprocess/*.py`
- Build native sampler library (when C++ changes):
  - `cmake -S . -B build && cmake --build build -j`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, type hints preferred for public functions.
- Naming:
  - modules/files: `snake_case.py`
  - functions/variables: `snake_case`
  - classes: `PascalCase`
  - constants: `UPPER_SNAKE_CASE`
- Keep artifact field names stable (`time_ptr_2`, `split_time_ptr`, `read_dist_index`, etc.) to avoid runtime incompatibility.
- Prefer small, composable helpers over large monolithic functions.

## Testing Guidelines
- Framework: `pytest`.
- New behavior must include focused tests in `tests/` with deterministic tensors/small fixtures.
- For preprocess changes, validate both content and shape/ordering (e.g., stable sort, split windows, index mapping).
- Run related test files before opening a PR; run full `tests/` for cross-module changes.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative, and scoped (e.g., `preprocess: add split_time_ptr for event mode`).
- PRs should include:
  - what changed and why,
  - affected artifacts/interfaces,
  - test commands run and results,
  - migration notes if formats/config keys changed.
- For runtime/preprocess contract updates, explicitly call out backward compatibility and fallback behavior.

## Configuration & Migration Notes
- Prefer config-driven behavior (`preprocess.*`) over hardcoded paths.
- User configs should not expose CTDG/DTDG backend names. `model.*` and `sampling.*` drive `runtime.execution_plan`:
  - `temporal_sampling`: internal CTDG/MemShare-style sampled execution.
  - `snapshot_full_graph`: internal DTDG/Flare-style STGraphLoader execution.
  - Sampling config has priority; a DTDG-family model with neighbor sampling routes to `temporal_sampling`.
  - Prefer user-facing `gnn.*` for execution-shape settings:
    - `gnn.sampling.fanouts`, `gnn.sampling.policy`, and optional `gnn.sampling.probability` for sampled execution.
    - `gnn.full_graph` / `gnn.slice_config` for full-snapshot settings such as `chunk_order`, `chunk_decay`, and `num_full_snapshots`.
    - `gnn.history` is the default temporal history and should match full-graph `num_full_snapshots` unless intentionally overridden.
    - `gnn.memory_update` and `gnn.memory_history` configure CTDG memory update style and mailbox history depth.
- For event mode, preserve rule: split train/val/test first, then batch per split.
- Keep `time_ptr_2` compatibility when introducing newer structures like `split_time_ptr`.

## Current Migration Status
- Preprocess event pipeline is the main migration path:
  - `dataset -> dist -> rank -> feature -> partition_data` artifacts are available.
  - `split_time_ptr` is the preferred split-aware event window contract.
  - `time_ptr_2` remains only for compatibility.
  - Rank-local `split_time_ptr` keeps the same window semantics but remaps indices after local event filtering.
- CTDG new-pipeline runtime reader is wired:
  - Loads `graph.pt`, `dist.pt`, `rank_XXX.pt`, and optional `feature_XXX.pt`.
  - `iter_batches(split)` supports edge event batches and node label batches.
  - Edge batches expose stable fields: `eids`, `src`, `dst`, `ts`, `roots`, `timestamps`, `pos_src`, `pos_dst`, optional `neg_dst`.
  - Node batches expose `roots`, `timestamps`, `node_ids`, and `labels`.
- CTDG sampling path is MemShare-compatible:
  - Runtime can build or accept a sampler.
  - DGL blocks are materialized directly from CSC via `dgl.create_block(("csc", ...))`; do not switch back to COO conversion.
  - Block fields follow the current contract: `srcdata["__ID"]`, `dstdata["__ID"]` are compute row ids; `srcdata["ID"]`, `dstdata["ID"]` are global node ids; `edata["__ID"]` is edge compute row id; `edata["ID"]` is global edge id.
  - `SampleOutput` / `RootSet.groups` drive remapping of `pos_src`, `pos_dst`, and `neg_dst` to embedding row indices.
  - Native sampled export now prefers `get_sampling_output_compact()` over the older full `get_sampling_output()` layout build when the sampler exposes it.
  - `get_sampling_output_compact()` stays on the C++ path and returns the same `NativeSamplingOutput` contract, but its hot path is narrowed to root initialization, layer-local source/edge indexing, and CSC fill.
  - Native compact profile stats are exposed through the runtime profile stream: `compact_total_seconds`, `compact_root_seconds`, `compact_index_seconds`, and `compact_fill_seconds`.
  - Current best verified CTDG sampled implementation still keeps per-layer node and edge dedup in the compact builder; removing node dedup inflated encode time and regressed end-to-end runtime, so that experiment should not be revived blindly.
- CTDG feature and state prefetch:
  - Node feature, edge feature, memory, and mailbox reads are submitted after sampling and patched before yielding the batch.
  - Node feature patches `srcdata["h"]`; edge feature patches `edata["f"]`; memory patches `srcdata["mem"]` / `srcdata["mem_ts"]`; mailbox patches `srcdata["mem_input"]` / `srcdata["mail_ts"]`.
  - Keep the two-level pipeline: prefetch the next sampled batch while asynchronously fetching features/state for the current sampled batch.
- CTDG minimal training loop is available:
  - `GeneralModel.encode(mfgs) -> emb` is the preferred encoder interface.
  - Heads consume `head(emb, batch) -> output`.
  - Task classes should stay thin: use them for `compute_loss()` and `compute_metrics()` only.
  - Avoid introducing heavy task adapter wrappers around batch prep, model forward, head, and metrics.
  - Edge prediction negative sampling stays as a separate pre-sampling batch transform; it must extend `roots/timestamps` before sampler execution so `neg_dst` can be remapped to embedding rows.
  - Memory/mailbox writeback is an explicit post-step hook via `CTDGMemoryCommitHook`; keep writeback outside task adapters.
  - Runtime can construct default sampler, feature runtime, memory runtime, and mailbox runtime from config when requested.
- Current CTDG execution shape:
  - `session.iter_batches(split)` -> optional negative sampling before sampler -> sampler -> async feature/memory/mailbox patch -> `encoder.encode(batch.graph)` -> `head(emb, batch)` -> `task.compute_loss/metrics`.
- DTDG new-pipeline runtime is wired for node snapshot training:
  - `FlareDTDGBackend` can run the new snapshot preprocess path and load `partition_data_XXX.pt`.
  - `STGraphLoader` materializes DGL blocks from `partition_data` with `srcdata["ID"]`, `dstdata["ID"]`, `srcdata["__ID"]`, `dstdata["__ID"]`, `edata["ID"]`, and `edata["__ID"]`.
  - Node features are patched as `srcdata["x"]`; node labels are patched as `dstdata["y"]`; edge weights/norms/features are patched under `edata`.
  - `iter_batches("train")` can yield `STGraphWindow` objects for sliding-window DTDG models; eval/test yield single snapshot graphs.
  - Sliding-window state hooks are patched as `flare_fetch_state()` / `flare_store_state()`, and route hooks as `flare_apply_route()` / `flare_async_route()`.
  - `dtdg.train_loop.train_epoch/evaluate` support recurrent node regression/classification style models such as `TGCN`.
  - `tools/dtdg_stgraph_smoke.py` runs a single-process synthetic or `.pth`-backed node-regression smoke through prepare, runtime build, train, val, and test.
- DTDG edge prediction is partially wired:
  - Backend can emit edge prediction batches with `pos_src`, `pos_dst`, optional `neg_dst`, `src`, `dst`, and `eids`.
  - `train_edge_prediction_epoch()` / `evaluate_edge_prediction()` support encoder/head training and evaluation on these batches.
  - Distributed endpoint embedding communication uses route send/recv metadata and preserves gradient propagation through `Route`.
  - Node regression/classification does not do endpoint lookup; labels/features are read from `partition_data` and attached to dst rows.
- Unified runtime entry is available:
  - `tools/atc_run.py prepare` is the single-machine preprocessing entry; copy the resulting artifact directory to workers.
  - `tools/atc_run.py run --no-prepare` is the multi-GPU train/val/test entry.
  - `tools/atc_run.py eval` and `tools/atc_run.py predict` reuse the same config/runtime.
  - `predict` can update CTDG memory between test batches via `runtime.predict_updates_memory`; outputs are emitted before the memory commit.

## Current Performance Status
- CTDG/WikiTalk sampled route (`configs/ctdg_wikitalk_tgn_speed_hot01_bs12000.json`, 4 GPU, unified entry) is the current performance focus.
- The large Python-side batch build hotspot has already been removed:
  - Training windows (`split_event_pos`) are precomputed in preprocess.
  - Runtime caches graph-side tensors for batch materialization instead of redoing repeated `.cpu()`/dtype conversions.
  - Memory commit waiting is moved out of the per-batch hook so it can overlap with the next batch.
- Native sampler compact export is now the main sampled hot path:
  - Before compact optimization, steady-state `backend_get_sampling_output_seconds` was around `6.2s+` and `backend_sampling_native_seconds` around `8.0s+`.
  - With the current compact builder and profile-guided caching, steady-state `backend_get_sampling_output_seconds` is around `4.1s`, and `backend_sampling_native_seconds` around `5.8s`.
  - Recent verified 3-epoch WikiTalk run showed train seconds around `16.00s`, `15.89s`, and `15.88s`.
- Current dominant remaining native hotspot inside the compact builder is `compact_index_seconds` at roughly `3.3s`; `compact_fill_seconds` is already small (`~0.15s`).
- A more aggressive experiment that removed per-layer source dedup reduced compact indexing work but caused much larger sampled blocks and pushed `stage_encode_seconds` up sharply, regressing end-to-end runtime; keep the current compact implementation instead.

## Near-Term Migration Priorities
- Consolidate all smoke scripts onto `tools/atc_run.py` after model/task coverage reaches parity.
- Productionize sampled-block support for DTDG-family temporal-sampling models beyond the current GCN encoder.
- Replace remaining preprocessing Python per-node/per-update loops in rank memory route construction with vectorized/native builders.
- Continue CTDG sampled native optimization around `compact_index_seconds`; prioritize better layer-local `(node, ts)` / edge indexing strategies before touching CSC fill or reintroducing Python tensor dedup.
- Once the new CTDG and DTDG paths are stable, remove unused legacy fallback and any task adapter abstraction that is not carrying real behavior.
