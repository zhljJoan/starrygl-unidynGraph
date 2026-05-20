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
- Current CTDG execution shape:
  - `session.iter_batches(split)` -> optional negative sampling before sampler -> sampler -> async feature/memory/mailbox patch -> `encoder.encode(batch.graph)` -> `head(emb, batch)` -> `task.compute_loss/metrics`.

## Near-Term Migration Priorities
- Write and run a WIKI edge-prediction smoke using the new CTDG loop and report AP/AUC.
- Add default config-driven construction for memory and mailbox runtimes.
- Add memory/mailbox writeback after training steps; keep it as an explicit hook, not hidden inside task adapters.
- Continue DTDG integration separately:
  - `STGraphLoader` should use `partition_data_*.pt` for full-snapshot node tasks.
  - DTDG edge prediction still needs endpoint embedding communication based on master/read routing.
- Once the new CTDG and DTDG paths are stable, remove unused legacy fallback and any task adapter abstraction that is not carrying real behavior.
