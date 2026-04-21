# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StarryUniGraph** is a unified high-performance training system for large-scale dynamic graphs supporting both **CTDG** (Continuous-Time Dynamic Graphs) and **DTDG** (Discrete-Time Dynamic Graphs) under a single distributed runtime.

Key principle: Three completely independent code paths (CTDG, DTDG, Chunk) with no code sharing. Single dispatch point in `SchedulerSession` routes by graph_mode.

## Architecture (High-Level)

The codebase is organized around three independent graph processing modes:

### Three Independent Paths

1. **CTDG (Continuous-Time)** — `backends/ctdg/`
   - Online event-driven processing with temporal neighbors
   - Components: CTDGPreprocessor, CTDGSession, online runtime (data, memory, sampler, models)
   - Models: TGN, MemShare (event stream + memory banks)
   - No code sharing with DTDG or Chunk

2. **DTDG (Discrete-Time with Flare)** — `backends/dtdg/`
   - Snapshot-based processing with multi-GPU support (Flare architecture)
   - Components: FlareDTDGPreprocessor, FlareRuntimeLoader, Flare runtime (state, session_loader, training)
   - Models: MPNN-LSTM, snapshot-aware GNNs
   - No code sharing with CTDG or Chunk

3. **Chunk** — `runtime/chunk/` + `preprocess/chunk.py`
   - Independent chunk-based processing pathway (under development)
   - Will be ready for implementation after CTDG/DTDG migration

### Unified Dispatch Point

**`SchedulerSession`** (`session.py`) is the single entry point:
```python
SchedulerSession.from_config(config) → SessionContext
SchedulerSession.build_runtime() → dispatches by graph_mode:
  - "ctdg" → CTDGSession
  - "dtdg" → FlareRuntimeLoader
  - "chunk" → ChunkRuntimeLoader (future)
```

The dispatch is minimal: `prepared.provider_meta["graph_mode"]` determines which path to take.

### Unified Interfaces (but separate implementations)

- **Preprocessor Protocol**: `prepare_data() → PreparedArtifacts`
- **Training Step Functions**: `train_step()`, `eval_step()`, `predict_step()`
- **Runtime State**: `RuntimeBundle` holds model, optimizer, scheduler

## Recent Migration (April 21, 2026)

CTDG and DTDG components have been moved to isolated backend directories to clear the main `starry_unigraph/` structure for new implementations:

- **Moved to `backends/ctdg/`**: All online runtime components + CTDGPreprocessor
- **Moved to `backends/dtdg/`**: All Flare runtime components + FlareDTDGPreprocessor
- **New `backends/dtdg/types.py`**: Holds `SnapshotRoutePlan` and `DTDGPartitionBook` to break circular imports
- **Updated imports**: All references in `session.py`, `data/partition.py`, and chunk runtime now use backend paths
- **Backward compatibility**: `backends/flare/__init__.py` re-exports for legacy code

**Key file to understand the dispatch**: `starry_unigraph/session.py` (lines 75-134 show the graph_mode branching)

## Directory Structure (Simplified)

```
starry_unigraph/
├── backends/
│   ├── ctdg/              # CTDG (online) — completely independent
│   │   ├── preprocess.py
│   │   ├── runtime/       (data, memory, sampler, factory, etc.)
│   │   └── __init__.py
│   ├── dtdg/              # DTDG (Flare) — completely independent
│   │   ├── preprocess.py, dtdg_prepare.py
│   │   ├── types.py       (schema: SnapshotRoutePlan, DTDGPartitionBook)
│   │   ├── runtime/       (state, session_loader, training, models, etc.)
│   │   └── __init__.py
│   ├── chunk/             # Chunk preprocessing (reference, not yet active)
│   ├── flare/             # Backward compatibility re-exports
│   └── __init__.py
├── runtime/
│   ├── chunk/             # Chunk runtime (ready for implementation)
│   ├── modules/           # Base neural network utilities
│   ├── route/             # Routing utilities
│   └── store/             # Storage utilities
├── preprocess/
│   ├── chunk.py           # Chunk preprocessing (independent)
│   ├── base.py            (GraphPreprocessor protocol)
│   └── __init__.py
├── data/
│   ├── partition.py       (PartitionData, RouteData classes)
│   ├── chunk_data.py      (ChunkAtomic reference implementation)
│   ├── raw_temporal.py    (Event loading utilities)
│   └── ...
├── session.py             # MAIN ENTRY: SchedulerSession unified dispatch
├── types.py               (PreparedArtifacts, RuntimeBundle, SessionContext)
├── distributed.py
├── checkpoint.py
└── ...
```

## Development Workflows

### Running a Single Training Example

```bash
# 1. Prepare data (one-time, creates artifacts/)
python -c "
from starry_unigraph import SchedulerSession
session = SchedulerSession.from_config('configs/mpnn_lstm_4gpu.yaml')
session.prepare_data()
"

# 2. Train
python -c "
from starry_unigraph import SchedulerSession
session = SchedulerSession.from_config('configs/mpnn_lstm_4gpu.yaml')
session.build_runtime()
result = session.run_task()
"

# 3. Or use provided scripts
python train_mpnn_lstm_4gpu.py --mode prepare
python train_mpnn_lstm_4gpu.py --mode train
python train_mpnn_lstm_4gpu.py --mode predict
```

### Multi-GPU Training

```bash
# Use torchrun for distributed execution
torchrun --nproc_per_node=4 train_mpnn_lstm_4gpu.py --mode train
```

### Testing Imports

```bash
# Verify migration worked
python -c "
from starry_unigraph.backends.ctdg import CTDGSession, CTDGPreprocessor
from starry_unigraph.backends.dtdg import FlareDTDGPreprocessor, FlareRuntimeLoader
from starry_unigraph.runtime.chunk import ChunkRuntimeLoader
print('All imports OK')
"
```

## Key Paths and Dependencies

### For CTDG Development

- Preprocessor: `backends/ctdg/preprocess.py:CTDGPreprocessor`
- Session: `backends/ctdg/runtime/session.py:CTDGSession`
- Data: `backends/ctdg/runtime/data.py:TGTemporalDataset`
- Memory: `backends/ctdg/runtime/memory.py:CTDGMemoryBank`
- Sampler: `backends/ctdg/runtime/sampler.py:NativeTemporalSampler` (wraps C++ BTS sampler)

### For DTDG Development

- Preprocessor: `backends/dtdg/preprocess.py:FlareDTDGPreprocessor`
- Runtime Loader: `backends/dtdg/runtime/session_loader.py:FlareRuntimeLoader`
- State Manager: `backends/dtdg/runtime/state.py:RNNStateManager`
- Training Steps: `backends/dtdg/runtime/training.py:run_flare_train_step`, etc.
- Models: `backends/dtdg/runtime/models.py:FlareMPNNLSTM`

### For Chunk Development (Future)

- Preprocessor: `preprocess/chunk.py:ChunkPreprocessor`
- Runtime Loader: `runtime/chunk/session_loader.py:ChunkRuntimeLoader` (currently stubs)
- Training Steps: `runtime/chunk/training.py` (imports from backends/dtdg for compatibility)

### Shared Data Layer

- `data/partition.py`: `PartitionData`, `RouteData`, `TensorData` (used by DTDG)
- `data/chunk_data.py`: `ChunkAtomic` reference (for chunk design)
- `data/raw_temporal.py`: Raw event loading
- Base utilities: `runtime/modules/`, `runtime/route/`, `runtime/store/`

### Configuration

- Base config: `config/mpnn_lstm.yaml` (world_size=1, snaps=4)
- 4-GPU config: `configs/mpnn_lstm_4gpu.yaml` (snaps=200)
- Config loading: `config/schema.py:load_config()`, `validate_config()`

## Critical Implementation Details

### CTDG Training Loop (in CTDGSession)

1. `prepare_data()` → CTDGPreprocessor.run() creates artifacts
2. `build_runtime()` → Factory builds CTDGOnlineRuntime with memory banks, sampler
3. `iter_train()` → yields temporal event batches
4. `train_step(batch)` → forward + backward + memory update

### DTDG Training Loop (in SchedulerSession.run_epoch for dtdg_session=None)

1. `prepare_data()` → FlareDTDGPreprocessor.run() creates partitioned artifacts
2. `build_runtime()` → FlareRuntimeLoader.from_partition_data()
3. `iter_train()` → yields STGraphBlob (multi-frame RNN state blob)
4. `run_train_step()` → delegates to `run_flare_train_step()` from training.py

### Distributed Setup

- `apply_distributed_env()` reads WORLD_SIZE/RANK/LOCAL_RANK from environment (set by torchrun)
- `build_distributed_context()` creates DistributedContext with rank/world_size
- DDP wrapping happens in `init_flare_training()` when distributed
- Session checks `dist.is_initialized()` before using distributed features

### Artifact Structure

After `prepare_data()`:
- CTDG: `artifacts/<dataset>/{meta, ...}` (simple metadata)
- DTDG: `artifacts/<dataset>/{meta, partitions, routes, flare, snapshots}` (complex)
  - `meta/artifacts.json` stores graph_mode, num_parts, feature_dim
  - `flare/part_XXX.pth` contains PartitionData objects
  - Validation: `validate_artifacts(expected_graph_mode="dtdg", expected_num_parts=world_size)`

## Memory and Performance Notes

### Native Sampler (BTS)

- Built C++ code in `vendor/bts_sampler/` → `lib/libstarrygl_sampler.so`
- Build requires: `tgnn_3.10` conda env (PyTorch 2.1.1+cu118)
- Must use correct Python executable to avoid ABI mismatch
- Rebuild: `cd starry_unigraph/vendor/bts_sampler && mkdir build && cd build && cmake .. && make -j$(nproc)`

### Dataset: rec-amazon-ratings

- Location: `/mnt/data/zlj/starrygl-data/raw/rec-amazon-ratings/rec-amazon-ratings.edges`
- Scale: ~5.8M edges, ~2.1M nodes
- Load time: ~56 seconds
- Handles `%` comment lines in edge file

### FlareMPNNLSTM State Handling

- Uses custom `_LSTMCell` (not nn.LSTMCell)
- State is 4-tuple: `(h1, c1, h2, c2)` from two LSTM layers concatenated
- `flare_fetch_state(h)` reads from RNNStateManager, `flare_store_state(h)` writes back
- During eval/predict: do NOT pre-patch graph before `states.add()` — let the manager handle it

## Code Organization Principles

### Isolation by Graph Mode

- **CTDG and DTDG share NO code**. Each has complete implementations:
  - Preprocessors, runtime loaders, state managers, training step functions
  - Even models (CTDGLinkPredictor vs FlareMPNNLSTM) are separate

### Unified Interfaces (Minimal Coupling)

- Session only knows about dispatch: `if graph_mode == "ctdg": use CTDGSession else use FlareRuntimeLoader`
- Models, training steps, state management are backend-specific
- CommEngine, FeatureStore, and base utilities are shared (used by both paths)

### Protocol-Oriented Design (for Future Chunk)

- `Runner` protocol (build_sequence, prepare_data, sync_data, compute, post_compute)
- `StateManager` protocol (prepare, patch, update, step)
- Chunk implementation should follow these protocols to integrate with SchedulerSession

## When to Edit Key Files

- **To add a CTDG model**: Edit `backends/ctdg/runtime/models.py`
- **To add a DTDG model**: Edit `backends/dtdg/runtime/models.py`
- **To change session dispatch**: Edit `session.py:build_runtime()` (be careful to maintain independence)
- **To add config options**: Edit `config/schema.py` and appropriate `*_yaml` files
- **To implement chunk**: Edit `runtime/chunk/session_loader.py` and `preprocess/chunk.py`

## Common Pitfalls

1. **Circular imports**: DTDG has `backends/dtdg/types.py` to avoid cycles. Import from there, not preprocess.
2. **Wrong import path**: After migration, imports must use `backends.ctdg` or `backends.dtdg`, not old `runtime.online` or `runtime.flare`
3. **Artifact mismatch**: `validate_artifacts()` expects `num_parts == world_size`. Prepare and train must use same world_size.
4. **DDP assumptions**: Code checks `dist.is_initialized()` — some features only work in distributed mode
5. **State handling**: RNNStateManager expects graphs to be patched properly. Don't manually modify graph state in training steps.

## Future Work: Chunk Implementation

The `runtime/chunk/session_loader.py:ChunkRuntimeLoader` provides the minimal interface template:
- `iter_train()`, `iter_eval()`, `iter_predict()` (currently stubs)
- `build_snapshot_index()`, `dump_state()`, `describe_window_state()`, `describe_route_cache()`

Chunk training should:
1. Implement the Runner protocol (build_sequence, prepare_data, sync_data, compute, post_compute)
2. Implement the StateManager protocol (prepare, patch, update, step)
3. Use the same training step function signature as DTDG (reuse `run_flare_*_step`)
4. Follow the pattern: chunk materialize → MFG → model forward/backward (same as DTDG path)
