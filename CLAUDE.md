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
