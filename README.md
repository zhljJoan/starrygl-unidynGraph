# StarryGL 0.1.1

[DOCUMENT](https://shadow-greasily-unretired.ngrok-free.dev/)

StarryGL is a unified distributed training stack for large-scale dynamic graphs.
It supports both:

- **DTDG**: discrete-time dynamic graphs represented as ordered snapshots
- **CTDG**: continuous-time dynamic graphs represented as ordered event streams

The 0.1.1 documentation describes the current production-ready DTDG and CTDG
training paths, plus an experimental chunk-centric execution model that aims to
unify both modes under one runtime.

## Why StarryGL

Training temporal GNNs at scale is difficult because the system must handle:

- **memory pressure** from temporal history, sampled neighborhoods, and runtime state
- **communication overhead** from distributed feature, memory, and routing exchange
- **workload imbalance** caused by skewed graph topology and bursty temporal activity

StarryGL is built to address these problems with:

- scalable distributed execution across GPUs and nodes
- adaptive communication-computation overlap
- mode-specific runtime optimizations for DTDG and CTDG workloads
- an experimental chunk-centric abstraction for future unified execution

## Two Graph Modes

### DTDG

Use **DTDG** when your data naturally forms discrete snapshots such as hourly,
daily, or weekly graph states.

This mode is appropriate when:

- topology inside each time window matters
- the model reasons over snapshot-level temporal evolution
- exact ordering of individual events is less important than window-level state

Built-in DTDG models include:

- `mpnn_lstm`
- `tgcn`
- `evolvegcn`

### CTDG

Use **CTDG** when your data is a continuous timestamped event stream.

This mode is appropriate when:

- exact event ordering matters
- the model depends on memory, mailbox, or time-aware state
- temporal dependencies are defined by interaction streams instead of snapshots

Built-in CTDG models include:

- `tgn`
- `jodie`

### CHUNK (Experimental)

The experimental **CHUNK** path introduces a mode-agnostic execution unit that
encodes:

- a bounded time range
- a bounded topology region
- distributed routing metadata
- temporal and spatial dependency information

This abstraction is designed to let future StarryGL runtimes reuse scheduling,
prefetching, and communication logic across both DTDG and CTDG workloads.

## End-to-End Workflow

StarryGL uses the same high-level lifecycle in both stable graph modes:

1. **Prepare**: preprocess raw graph data into partition-aware artifacts
2. **Train and Validate**: launch the distributed runtime with `torchrun`
3. **Predict**: load the saved checkpoint and run inference on the test split

At the Python level, the unified lifecycle is exposed through
`starry_unigraph.session.SchedulerSession`.

## Quick Start

Before running the examples, install the package and build the native
extensions in a Python 3.10+ environment. StarryGL expects **PyTorch** and
**DGL** to be installed first in the same environment, with CUDA-compatible
versions if you plan to train on GPUs.

### Installation

```bash
# 1. Activate your Python environment first.
conda activate starrygl_graph

# 2. Make sure PyTorch and DGL are already installed in this environment.
python - <<'PY'
import torch
import dgl
print("torch:", torch.__version__)
print("dgl:", dgl.__version__)
PY

# 3. Build native extensions with the active interpreter.
cmake -S . -B build -DPython3_EXECUTABLE=$(which python)
cmake --build build -j

# 4. Install the package in editable mode.
python -m pip install -e .

# 5. Verify the CLI entry point.
python -m starry_unigraph -h
```

Before launching any config, update machine-specific dataset or checkpoint
paths in `configs/*.yaml` if they still point to local absolute directories such
as `/mnt/...`.

### DTDG example

```bash
# 1. Prepare artifacts once.
python -m starry_unigraph \
    --config configs/mpnn_lstm_4gpu.yaml \
    --phase prepare

# 2. Train on 4 GPUs.
torchrun --nproc_per_node=4 -m starry_unigraph \
    --config configs/mpnn_lstm_4gpu.yaml \
    --phase train

# 3. Predict on the test split.
torchrun --nproc_per_node=4 -m starry_unigraph \
    --config configs/mpnn_lstm_4gpu.yaml \
    --phase predict
```

### CTDG example

```bash
# 1. Prepare artifacts once.
python -m starry_unigraph \
    --config configs/tgn_wikitalk.yaml \
    --phase prepare

# 2. Train on 4 GPUs.
torchrun --nproc_per_node=4 -m starry_unigraph \
    --config configs/tgn_wikitalk.yaml \
    --phase train

# 3. Predict on the test split.
torchrun --nproc_per_node=4 -m starry_unigraph \
    --config configs/tgn_wikitalk.yaml \
    --phase predict
```

## Configuration Notes

### DTDG example config

`configs/mpnn_lstm_4gpu.yaml` highlights the main DTDG controls:

- `data.graph_mode: dtdg`
- `model.name`: selects the concrete temporal model
- `model.window.size`: controls historical window length
- `train.snaps`: controls snapshot coverage per epoch
- `dtdg.pipeline: flare_native`: enables the optimized Flare pipeline
- `dist.world_size`: must match the distributed launch world size

### CTDG example config

`configs/tgn_wikitalk.yaml` highlights the main CTDG controls:

- `data.graph_mode: ctdg`
- `model.family`: selects the CTDG model family
- `train.batch_size`: controls events per optimization step
- `ctdg.pipeline: online`: enables the event-driven runtime
- `ctdg.mailbox_slots`: controls retained pending messages per node
- `ctdg.async_sync`: enables overlap of memory sync and computation
- `sampling.neighbor_limit`: controls temporal neighbor depth
- `dist.world_size`: must match the distributed launch world size

## Repository Structure

```text
StarryUniGraph/
├── configs/                    # Ready-to-use YAML configuration templates
├── docs/source/rel_0_1_1/      # 0.1.1 documentation source
├── examples/                   # Example training and launch scripts
├── tests/                      # Test suite
└── starry_unigraph/            # Core framework source code
    ├── __main__.py             # CLI entry point
    ├── session.py              # Unified lifecycle API
    ├── distributed.py          # Distributed communication primitives
    ├── preprocess/             # Shared preprocessing logic
    ├── models/                 # Model wrappers and task heads
    ├── tasks/                  # Task adapters
    ├── registry/               # Model/task registration
    ├── runtime/                # Shared runtime modules
    └── backends/
        ├── dtdg/               # Snapshot-based runtime
        ├── ctdg/               # Event-driven runtime
        ├── chunk/              # Experimental chunk backend pieces
        └── flare/              # Backward-compatibility exports
```

## Key Modules

- `starry_unigraph/session.py`: orchestrates preprocessing, runtime creation,
  training, evaluation, and prediction
- `starry_unigraph/backends/dtdg/preprocess.py`: builds DTDG artifacts
- `starry_unigraph/backends/dtdg/runtime/session_loader.py`: loads partitioned DTDG runtime data
- `starry_unigraph/backends/ctdg/preprocess.py`: builds CTDG artifacts
- `starry_unigraph/backends/ctdg/runtime/factory.py`: constructs the CTDG runtime
- `starry_unigraph/backends/ctdg/runtime/runtime.py`: executes CTDG train, eval, and predict steps
- `starry_unigraph/preprocess/chunk.py`: experimental chunk preprocessing path

## Mathematical Intuition

### DTDG temporal decay

The Flare backend conceptually reduces the representation cost of older
temporal blocks:

```math
\mathcal{N}_{t-1} = \lfloor \beta \cdot \mathcal{N}_t \rfloor
```

This decay is implemented as a batching and storage policy rather than as a
user-managed formula.

### CTDG memory update

For an interaction between nodes `u` and `v` at time `t`, StarryGL follows the
standard memory-update pattern:

```math
m_e^{uv} = \text{MSG}(s(u,t^-) \,\|\, s(v,t^-) \,\|\, e(u,v,t) \,\|\, \phi(t-t^-))
```

```math
s(u,t) = \text{UPDATE}(s(u,t^-), m_e^{uv})
```

The CTDG runtime handles mailbox buffering, chronological state updates,
temporal neighbor lookup, and distributed memory synchronization.

## Documentation Source

This README is derived from the 0.1.1 documentation in:

- `docs/source/rel_0_1_1/index.rst`
- `docs/source/rel_0_1_1/intro.rst`
- `docs/source/rel_0_1_1/reference.rst`
- `docs/source/rel_0_1_1/training/dtdg.rst`
- `docs/source/rel_0_1_1/training/ctdg.rst`
- `docs/source/rel_0_1_1/training/chunk.rst`

## Publications

If you use StarryGL or its methodologies in research, the 0.1.1 docs reference
the following core papers:

- Wenjie Huang, Rui Wang, Jing Cao, Tongya Zheng, Xinyu Wang, Mingli Song, Sai
  Wu, and Chun Chen. *FlareDTDG: Harnessing Temporal Recency for Scalable
  Discrete-Time Dynamic Graph Training*. PVLDB, 2025.
- Longjiao Zhang, Rui Wang, Tongya Zheng, Ziqi Huang, Wenjie Huang, Xinyu Wang,
  Can Wang, Mingli Song, Sai Wu, and Shuibing He. *Effective and Efficient
  Distributed Temporal Graph Learning through Hotspot Memory Sharing*. PVLDB,
  18(9): 3093-3105, 2025. doi:10.14778/3746405.3746430.
