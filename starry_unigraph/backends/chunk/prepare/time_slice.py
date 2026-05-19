"""Time slice generation for CTDG and DTDG modes.

Provides unified time_ptr generation:
- CTDG: Adaptive slicing based on staleness and graph features (C++ native)
- DTDG: Fixed slicing from snapshot boundaries or time windows

time_ptr format: torch.int64[T + 1]
  time_ptr[t] = start index of slice t
  time_ptr[t+1] = end index of slice t (exclusive)
  slice t contains events/edges in range [time_ptr[t], time_ptr[t+1])
"""

from __future__ import annotations

from typing import Optional
import warnings

import torch
from torch import Tensor

# Try to import C++ adaptive split module
try:
    from starry_unigraph.lib import adaptive_split_cpp
    _HAS_ADAPTIVE_SPLIT_CPP = True
except ImportError:
    _HAS_ADAPTIVE_SPLIT_CPP = False
    warnings.warn(
        "adaptive_split_cpp not available. "
        "CTDG adaptive slicing will fall back to uniform slicing. "
        "To enable C++ adaptive split, compile the bts_sampler extension.",
        ImportWarning
    )


def build_time_ptr_dtdg(
    snapshot_boundaries: Tensor,
    total_edges: int,
) -> Tensor:
    """Build time_ptr from DTDG snapshot boundaries.

    Args:
        snapshot_boundaries: [num_snapshots + 1] Edge index boundaries for each snapshot
        total_edges: Total number of edges across all snapshots

    Returns:
        time_ptr: [num_snapshots + 1] Pointer array for time slices
    """
    if snapshot_boundaries.numel() == 0:
        return torch.tensor([0, total_edges], dtype=torch.long)

    time_ptr = snapshot_boundaries.long().contiguous()
    if time_ptr[-1] != total_edges:
        raise ValueError(f"Last snapshot boundary {time_ptr[-1]} != total_edges {total_edges}")

    return time_ptr


def build_time_ptr_ctdg_adaptive(
    src: Tensor,
    dst: Tensor,
    timestamps: Tensor,
    target_batch_size: int = 200,
    graph_feature: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    aggl: float = 0.0,
    use_cpp: bool = True,
) -> Tensor:
    """Build time_ptr for CTDG using adaptive slicing based on staleness.

    Uses C++ native adaptive_split algorithm that considers:
    - Temporal staleness (memory freshness)
    - Aggregation loss (cache locality)
    - Graph feature density

    Args:
        src: [num_events] Source node IDs
        dst: [num_events] Destination node IDs
        timestamps: [num_events] Event timestamps (sorted ascending)
        target_batch_size: Target number of events per slice
        graph_feature: Graph density feature (default 1.0)
        alpha: Staleness weight for exponential decay
        beta: Staleness weight for dictionary size
        aggl: Aggregation loss weight
        use_cpp: Use C++ native implementation (default True)

    Returns:
        time_ptr: [num_slices + 1] Pointer array for time slices
    """
    num_events = int(timestamps.numel())
    if num_events == 0:
        return torch.tensor([0], dtype=torch.long)

    if use_cpp and _HAS_ADAPTIVE_SPLIT_CPP:
        try:
            # Call C++ adaptive split
            result = adaptive_split_cpp.adaptive_split(
                src.cpu().long().contiguous(),
                dst.cpu().long().contiguous(),
                timestamps.cpu().double().contiguous(),
                int(target_batch_size),
                float(graph_feature),
                float(alpha),
                float(beta),
                float(aggl),
                False,
            )

            # Convert group_index to time_ptr
            group_index = result.group_index  # [num_events] batch id for each event
            num_slices = int(group_index.max().item()) + 1

            # Build time_ptr from group_index
            time_ptr = torch.zeros(num_slices + 1, dtype=torch.long)
            for i in range(num_events):
                batch_id = int(group_index[i].item())
                time_ptr[batch_id + 1] += 1

            # Cumsum to get CSR pointer
            time_ptr = time_ptr.cumsum(0)

            return time_ptr
        except Exception as exc:
            warnings.warn(
                "adaptive_split_cpp.adaptive_split failed; "
                f"falling back to uniform slicing. error={type(exc).__name__}: {exc}",
                RuntimeWarning,
            )

    # Fallback: simple uniform slicing by event count.
    step = max(1, int(target_batch_size))
    time_ptr = torch.arange(0, num_events, step, dtype=torch.long)
    if time_ptr.numel() == 0 or int(time_ptr[-1]) != num_events:
        time_ptr = torch.cat([time_ptr, torch.tensor([num_events], dtype=torch.long)])
    return time_ptr


def build_time_ptr_ctdg_fixed_window(
    timestamps: Tensor,
    window_size: float,
) -> Tensor:
    """Build time_ptr for CTDG using fixed time windows.

    Args:
        timestamps: [num_events] Event timestamps (sorted ascending)
        window_size: Time window size (in timestamp units)

    Returns:
        time_ptr: [num_slices + 1] Pointer array for time slices
    """
    num_events = int(timestamps.numel())
    if num_events == 0:
        return torch.tensor([0], dtype=torch.long)

    min_ts = float(timestamps[0].item())
    max_ts = float(timestamps[-1].item())

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    # Compute number of windows
    num_windows = int((max_ts - min_ts) / window_size) + 1

    # Find event indices at window boundaries
    boundaries = [0]
    for i in range(1, num_windows):
        window_end = min_ts + i * window_size
        # Binary search for first event >= window_end
        idx = torch.searchsorted(timestamps, window_end, right=False)
        if idx > boundaries[-1]:
            boundaries.append(int(idx.item()))

    boundaries.append(num_events)

    return torch.tensor(boundaries, dtype=torch.long)


def validate_time_ptr(time_ptr: Tensor, total_size: int) -> None:
    """Validate time_ptr format and consistency.

    Args:
        time_ptr: [num_slices + 1] Pointer array
        total_size: Expected total number of elements (edges/events)

    Raises:
        ValueError: If time_ptr is invalid
    """
    if time_ptr.numel() < 2:
        raise ValueError(f"time_ptr must have at least 2 elements, got {time_ptr.numel()}")

    if time_ptr[0] != 0:
        raise ValueError(f"time_ptr[0] must be 0, got {time_ptr[0]}")

    if time_ptr[-1] != total_size:
        raise ValueError(f"time_ptr[-1] must equal total_size {total_size}, got {time_ptr[-1]}")

    # Check monotonicity
    if not torch.all(time_ptr[1:] >= time_ptr[:-1]):
        raise ValueError("time_ptr must be monotonically non-decreasing")
