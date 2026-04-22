"""SPEED temporal graph partitioning wrapper for CTDG.

This module wraps the C++ SPEED algorithm to compute node-to-partition
assignments for CTDG online streaming. Unlike DTDG which partitions edges,
CTDG partitions nodes to direct memory updates and feature routing.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


SPEED_BINARY = Path.home() / "SPEED" / "partition" / "starrygl_partition"

SPEED_DEFAULTS = {
    "beta": 1.0,
    "topk_type": "decay",
    "topk_ratio": 0.01,
    "reorder_type": "normal",
}


def write_edge_file(edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor], path: Path) -> None:
    """Write edge list in SPEED input format (src dest timestamp).

    Args:
        edges: Tuple of (src, dst, ts) tensors, each shape [num_edges].
        path: Output file path.
    """
    src, dst, ts = edges
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    ts_np = ts.cpu().numpy()

    with open(path, "w") as f:
        for s, d, t in zip(src_np, dst_np, ts_np):
            f.write(f"{int(s)} {int(d)} {float(t)}\n")


def parse_speed_output(output_dir: Path, num_nodes: int) -> tuple[torch.Tensor, dict[int, int]]:
    """Parse SPEED output files into node partitions and edge partitions.

    SPEED writes files matching pattern `output*.txt` where each line is
    "node_id partition_id". Returns tensor [num_nodes] mapping each node to partition.

    Args:
        output_dir: Directory containing SPEED output files.
        num_nodes: Total number of nodes.

    Returns:
        Tuple of (node_parts, edge_parts) where:
        - node_parts: Tensor [num_nodes] mapping node_id → partition_id
        - edge_parts: Dict mapping edge_id → partition_id (for validation)
    """
    # Initialize node_parts with -1 (unassigned)
    node_parts = torch.full((num_nodes,), -1, dtype=torch.long)

    # Find and parse output files
    output_files = sorted(output_dir.glob("output*.txt"))
    if not output_files:
        raise FileNotFoundError(f"No output*.txt files found in {output_dir}")

    for output_file in output_files:
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    node_id = int(parts[0])
                    part_id = int(parts[1])
                    if 0 <= node_id < num_nodes:
                        node_parts[node_id] = part_id
                except (ValueError, IndexError):
                    continue

    # Verify all nodes assigned
    unassigned = (node_parts == -1).sum().item()
    if unassigned > 0:
        print(f"Warning: {unassigned} nodes not assigned by SPEED, using round-robin fallback")
        # Fallback: assign unassigned nodes round-robin to distribute load
        unassigned_ids = torch.where(node_parts == -1)[0]
        for i, node_id in enumerate(unassigned_ids):
            node_parts[node_id] = i % (node_parts.max().item() + 1)

    # Parse edge output if available (for validation/analysis)
    edge_parts = {}
    edge_files = sorted(output_dir.glob("edge_output*.txt"))
    for edge_file in edge_files:
        with open(edge_file, "r") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        part_id = int(parts[0])
                        edge_parts[line_no] = part_id
                    except ValueError:
                        continue

    return node_parts, edge_parts


def derive_edge_parts(
    src: torch.Tensor,
    node_parts: torch.Tensor,
) -> torch.Tensor:
    """Assign each edge to the partition of its source node.

    This matches the vertex-cut ownership convention used by MemShare's
    ``partition_data_for_tgnn`` path, where local training edges on a rank are
    the edges whose source node belongs to that rank.
    """
    return node_parts[src.long()].long()


def build_round_robin_node_parts(num_nodes: int, num_parts: int) -> torch.Tensor:
    """Fallback node partitioning when SPEED artifacts are unavailable."""
    return torch.arange(num_nodes, dtype=torch.long) % max(num_parts, 1)


def run_speed_partition(
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_nodes: int,
    num_parts: int,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute SPEED partitioning algorithm and return node assignments.

    Args:
        edges: Tuple of (src, dst, ts) tensors.
        num_nodes: Total number of nodes.
        num_parts: Number of partitions.
        config: Optional config dict with SPEED parameters (beta, topk_type, topk_ratio, reorder_type).

    Returns:
        Tuple of:
        - node_parts: Tensor [num_nodes] mapping each node to partition_id [0, num_parts).
        - edge_parts: Tensor [num_edges] mapping each edge to partition_id [0, num_parts).

    Raises:
        FileNotFoundError: If SPEED binary not found.
        RuntimeError: If SPEED execution fails.
    """
    if not SPEED_BINARY.exists():
        raise FileNotFoundError(
            f"SPEED binary not found at {SPEED_BINARY}. "
            f"Please build SPEED at ~/SPEED/partition/ or provide binary path."
        )

    # Extract SPEED parameters from config
    cfg = config or {}
    beta = float(cfg.get("ctdg", {}).get("speed_beta", SPEED_DEFAULTS["beta"]))
    topk_type = str(cfg.get("ctdg", {}).get("speed_topk_type", SPEED_DEFAULTS["topk_type"]))
    topk_ratio = float(cfg.get("ctdg", {}).get("speed_topk_ratio", SPEED_DEFAULTS["topk_ratio"]))
    reorder_type = str(cfg.get("ctdg", {}).get("speed_reorder_type", SPEED_DEFAULTS["reorder_type"]))

    # Create temporary directory for SPEED I/O
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write edges to temp file
        edge_file = tmpdir_path / "edges.txt"
        write_edge_file(edges, edge_file)

        # Run SPEED binary
        cmd = [
            str(SPEED_BINARY),
            str(edge_file),
            str(num_parts),
            str(beta),
            topk_type,
            str(topk_ratio),
            reorder_type,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(tmpdir_path),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SPEED execution failed with code {result.returncode}:\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SPEED execution timeout after 3600s")
        except Exception as e:
            raise RuntimeError(f"SPEED execution error: {e}")

        # Parse output
        node_parts, parsed_edge_parts = parse_speed_output(tmpdir_path, num_nodes)

    # Validate partition IDs are in range
    min_part = node_parts.min().item()
    max_part = node_parts.max().item()
    if min_part < 0 or max_part >= num_parts:
        raise ValueError(
            f"SPEED produced invalid partition IDs: min={min_part}, max={max_part}, expected [0, {num_parts})"
        )

    src, _dst, _ts = edges
    edge_parts = derive_edge_parts(src=src, node_parts=node_parts)

    # Keep parsed edge outputs available only for sanity checking.
    if parsed_edge_parts:
        parsed_values = torch.tensor(list(parsed_edge_parts.values()), dtype=torch.long)
        if parsed_values.numel() == edge_parts.numel() and not torch.equal(parsed_values, edge_parts.cpu()):
            print("Warning: SPEED edge_output ownership differs from source-node ownership; using source-node ownership")

    return node_parts, edge_parts


def speed_partition(
    dataset: Any,  # TGTemporalDataset
    num_parts: int,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-level API: partition dataset using SPEED algorithm.

    Args:
        dataset: TGTemporalDataset instance with src, dst, ts attributes.
        num_parts: Number of partitions.
        config: Optional config dict with SPEED parameters.

    Returns:
        Tuple of:
        - node_parts: Tensor [num_nodes] mapping node_id → partition_id.
        - edge_parts: Tensor [num_edges] mapping edge_id → partition_id.
    """
    edges = (dataset.src, dataset.dst, dataset.ts)
    return run_speed_partition(edges, dataset.num_nodes, num_parts, config=config)
