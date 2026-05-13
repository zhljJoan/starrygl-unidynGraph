"""Chunk-line FeatureStore: CPU-memory feature management with k-hop caching.

Manages node/edge features stored in CPU memory, supporting:
- Local partition features (point and edge)
- k-hop neighborhood features
- Hot data caching and CPU-GPU transfer

The actual feature access (hit/miss, eviction) will be handled by C++ in
a later phase, but this Python layer provides the interface and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class FeatureStoreConfig:
    """Configuration for chunk feature storage and caching.

    Attributes:
        num_nodes: Total number of nodes in the graph
        num_local_nodes: Number of nodes owned by this partition
        feature_dim: Feature dimensionality
        enable_k_hop: Whether to cache k-hop neighbors
        k_hop_size: Number of hops to cache (if enabled)
        hot_data_ratio: Fraction of features to keep in "hot" cache
        device: CPU device for storage ("cpu")
    """

    num_nodes: int
    num_local_nodes: int
    feature_dim: int
    enable_k_hop: bool = True
    k_hop_size: int = 2
    hot_data_ratio: float = 0.2
    device: str = "cpu"


@dataclass
class FeatureStore:
    """CPU-memory feature storage for chunk pipeline.

    Stores node and edge features for the local partition and optionally
    caches k-hop neighborhood features. Provides interfaces for CPU-GPU
    transfer, which will be optimized in C++ later.

    Attributes:
        config: FeatureStoreConfig
        node_features: [num_local_nodes, feature_dim] Local node features
        edge_features: [num_local_edges, feature_dim] Local edge features (optional)
        khop_features: [num_khop_nodes, feature_dim] k-hop neighbor features (optional)
        khop_node_ids: [num_khop_nodes] Global IDs of k-hop nodes
        feature_index: Dict mapping node_id → local storage index
    """

    config: FeatureStoreConfig
    node_features: Optional[Tensor] = None
    edge_features: Optional[Tensor] = None
    khop_features: Optional[Tensor] = None
    khop_node_ids: Optional[Tensor] = None
    feature_index: Dict[int, int] = field(default_factory=dict)

    def allocate_node_features(self, dtype: torch.dtype = torch.float32) -> None:
        """Allocate storage for local node features.

        Args:
            dtype: Data type for feature storage
        """
        self.node_features = torch.zeros(
            (self.config.num_local_nodes, self.config.feature_dim),
            dtype=dtype,
            device=self.config.device,
        )

    def allocate_khop_features(self, num_khop_nodes: int, dtype: torch.dtype = torch.float32) -> None:
        """Allocate storage for k-hop neighbor features.

        Args:
            num_khop_nodes: Number of k-hop nodes to cache
            dtype: Data type for feature storage
        """
        self.khop_features = torch.zeros(
            (num_khop_nodes, self.config.feature_dim),
            dtype=dtype,
            device=self.config.device,
        )
        self.khop_node_ids = torch.zeros(num_khop_nodes, dtype=torch.long, device=self.config.device)

    def copy_node_features_to_gpu(
        self,
        node_ids: Tensor,
        gpu_device: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tensor:
        """Copy node features from CPU to GPU.

        This is a Python-level interface. Later optimization can move this to C++.

        Args:
            node_ids: [M] Local indices or global IDs of nodes to transfer
            gpu_device: Target GPU device (e.g., "cuda:0")
            stream: Optional CUDA stream for async transfer

        Returns:
            Feature tensor on GPU
        """
        if self.node_features is None:
            raise ValueError("Node features not allocated")

        if stream is not None:
            with torch.cuda.stream(stream):
                features = self.node_features[node_ids].to(gpu_device, non_blocking=True)
        else:
            features = self.node_features[node_ids].to(gpu_device)

        return features

    def prefetch_khop_to_gpu(self, gpu_device: str) -> None:
        """Prefetch k-hop features to GPU for hot data caching.

        Args:
            gpu_device: Target GPU device
        """
        if self.khop_features is not None:
            self.khop_features = self.khop_features.to(gpu_device)

    def save_to_disk(self, output_dir: Path) -> None:
        """Save feature store to disk.

        Args:
            output_dir: Directory to save features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.node_features is not None:
            torch.save(self.node_features, output_dir / "node_features.pth")

        if self.edge_features is not None:
            torch.save(self.edge_features, output_dir / "edge_features.pth")

        if self.khop_features is not None:
            torch.save(self.khop_features, output_dir / "khop_features.pth")
            torch.save(self.khop_node_ids, output_dir / "khop_node_ids.pth")

        if self.feature_index:
            torch.save(self.feature_index, output_dir / "feature_index.pth")

    @classmethod
    def load_from_disk(cls, input_dir: Path, config: FeatureStoreConfig) -> FeatureStore:
        """Load feature store from disk.

        Args:
            input_dir: Directory containing saved features
            config: FeatureStoreConfig for this store

        Returns:
            Loaded FeatureStore
        """
        input_dir = Path(input_dir)
        store = cls(config=config)

        node_features_path = input_dir / "node_features.pth"
        if node_features_path.exists():
            store.node_features = torch.load(node_features_path, map_location=config.device)

        edge_features_path = input_dir / "edge_features.pth"
        if edge_features_path.exists():
            store.edge_features = torch.load(edge_features_path, map_location=config.device)

        khop_features_path = input_dir / "khop_features.pth"
        if khop_features_path.exists():
            store.khop_features = torch.load(khop_features_path, map_location=config.device)

        khop_node_ids_path = input_dir / "khop_node_ids.pth"
        if khop_node_ids_path.exists():
            store.khop_node_ids = torch.load(khop_node_ids_path, map_location=config.device)

        index_path = input_dir / "feature_index.pth"
        if index_path.exists():
            store.feature_index = torch.load(index_path)

        return store
