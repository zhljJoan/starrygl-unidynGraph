"""Task adapters for chunk pipeline.

Four adapters covering the 2×2 matrix of (edge/node) × (classify/regress):
  EdgePredictAdapter  — link existence prediction (BCE + AUC/AP)
  EdgeRegressAdapter  — edge attribute regression  (MSE + MAE/RMSE)
  NodeClassifyAdapter — node classification        (CE  + Acc/F1)
  NodeRegressAdapter  — node regression            (MSE + MAE/RMSE)

Each adapter:
  build_batch(part_data, event_pos, split, neg_sampler) -> BatchData
      Fills task-specific fields (pos_src/pos_dst/neg_*/target_nodes/labels).
  compute_loss(model_output, batch) -> Tensor
  compute_metrics(model_output, batch) -> Dict[str, float]

These adapters operate on chunk-specific BatchData
(starry_unigraph.backends.chunk.data.batch.BatchData).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from starry_unigraph.backends.chunk.data.batch import BatchData
from starry_unigraph.backends.chunk.data.partition import PartitionData
from starry_unigraph.backends.chunk.runtime.sampler import NegativeSamplerHook, _RandomNegativeSampler


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ChunkTaskAdapter:
    """Base class for chunk task adapters."""

    task_type: str = "base"

    def build_batch(
        self,
        part:        PartitionData,
        snapshot_idx: int,
        event_pos:   Tensor,
        split:       str,
        neg_sampler: Optional[NegativeSamplerHook] = None,
        num_nodes:   int = 0,
    ) -> BatchData:
        raise NotImplementedError

    def compute_loss(self, model_output: Dict[str, Tensor], batch: BatchData) -> Tensor:
        raise NotImplementedError

    def compute_metrics(self, model_output: Dict[str, Tensor], batch: BatchData) -> Dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_pairs_from_part(
    part: PartitionData,
    snapshot_idx: int,
    event_pos: Tensor,
    global_ids: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Extract (src, dst) pairs for the selected event positions."""
    src_all, dst_all = part.to_edge_index(snapshot_index=snapshot_idx, global_ids=global_ids)
    valid = event_pos[event_pos >= 0]
    # event_pos values < E index into src/dst of this snapshot
    E = src_all.numel()
    valid = valid[valid < E]
    if valid.numel() == 0:
        return src_all, dst_all
    return src_all[valid], dst_all[valid]


def _node_ids_from_part(part: PartitionData, snapshot_idx: int) -> Tensor:
    return part.dst_ids[snapshot_idx].item()


def _mfg_from_part(part: PartitionData, snapshot_idx: int) -> Any:
    return part.to_block(snapshot_idx, keep_ids=True)


# ---------------------------------------------------------------------------
# Edge prediction (link existence)
# ---------------------------------------------------------------------------

class EdgePredictAdapter(ChunkTaskAdapter):
    """Binary edge existence prediction.

    Loss: BCE with logits.
    Metrics: AUC (pairwise), AP (approximate).
    """

    task_type = "edge_predict"

    def __init__(self, neg_ratio: int = 1) -> None:
        self.neg_ratio = neg_ratio

    def build_batch(
        self,
        part:         PartitionData,
        snapshot_idx: int,
        event_pos:    Tensor,
        split:        str,
        neg_sampler:  Optional[NegativeSamplerHook] = None,
        num_nodes:    int = 0,
    ) -> BatchData:
        pos_src, pos_dst = _edge_pairs_from_part(part, snapshot_idx, event_pos)
        node_ids = _node_ids_from_part(part, snapshot_idx)

        sampler = neg_sampler or _RandomNegativeSampler()
        neg_src, neg_dst = sampler.sample(
            pos_src,
            pos_dst,
            num_nodes or int(node_ids.max()) + 1,
            self.neg_ratio,
            split=split,
        )

        return BatchData(
            mfgs     = _mfg_from_part(part, snapshot_idx),
            node_ids = node_ids,
            pos_src  = pos_src,
            pos_dst  = pos_dst,
            neg_src  = neg_src,
            neg_dst  = neg_dst,
            chunk_id = snapshot_idx,
        )

    def compute_loss(self, model_output: Dict[str, Tensor], batch: BatchData) -> Tensor:
        pos = model_output.get("pos_score")
        neg = model_output.get("neg_score")
        if pos is None or neg is None:
            return torch.tensor(0.0)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_metrics(self, model_output: Dict[str, Tensor], batch: BatchData) -> Dict[str, float]:
        pos = model_output.get("pos_score")
        neg = model_output.get("neg_score")
        if pos is None or neg is None:
            return {}
        pos_p = pos.sigmoid().detach().cpu()
        neg_p = neg.sigmoid().detach().cpu()
        auc = float((pos_p[:, None] > neg_p[None, :]).float().mean())
        ap  = float(torch.clamp((pos_p.mean() - neg_p.mean()) / 2 + 0.5, 0, 1))
        return {"auc": auc, "ap": ap}


# ---------------------------------------------------------------------------
# Edge regression (edge attribute prediction)
# ---------------------------------------------------------------------------

class EdgeRegressAdapter(ChunkTaskAdapter):
    """Regression on edge attributes (e.g., interaction weight).

    Expects model_output['edge_pred'] [M, output_dim] and
    batch.labels [M, output_dim].

    Loss: MSE.  Metrics: MAE, RMSE.
    """

    task_type = "edge_regress"

    def build_batch(
        self,
        part:         PartitionData,
        snapshot_idx: int,
        event_pos:    Tensor,
        split:        str,
        neg_sampler:  Optional[NegativeSamplerHook] = None,
        num_nodes:    int = 0,
    ) -> BatchData:
        pos_src, pos_dst = _edge_pairs_from_part(part, snapshot_idx, event_pos)
        node_ids = _node_ids_from_part(part, snapshot_idx)

        # Labels: edge features stored in part.edge_data["y"] if available
        labels: Optional[Tensor] = None
        if "y" in part.edge_data:
            raw = part.edge_data["y"][snapshot_idx].item()
            E = pos_src.numel()
            valid = event_pos[event_pos >= 0]
            valid = valid[valid < raw.size(0)]
            labels = raw[valid] if valid.numel() > 0 else raw[:E]

        return BatchData(
            mfgs     = _mfg_from_part(part, snapshot_idx),
            node_ids = node_ids,
            pos_src  = pos_src,
            pos_dst  = pos_dst,
            labels   = labels,
            chunk_id = snapshot_idx,
        )

    def compute_loss(self, model_output: Dict[str, Tensor], batch: BatchData) -> Tensor:
        pred = model_output.get("edge_pred")
        if pred is None or batch.labels is None:
            return torch.tensor(0.0)
        return F.mse_loss(pred, batch.labels.to(pred.dtype))

    def compute_metrics(self, model_output: Dict[str, Tensor], batch: BatchData) -> Dict[str, float]:
        pred = model_output.get("edge_pred")
        if pred is None or batch.labels is None:
            return {}
        pred = pred.detach().cpu()
        tgt  = batch.labels.cpu().to(pred.dtype)
        mae  = float((pred - tgt).abs().mean())
        rmse = float(((pred - tgt) ** 2).mean().sqrt())
        return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

class NodeClassifyAdapter(ChunkTaskAdapter):
    """Node classification (discrete label per node).

    Expects model_output['logits'] [M, num_classes] and
    batch.labels [M] (int64 class indices).

    Loss: CrossEntropy.  Metrics: accuracy, macro-F1.
    """

    task_type = "node_classify"

    def build_batch(
        self,
        part:         PartitionData,
        snapshot_idx: int,
        event_pos:    Tensor,
        split:        str,
        neg_sampler:  Optional[NegativeSamplerHook] = None,
        num_nodes:    int = 0,
    ) -> BatchData:
        node_ids = _node_ids_from_part(part, snapshot_idx)
        labels: Optional[Tensor] = None
        if "y" in part.node_data:
            labels = part.node_data["y"][snapshot_idx].item().long()

        return BatchData(
            mfgs         = _mfg_from_part(part, snapshot_idx),
            node_ids     = node_ids,
            target_nodes = None,   # all local dst nodes
            labels       = labels,
            chunk_id     = snapshot_idx,
        )

    def compute_loss(self, model_output: Dict[str, Tensor], batch: BatchData) -> Tensor:
        logits = model_output.get("logits")
        if logits is None or batch.labels is None:
            return torch.tensor(0.0)
        return F.cross_entropy(logits, batch.labels.to(logits.device))

    def compute_metrics(self, model_output: Dict[str, Tensor], batch: BatchData) -> Dict[str, float]:
        logits = model_output.get("logits")
        if logits is None or batch.labels is None:
            return {}
        pred = logits.argmax(dim=1).detach().cpu()
        tgt  = batch.labels.cpu()
        acc  = float((pred == tgt).float().mean())
        return {"accuracy": acc}


# ---------------------------------------------------------------------------
# Node regression
# ---------------------------------------------------------------------------

class NodeRegressAdapter(ChunkTaskAdapter):
    """Node regression (continuous value per node).

    Expects model_output['node_pred'] [M, output_dim] and
    batch.labels [M, output_dim].

    Loss: MSE.  Metrics: MAE, RMSE.
    """

    task_type = "node_regress"

    def build_batch(
        self,
        part:         PartitionData,
        snapshot_idx: int,
        event_pos:    Tensor,
        split:        str,
        neg_sampler:  Optional[NegativeSamplerHook] = None,
        num_nodes:    int = 0,
    ) -> BatchData:
        node_ids = _node_ids_from_part(part, snapshot_idx)
        labels: Optional[Tensor] = None
        if "y" in part.node_data:
            labels = part.node_data["y"][snapshot_idx].item().float()

        return BatchData(
            mfgs         = _mfg_from_part(part, snapshot_idx),
            node_ids     = node_ids,
            target_nodes = None,
            labels       = labels,
            chunk_id     = snapshot_idx,
        )

    def compute_loss(self, model_output: Dict[str, Tensor], batch: BatchData) -> Tensor:
        pred = model_output.get("node_pred")
        if pred is None or batch.labels is None:
            return torch.tensor(0.0)
        return F.mse_loss(pred, batch.labels.to(pred.dtype).to(pred.device))

    def compute_metrics(self, model_output: Dict[str, Tensor], batch: BatchData) -> Dict[str, float]:
        pred = model_output.get("node_pred")
        if pred is None or batch.labels is None:
            return {}
        pred = pred.detach().cpu()
        tgt  = batch.labels.cpu().to(pred.dtype)
        mae  = float((pred - tgt).abs().mean())
        rmse = float(((pred - tgt) ** 2).mean().sqrt())
        return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_ADAPTERS: Dict[str, type] = {
    "edge_predict":  EdgePredictAdapter,
    "edge_regress":  EdgeRegressAdapter,
    "node_classify": NodeClassifyAdapter,
    "node_regress":  NodeRegressAdapter,
    # aliases
    "link_prediction": EdgePredictAdapter,
    "node_classification": NodeClassifyAdapter,
    "node_regression": NodeRegressAdapter,
}


def get_task_adapter(task_type: str, **kwargs) -> ChunkTaskAdapter:
    """Instantiate a task adapter by name."""
    cls = TASK_ADAPTERS.get(task_type)
    if cls is None:
        raise ValueError(f"Unknown task_type '{task_type}'. Available: {list(TASK_ADAPTERS)}")
    return cls(**kwargs)
