"""End-to-end CTDG online runtime for TGN-style temporal link prediction.

:class:`CTDGOnlineRuntime` orchestrates the full train / eval / predict loop:
negative sampling, BTS neighbor sampling, memory read/update, mailbox write,
temporal attention convolution, loss computation, and async distributed sync.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from starry_unigraph.types import DistributedContext

from .data import CTDGDataBatch, TGTemporalDataset
from .memory import CTDGMemoryBank
from .models import CTDGLinkPredictor, CTDGMemoryUpdater, build_dgl_block
from .route import CTDGFeatureRoute
from .sampler import CTDGSampleOutput, NativeTemporalSampler

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


@dataclass
class CTDGPreparedBatch:
    batch: CTDGDataBatch
    neg_dst: torch.Tensor
    sample: CTDGSampleOutput


class CTDGOnlineRuntime:
    """End-to-end CTDG online runtime for temporal link prediction.

    Orchestrates the full pipeline per batch:

    1. **Negative sampling** — sample negatives from the dst node pool.
    2. **BTS temporal neighbor sampling** — via :class:`NativeTemporalSampler`.
    3. **Memory read & update** — read old memory, run :class:`CTDGMemoryUpdater`,
       write updated memory back to :class:`CTDGMemoryBank`.
    4. **Mailbox write** — store ``[src_mem, dst_mem, edge_feat]`` slots.
    5. **Temporal attention** — :class:`CTDGLinkPredictor` conv over sampled
       DGL block with memory as node features.
    6. **Loss & metrics** — BPR loss + AP / AUC / MRR.
    7. **Async distributed sync** — submit memory/mail syncs for remote nodes.

    Args:
        dataset: :class:`TGTemporalDataset` for batching.
        sampler: :class:`NativeTemporalSampler` for neighbor sampling.
        memory: :class:`CTDGMemoryBank` for per-node state.
        model: :class:`CTDGLinkPredictor` (possibly DDP-wrapped).
        optimizer: PyTorch optimizer.
        route: :class:`CTDGFeatureRoute` for distributed exchange.
        device: Target device string.
        dist_ctx: :class:`DistributedContext`.
        memory_updater: Optional :class:`CTDGMemoryUpdater`.

    Example::

        runtime = CTDGOnlineRuntime(...)
        for batch in runtime.iter_batches("train", batch_size=200):
            result = runtime.train_step(batch)
            print(result["loss"], result["meta"]["metrics"]["ap"])
    """
    def __init__(
        self,
        dataset: TGTemporalDataset,
        sampler: NativeTemporalSampler,
        memory: CTDGMemoryBank,
        model: CTDGLinkPredictor,
        optimizer: torch.optim.Optimizer,
        route: CTDGFeatureRoute,
        device: str,
        dist_ctx: DistributedContext,
        memory_updater: CTDGMemoryUpdater | None = None,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.memory = memory
        self.model = model
        self.optimizer = optimizer
        self.route = route
        self.device = device
        self.dist_ctx = dist_ctx
        self.memory_updater = memory_updater
        # Cache of unique dst nodes across full graph for negative sampling
        # (matches MemShare: sample negatives only from nodes that appear as dst)
        self._dst_node_pool: torch.Tensor | None = None

    def iter_batches(self, split: str, batch_size: int) -> Iterable[CTDGDataBatch]:
        yield from self.dataset.iter_batches(
            split=split,
            batch_size=batch_size,
            rank=self.dist_ctx.rank,
            world_size=self.dist_ctx.world_size,
        )

    def _negative_sample(self, batch: CTDGDataBatch) -> torch.Tensor:
        # Sample negatives only from nodes that appear as dst in the full graph
        # (matches MemShare LocalNegativeSampling with dst_node_list=full_dst.unique())
        if self._dst_node_pool is None:
            self._dst_node_pool = self.dataset.dst.unique()
        pool = self._dst_node_pool
        idx = torch.randint(low=0, high=pool.numel(), size=batch.dst.shape, dtype=torch.long)
        negatives = pool[idx]
        collision = negatives.eq(batch.dst.cpu())
        if collision.any():
            idx[collision] = (idx[collision] + 1) % pool.numel()
            negatives = pool[idx]
        return negatives

    def _prepare(self, batch: CTDGDataBatch) -> CTDGPreparedBatch:
        neg_dst = self._negative_sample(batch)
        # Sample only src+dst (not neg) — neg nodes reuse dst's conv output (MemShare pattern)
        roots = torch.unique(torch.cat([batch.src, batch.dst], dim=0)).long()
        ts = torch.full((roots.numel(),), int(batch.ts.max().item()), dtype=torch.long)
        sample = self.sampler.sample(batch.split if batch.split != "predict" else "test", roots, ts)
        return CTDGPreparedBatch(batch=batch, neg_dst=neg_dst, sample=sample)

    def _conv_output(
        self,
        nodes: torch.Tensor,
        sample: CTDGSampleOutput,
    ) -> torch.Tensor:
        dev = self.device
        D = self.memory.hidden_dim
        model_core = self.model.module if hasattr(self.model, "module") else self.model

        if model_core.conv is None or not sample.blocks:
            return torch.zeros(nodes.numel(), D, device=dev)

        roots = sample.root_nodes.to(dev)   # [R]

        # cache edge features on device
        if not hasattr(self, '_edge_feat_gpu'):
            self._edge_feat_gpu = self.dataset.edge_feat.to(dev)

        dgl_block = build_dgl_block(
            sample.blocks[0], roots=roots,
            edge_feat_all=self._edge_feat_gpu, device=dev,
        )
        if dgl_block is None:
            return torch.zeros(nodes.numel(), D, device=dev)

        # index only subgraph nodes from memory via gather() (handles global→local mapping)
        dgl_block.srcdata['h'] = self.memory.gather(dgl_block.srcdata['__ID'].cpu()).to(dev).detach()

        conv_all = model_core.conv(dgl_block)  # [n_roots, D]

        nodes_dev = nodes.to(dev)
        idx = torch.searchsorted(roots, nodes_dev)
        # nodes not in roots (e.g. neg_dst not sampled) → zero vector
        idx = idx.clamp(0, conv_all.size(0) - 1)
        valid = (roots[idx] == nodes_dev)
        out = conv_all[idx]
        out[~valid] = 0.0
        return out

    def _scores_direct(
        self,
        prepared: CTDGPreparedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        src = prepared.batch.src.to(dev)
        dst = prepared.batch.dst.to(dev)
        neg = prepared.neg_dst.to(dev)
        src_conv = self._conv_output(src, prepared.sample)
        dst_conv = self._conv_output(dst, prepared.sample)
        neg_conv = self._conv_output(neg, prepared.sample)
        output = self.model(src_conv=src_conv, dst_conv=dst_conv, neg_conv=neg_conv)
        return output.pos_logits, output.neg_logits

    def _scores(self, prepared: CTDGPreparedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self._scores_direct(prepared)

    def _metric_dict(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> dict[str, float]:
        pos_prob = torch.sigmoid(pos_logits).detach().cpu()
        neg_prob = torch.sigmoid(neg_logits).detach().cpu()
        scores = torch.cat([pos_prob, neg_prob], dim=0).numpy()
        labels = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0).numpy()
        metrics: dict[str, float] = {}
        if average_precision_score is not None:
            metrics["ap"] = float(average_precision_score(labels, scores))
        if roc_auc_score is not None:
            metrics["auc"] = float(roc_auc_score(labels, scores))
        rank = (pos_prob.unsqueeze(1) > neg_prob.unsqueeze(0)).float().mean().item()
        metrics["mrr"] = float(rank)
        return metrics

    def _sync_updater_params(self) -> None:
        if self.memory_updater is None:
            return
        import torch.distributed as dist
        if not dist.is_initialized():
            return
        for param in self.memory_updater.parameters():
            dist.broadcast(param.data, src=0)

    def train_step(self, batch: CTDGDataBatch) -> dict[str, Any]:
        """Execute one training step.

        Steps: drain pending syncs -> negative sample -> BTS sample ->
        memory update -> mailbox write -> conv forward -> loss + backward ->
        optimizer step -> submit async syncs.

        Args:
            batch: A :class:`CTDGDataBatch` mini-batch.

        Returns:
            Dict with ``"loss"`` (float), ``"predictions"`` (list),
            ``"targets"`` (list), ``"meta"`` (dict with metrics, timings).
        """
        self.model.train()
        dev = self.device
        t_step0 = time.perf_counter()

        # Step 1: drain pending async syncs from previous batch
        t_wait0 = time.perf_counter()
        self.memory.wait_pending_syncs()
        sync_wait_ms = (time.perf_counter() - t_wait0) * 1000.0

        # Step 2: negative sampling + BTS sampling
        prepared = self._prepare(batch)
        src = batch.src.to(dev)
        dst = batch.dst.to(dev)
        neg = prepared.neg_dst.to(dev)
        ts_scalar = float(batch.ts.max().item())
        ts_val = batch.ts.to(dev)

        if self.memory_updater is not None:
            # Step 3: read old memory for mailbox
            src_mem_prev = self.memory.gather(src).detach().clone()
            dst_mem_prev = self.memory.gather(dst).detach().clone()

            # Step 4: update memory for src+dst nodes only
            all_nodes = torch.unique(torch.cat([src, dst], dim=0))
            mail = self.memory.read_mailbox(all_nodes)
            mem_cur = self.memory.gather(all_nodes)
            updated_mem = self.memory_updater(mail, mem_cur)
            update_ts = torch.full((all_nodes.numel(),), ts_scalar, dtype=torch.float32, device=dev)
            self.memory._apply_memory_update(all_nodes, updated_mem.detach(), update_ts)

            # Step 5: write mailbox
            self.memory.write_mailbox(src, dst, src_mem_prev, dst_mem_prev, batch.edge_feat.to(dev), ts_val)

            # Step 6+7: forward (conv reads updated memory via srcdata['h'])
            pos_logits, neg_logits = self._scores_direct(prepared)
        else:
            pos_logits, neg_logits = self._scores(prepared)
            all_nodes = torch.unique(torch.cat([src, dst], dim=0))

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss

        # Step 8: backward + optimizer
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._sync_updater_params()

        # Step 9: submit non-blocking async syncs
        t_submit0 = time.perf_counter()
        update_ids = torch.unique(torch.cat([src, dst], dim=0))
        update_vals = self.memory.gather(update_ids)
        update_ts_vec = torch.full((update_ids.numel(),), ts_scalar, dtype=torch.float32, device=dev)
        self.memory.submit_async_memory_sync(self.dist_ctx, update_ids, update_vals.detach(), update_ts_vec)
        mail_slots = self.memory.read_mailbox(update_ids)
        mail_ts = self.memory.read_mailbox_ts(update_ids)
        self.memory.submit_async_mail_sync(self.dist_ctx, update_ids, mail_slots, mail_ts)
        sync_submit_ms = (time.perf_counter() - t_submit0) * 1000.0
        step_ms = (time.perf_counter() - t_step0) * 1000.0

        metrics = self._metric_dict(pos_logits, neg_logits)
        return {
            "loss": float(loss.item()),
            "predictions": torch.sigmoid(pos_logits).detach().cpu().tolist(),
            "targets": [1 for _ in range(batch.size)],
            "meta": {
                "split": batch.split,
                "batch_size": batch.size,
                "sample": prepared.sample.describe(),
                "metrics": metrics,
                "memory": self.memory.describe(),
                "sync_wait_ms": sync_wait_ms,
                "sync_submit_ms": sync_submit_ms,
                "step_ms": step_ms,
            },
        }

    def eval_step(self, batch: CTDGDataBatch, split: str) -> dict[str, Any]:
        """Execute one evaluation step (no-grad, with memory update).

        Memory and mailbox are still updated during eval to maintain
        temporal consistency (mirrors the MemShare validation loop).

        Args:
            batch: A :class:`CTDGDataBatch` mini-batch.
            split: ``"val"`` or ``"test"``.

        Returns:
            Dict with ``"loss"``, ``"predictions"``, ``"targets"``, ``"meta"``.
        """
        self.memory.wait_pending_syncs()
        self.model.eval()
        dev = self.device
        with torch.no_grad():
            prepared = self._prepare(batch)
            src = batch.src.to(dev)
            dst = batch.dst.to(dev)
            ts_scalar = float(batch.ts.max().item())
            ts_val = batch.ts.to(dev)

            if self.memory_updater is not None:
                # Keep memory updated during eval (mirrors MemShare mailbox.update_* in val loop)
                src_mem_prev = self.memory.gather(src).detach().clone()
                dst_mem_prev = self.memory.gather(dst).detach().clone()

                all_nodes = torch.unique(torch.cat([src, dst], dim=0))
                mail = self.memory.read_mailbox(all_nodes)
                mem_cur = self.memory.gather(all_nodes)
                updated_mem = self.memory_updater(mail, mem_cur)
                update_ts = torch.full((all_nodes.numel(),), ts_scalar, dtype=torch.float32, device=dev)
                self.memory._apply_memory_update(all_nodes, updated_mem.detach(), update_ts)
                self.memory.write_mailbox(src, dst, src_mem_prev, dst_mem_prev,
                                          batch.edge_feat.to(dev), ts_val)
                pos_logits, neg_logits = self._scores_direct(prepared)
            else:
                pos_logits, neg_logits = self._scores(prepared)

            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss = pos_loss + neg_loss
            metrics = self._metric_dict(pos_logits, neg_logits)
        return {
            "loss": float(loss.item()),
            "predictions": torch.sigmoid(pos_logits).cpu().tolist(),
            "targets": [1 for _ in range(batch.size)],
            "meta": {
                "split": split,
                "batch_size": batch.size,
                "metrics": metrics,
                "memory": self.memory.describe(),
            },
        }

    def predict_step(self, batch: CTDGDataBatch) -> dict[str, Any]:
        result = self.eval_step(batch, split=batch.split)
        result["meta"]["predict"] = True
        return result
