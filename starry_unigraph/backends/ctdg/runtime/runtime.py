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

CTDG_CHAIN = "sample->feature_fetch->state_fetch->memory_updater->neighbor_attention_aggregate->message_generate->state_writeback"

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
    metadata: dict[str, torch.Tensor]
    model_output: Any | None = None


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

    def _empty_result(self, batch: CTDGDataBatch, split: str) -> dict[str, Any]:
        return {
            "loss": 0.0,
            "predictions": [],
            "targets": [],
            "meta": {
                "split": split,
                "batch_size": 0,
                "metrics": {},
                "memory": self.memory.describe(),
                "empty_batch": True,
                "chain": CTDG_CHAIN,
                "stage_payloads": {},
                "async_ops": [],
            },
        }

    def _ddp_safe_zero_loss(self) -> torch.Tensor:
        model_core = self.model.module if hasattr(self.model, "module") else self.model
        zero = next(model_core.parameters()).sum() * 0.0
        if self.memory_updater is not None:
            zero = zero + next(self.memory_updater.parameters()).sum() * 0.0
        return zero

    def iter_batches(self, split: str, batch_size: int) -> Iterable[CTDGDataBatch]:
        yield from self.dataset.iter_batches(
            split=split,
            batch_size=batch_size,
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
        num_pos = batch.src.numel()
        roots = torch.cat([batch.src, batch.dst, neg_dst], dim=0).long()
        ts = torch.cat([batch.ts, batch.ts, batch.ts], dim=0).float()
        metadata = {
            "src_pos_index": torch.arange(0, num_pos, dtype=torch.long),
            "dst_pos_index": torch.arange(num_pos, 2 * num_pos, dtype=torch.long),
            "dst_neg_index": torch.arange(2 * num_pos, 3 * num_pos, dtype=torch.long),
            "seed": roots,
            "seed_ts": ts,
        }
        sample = self.sampler.sample(batch.split if batch.split != "predict" else "test", roots, ts)
        return CTDGPreparedBatch(batch=batch, neg_dst=neg_dst, sample=sample, metadata=metadata)

    def _conv_all(
        self,
        prepared: CTDGPreparedBatch,
        updated_nodes: torch.Tensor | None = None,
        updated_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dev = self.device
        D = self.memory.hidden_dim
        model_core = self.model.module if hasattr(self.model, "module") else self.model
        sample = prepared.sample

        if model_core.conv is None or not sample.blocks:
            return torch.zeros(sample.root_nodes.numel(), D, device=dev)

        roots = sample.root_nodes.to(dev)   # [R]

        # cache edge features on device
        if not hasattr(self, '_edge_feat_gpu'):
            self._edge_feat_gpu = self.dataset.edge_feat.to(dev)

        dgl_block = build_dgl_block(
            sample.blocks[0],
            roots=roots,
            root_ts=sample.timestamps.to(dev),
            edge_feat_all=self._edge_feat_gpu, device=dev,
        )
        if dgl_block is None:
            return torch.zeros(roots.numel(), D, device=dev)
        dgl_block, metadata = dgl_block
        prepared.metadata.update({k: v.detach().cpu() for k, v in metadata.items()})

        # MemShare prepares one MFG and runs one GNN forward.  Start from the
        # persistent memory snapshot, then overlay current-batch updated memory
        # without detaching so the memory updater is on the loss path.
        block_ids = dgl_block.srcdata['__ID'].long()
        h, _, _, _ = self.memory.fetch_node_state(self.dist_ctx, block_ids)
        h = h.to(dev).detach()
        if updated_nodes is not None and updated_memory is not None and updated_nodes.numel() > 0:
            update_ids = updated_nodes.to(dev).long()
            order = torch.argsort(update_ids)
            sorted_ids = update_ids[order]
            sorted_mem = updated_memory.to(dev)[order]
            pos = torch.searchsorted(sorted_ids, block_ids)
            safe_pos = pos.clamp(max=sorted_ids.numel() - 1)
            valid = (pos < sorted_ids.numel()) & (sorted_ids[safe_pos] == block_ids)
            if valid.any():
                h[valid] = sorted_mem[pos[valid]]
        if self.dataset.node_feat is not None:
            node_feat = self.dataset.node_feat[block_ids.cpu()].to(dev).float()
            if getattr(model_core, "node_feat_map", None) is not None:
                h = h + model_core.node_feat_map(node_feat)
            elif node_feat.shape[1] == h.shape[1]:
                h = h + node_feat
        dgl_block.srcdata['h'] = h

        return model_core.conv(dgl_block)  # [n_roots, D]

    def _scores_direct(
        self,
        prepared: CTDGPreparedBatch,
        updated_nodes: torch.Tensor | None = None,
        updated_memory: torch.Tensor | None = None,
    ) -> Any:
        dev = self.device
        model_core = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_core, "forward_mfg"):
            sample = prepared.sample
            roots = sample.root_nodes.to(dev)
            if not hasattr(self, '_edge_feat_gpu'):
                self._edge_feat_gpu = self.dataset.edge_feat.to(dev)
            dgl_block = build_dgl_block(
                sample.blocks[0],
                roots=roots,
                root_ts=sample.timestamps.to(dev),
                edge_feat_all=self._edge_feat_gpu,
                device=dev,
            )
            if dgl_block is None:
                raise RuntimeError("CTDG forward_mfg received an empty sampled block")
            dgl_block, metadata = dgl_block
            prepared.metadata.update({k: v.detach().cpu() for k, v in metadata.items()})
            block_ids = dgl_block.srcdata["__ID"].long()
            memory, memory_ts, mailbox, _ = self.memory.fetch_node_state(self.dist_ctx, block_ids)
            node_feat = None
            if self.dataset.node_feat is not None:
                node_feat = self.dataset.node_feat[block_ids.cpu()].to(dev).float()
            edge_feat = prepared.batch.edge_feat.to(dev)
            output = model_core.forward_mfg(
                dgl_block,
                {k: v.to(dev) for k, v in prepared.metadata.items() if k.endswith("_index")},
                memory=memory.to(dev),
                memory_ts=memory_ts.to(dev),
                mailbox=mailbox.to(dev),
                node_feat=node_feat,
                edge_feat=edge_feat,
            )
            prepared.model_output = output
            return output
        conv_all = self._conv_all(prepared, updated_nodes=updated_nodes, updated_memory=updated_memory)
        meta = {k: v.to(dev) for k, v in prepared.metadata.items() if k.endswith("_index")}
        src_conv = conv_all[meta["src_pos_index"]]
        dst_conv = conv_all[meta["dst_pos_index"]]
        neg_conv = conv_all[meta["dst_neg_index"]]
        output = self.model(src_conv=src_conv, dst_conv=dst_conv, neg_conv=neg_conv)
        prepared.model_output = output
        return output

    def _scores(self, prepared: CTDGPreparedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        output = self._scores_direct(prepared)
        return output.pos_logits, output.neg_logits

    def _apply_model_state(self, output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        update_ids = output.updated_ids.to(dev).long()
        update_vals = output.updated_memory.to(dev)
        update_ts = output.updated_ts.to(dev).float()
        self.memory._apply_memory_update(update_ids, update_vals.detach(), update_ts.detach())
        mail_ids = output.mail_ids.to(dev).long()
        mail_slots = output.mail_slots.to(dev)
        mail_ts = output.mail_ts.to(dev).float()
        self.memory._write_slots(mail_ids.detach(), mail_slots.detach(), mail_ts.detach())
        return update_ids, update_vals, update_ts, mail_ids, mail_slots

    def _latest_node_times(
        self,
        node_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_ids.numel() == 0:
            return node_ids, timestamps
        order = torch.argsort(node_ids)
        sorted_nodes = node_ids[order]
        sorted_ts = timestamps[order]
        unique_nodes, inverse = torch.unique_consecutive(sorted_nodes, return_inverse=True)
        latest_ts = torch.full(
            (unique_nodes.numel(),),
            -1.0,
            dtype=sorted_ts.dtype,
            device=sorted_ts.device,
        )
        latest_ts.scatter_reduce_(0, inverse, sorted_ts, reduce="amax", include_self=False)
        return unique_nodes, latest_ts

    def _lookup_by_node(
        self,
        query_nodes: torch.Tensor,
        sorted_nodes: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        pos = torch.searchsorted(sorted_nodes, query_nodes)
        safe_pos = pos.clamp(max=sorted_nodes.numel() - 1)
        valid = (pos < sorted_nodes.numel()) & (sorted_nodes[safe_pos] == query_nodes)
        out = torch.zeros(query_nodes.numel(), values.size(-1), dtype=values.dtype, device=values.device)
        if valid.any():
            out[valid] = values[safe_pos[valid]]
        return out

    def _mail_slots_for_edges(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_mem: torch.Tensor,
        dst_mem: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E = max(0, self.memory.edge_feat_dim)
        efeat = edge_feat.float().to(src_mem.device)
        if efeat.size(-1) >= E:
            efeat = efeat[..., :E]
        else:
            efeat = torch.cat(
                [efeat, torch.zeros(*efeat.shape[:-1], E - efeat.size(-1), device=efeat.device)],
                dim=-1,
            )
        src_slot = torch.cat([src_mem, dst_mem, efeat], dim=-1)
        dst_slot = torch.cat([dst_mem, src_mem, efeat], dim=-1)
        return torch.cat([src, dst], dim=0), torch.cat([src_slot, dst_slot], dim=0)

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

        if batch.is_empty:
            model_core = self.model.module if hasattr(self.model, "module") else self.model
            if self.dist_ctx.is_distributed:
                empty_ids = torch.empty(0, dtype=torch.long, device=dev)
                # Keep distributed collectives aligned with non-empty ranks:
                # memory-updater fetch, conv fetch, memory sync, mailbox sync.
                if hasattr(model_core, "forward_mfg"):
                    self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                    self.memory.submit_async_memory_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.hidden_dim, dtype=torch.float32, device=dev),
                        torch.empty(0, dtype=torch.float32, device=dev),
                    )
                    self.memory.submit_async_mail_slot_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.slot_width, dtype=torch.float32, device=dev),
                        torch.empty(0, dtype=torch.float32, device=dev),
                    )
                elif self.memory_updater is not None:
                    self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                    self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                    self.memory.submit_async_memory_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.hidden_dim, dtype=torch.float32, device=dev),
                        torch.empty(0, dtype=torch.float32, device=dev),
                    )
                    self.memory.submit_async_mail_slot_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.slot_width, dtype=torch.float32, device=dev),
                        torch.empty(0, dtype=torch.float32, device=dev),
                    )
                else:
                    self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                    self.memory.submit_async_memory_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.hidden_dim, dtype=torch.float32, device=dev),
                        torch.empty(0, dtype=torch.float32, device=dev),
                    )
                    self.memory.submit_async_mail_sync(
                        self.dist_ctx,
                        empty_ids,
                        torch.empty(0, self.memory.mailbox_slots, self.memory.slot_width, dtype=torch.float32, device=dev),
                        torch.empty(0, self.memory.mailbox_slots, dtype=torch.float32, device=dev),
                    )
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._ddp_safe_zero_loss()
            loss.backward()
            self.optimizer.step()
            self._sync_updater_params()
            result = self._empty_result(batch, split=batch.split)
            result["meta"]["sync_wait_ms"] = sync_wait_ms
            result["meta"]["sync_submit_ms"] = 0.0
            result["meta"]["step_ms"] = (time.perf_counter() - t_step0) * 1000.0
            return result

        # Step 2: negative sampling + BTS sampling
        prepared = self._prepare(batch)
        src = batch.src.to(dev)
        dst = batch.dst.to(dev)
        neg = prepared.neg_dst.to(dev)
        ts_scalar = float(batch.ts.max().item())
        ts_val = batch.ts.to(dev)
        model_core = self.model.module if hasattr(self.model, "module") else self.model

        if hasattr(model_core, "forward_mfg"):
            model_output = self._scores_direct(prepared)
            pos_logits, neg_logits = model_output.pos_logits, model_output.neg_logits
            update_ids, update_vals, update_ts_vec, mail_update_ids, mail_update_slots = self._apply_model_state(model_output)
            mail_update_ts = model_output.mail_ts.to(dev).float()
            all_nodes = torch.unique(update_ids)
        elif self.memory_updater is not None:
            # Step 3: update memory for src+dst nodes only
            all_nodes, update_ts = self._latest_node_times(
                torch.cat([src, dst], dim=0),
                torch.cat([ts_val, ts_val], dim=0).float(),
            )
            mem_cur, mem_ts, mail, _ = self.memory.fetch_node_state(self.dist_ctx, all_nodes)
            updated_mem = self.memory_updater(mail, mem_cur, node_ts=update_ts, memory_ts=mem_ts)
            self.memory._apply_memory_update(all_nodes, updated_mem.detach(), update_ts)

            # Step 5: write mailbox from updated memory (MemShare last_updated_memory)
            src_mem_cur = self._lookup_by_node(src, all_nodes, updated_mem).detach()
            dst_mem_cur = self._lookup_by_node(dst, all_nodes, updated_mem).detach()
            self.memory.write_mailbox(src, dst, src_mem_cur, dst_mem_cur, batch.edge_feat.to(dev), ts_val)
            mail_update_ids, mail_update_slots = self._mail_slots_for_edges(
                src,
                dst,
                src_mem_cur,
                dst_mem_cur,
                batch.edge_feat.to(dev),
            )

            # Step 6+7: forward (conv reads updated memory via srcdata['h'])
            model_output = self._scores_direct(
                prepared,
                updated_nodes=all_nodes,
                updated_memory=updated_mem,
            )
            pos_logits, neg_logits = model_output.pos_logits, model_output.neg_logits
            update_ids = all_nodes
            update_vals = updated_mem
            update_ts_vec = update_ts
            mail_update_ts = torch.cat([ts_val, ts_val], dim=0).float()
        else:
            pos_logits, neg_logits = self._scores(prepared)
            all_nodes, _ = self._latest_node_times(
                torch.cat([src, dst], dim=0),
                torch.cat([ts_val, ts_val], dim=0).float(),
            )
            update_ids = torch.unique(torch.cat([src, dst], dim=0))
            update_vals = self.memory.gather(update_ids)
            update_ts_vec = torch.full((update_ids.numel(),), ts_scalar, dtype=torch.float32, device=dev)
            mail_update_ids = update_ids
            mail_update_slots = None
            mail_update_ts = update_ts_vec

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
        self.memory.submit_async_memory_sync(self.dist_ctx, update_ids, update_vals.detach(), update_ts_vec)
        if mail_update_slots is not None:
            self.memory.submit_async_mail_slot_sync(
                self.dist_ctx,
                mail_update_ids,
                mail_update_slots,
                mail_update_ts,
            )
        else:
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
                "chain": CTDG_CHAIN,
                "stage_payloads": {
                    "state_transition": {"num_nodes": int(all_nodes.numel())},
                    "neighbor_attention_aggregate": prepared.sample.describe(),
                },
                "async_ops": ["memory_sync", "mail_sync"],
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
            if batch.is_empty:
                if self.dist_ctx.is_distributed:
                    empty_ids = torch.empty(0, dtype=torch.long, device=dev)
                    if self.memory_updater is not None:
                        self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                        self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                    else:
                        self.memory.fetch_node_state(self.dist_ctx, empty_ids)
                return self._empty_result(batch, split=split)
            prepared = self._prepare(batch)
            src = batch.src.to(dev)
            dst = batch.dst.to(dev)
            ts_scalar = float(batch.ts.max().item())
            ts_val = batch.ts.to(dev)
            model_core = self.model.module if hasattr(self.model, "module") else self.model

            if hasattr(model_core, "forward_mfg"):
                model_output = self._scores_direct(prepared)
                pos_logits, neg_logits = model_output.pos_logits, model_output.neg_logits
                self._apply_model_state(model_output)
            elif self.memory_updater is not None:
                # Keep memory updated during eval (mirrors MemShare mailbox.update_* in val loop)
                all_nodes, update_ts = self._latest_node_times(
                    torch.cat([src, dst], dim=0),
                    torch.cat([ts_val, ts_val], dim=0).float(),
                )
                mem_cur, mem_ts, mail, _ = self.memory.fetch_node_state(self.dist_ctx, all_nodes)
                updated_mem = self.memory_updater(mail, mem_cur, node_ts=update_ts, memory_ts=mem_ts)
                self.memory._apply_memory_update(all_nodes, updated_mem.detach(), update_ts)
                src_mem_cur = self._lookup_by_node(src, all_nodes, updated_mem).detach()
                dst_mem_cur = self._lookup_by_node(dst, all_nodes, updated_mem).detach()
                self.memory.write_mailbox(src, dst, src_mem_cur, dst_mem_cur,
                                          batch.edge_feat.to(dev), ts_val)
                model_output = self._scores_direct(
                    prepared,
                    updated_nodes=all_nodes,
                    updated_memory=updated_mem,
                )
                pos_logits, neg_logits = model_output.pos_logits, model_output.neg_logits
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
                "chain": CTDG_CHAIN,
                "stage_payloads": {
                    "state_transition": {"num_nodes": int(src.numel() + dst.numel())},
                    "neighbor_attention_aggregate": prepared.sample.describe(),
                },
                "async_ops": [],
            },
        }

    def predict_step(self, batch: CTDGDataBatch) -> dict[str, Any]:
        result = self.eval_step(batch, split=batch.split)
        result["meta"]["predict"] = True
        return result
