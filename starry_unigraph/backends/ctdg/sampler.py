from __future__ import annotations

from dataclasses import dataclass

import torch

from starry_unigraph.native import BTSNativeSampler, build_temporal_neighbor_block, is_bts_sampler_available

from .data import TGTemporalDataset


@dataclass
class CTDGSampleOutput:
    split: str
    root_nodes: torch.Tensor
    timestamps: torch.Tensor
    sampled_nodes: torch.Tensor
    sampled_eids: torch.Tensor
    blocks: list[object]

    def describe(self) -> dict[str, int]:
        return {
            "root_count": int(self.root_nodes.numel()),
            "sampled_node_count": int(self.sampled_nodes.numel()),
            "sampled_edge_count": int(self.sampled_eids.numel()),
            "num_layers": len(self.blocks),
        }


class NativeTemporalSampler:
    def __init__(
        self,
        dataset: TGTemporalDataset,
        fanout: list[int],
        history: int,
        strategy: str,
        workers: int = 1,
    ):
        self.dataset = dataset
        self.fanout = list(fanout)
        self.history = int(history)
        self.strategy = str(strategy)
        self.workers = int(workers)
        self._samplers: dict[str, BTSNativeSampler] = {}

    def _build_sampler(self, split: str) -> BTSNativeSampler:
        graph = self.dataset.sampler_graph(split)
        tnb = build_temporal_neighbor_block(
            graph_name=f"{self.dataset.root.name}_{split}",
            row=graph["row"],
            col=graph["col"],
            num_nodes=self.dataset.num_nodes,
            eid=graph["eid"],
            timestamp=graph["ts"],
        )
        node_part = torch.zeros(self.dataset.num_nodes, dtype=torch.int32)
        edge_part = torch.zeros(graph["eid"].numel(), dtype=torch.int32)
        return BTSNativeSampler(
            tnb=tnb,
            num_nodes=self.dataset.num_nodes,
            num_edges=graph["eid"].numel(),
            num_layers=self.history,
            fanout=self.fanout,
            workers=self.workers,
            policy=self.strategy,
            edge_part=edge_part,
            node_part=node_part,
        )

    def sample(self, split: str, root_nodes: torch.Tensor, timestamps: torch.Tensor) -> CTDGSampleOutput:
        if not is_bts_sampler_available():
            raise RuntimeError("Native BTS sampler is not available")
        sampler = self._samplers.get(split)
        if sampler is None:
            sampler = self._build_sampler(split)
            self._samplers[split] = sampler
        blocks = sampler.sample_from_nodes(root_nodes.long().cpu(), timestamps.float().cpu())
        sampled_nodes = []
        sampled_eids = []
        for block in blocks:
            sampled_nodes.append(block.sample_nodes().long().cpu())
            sampled_eids.append(block.eid().long().cpu())
        merged_nodes = torch.unique(torch.cat([root_nodes.long().cpu(), *sampled_nodes], dim=0))
        merged_eids = torch.unique(torch.cat(sampled_eids, dim=0)) if sampled_eids else torch.empty(0, dtype=torch.long)
        return CTDGSampleOutput(
            split=split,
            root_nodes=root_nodes.long().cpu(),
            timestamps=timestamps.float().cpu(),
            sampled_nodes=merged_nodes,
            sampled_eids=merged_eids,
            blocks=list(blocks),
        )
