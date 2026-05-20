from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from atc_starrygl_lib.core.types import Batch, EdgePredOutput
from atc_starrygl_lib.memory import RuntimeAsyncMemoryUpdater
from .layers import TransformerAttentionLayer, IdentityNormLayer, JODIETimeEmbedding
from .memory_updater import GRUMemoryUpdater, RNNMemoryUpdater, TransformerMemoryUpdater
from ..shared.edge_predictor import EdgePredictor


class GeneralModel(nn.Module):
    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        sample_param: dict,
        memory_param: dict,
        gnn_param: dict,
        train_param: dict,
        num_nodes: int | None = None,
        mailbox: Any = None,
        runtime_memory_updater: RuntimeAsyncMemoryUpdater | None = None,
        combined: bool = False,
    ):
        super().__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        self.gnn_param = gnn_param
        self.train_param = train_param

        if 'dim_out' not in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']

        if memory_param['type'] == 'node':
            dim_in = 2 * memory_param['dim_out'] + dim_edge
            dim_hid = memory_param['dim_out']
            dim_time = memory_param['dim_time']
            upd = memory_param['memory_update']
            if upd == 'gru':
                self.memory_updater = GRUMemoryUpdater(
                    memory_param, dim_in, dim_hid, dim_time, dim_node,
                )
            elif upd == 'rnn':
                self.memory_updater = RNNMemoryUpdater(
                    memory_param, dim_in, dim_hid, dim_time, dim_node,
                )
            elif upd == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(
                    memory_param, dim_in, dim_hid, dim_time, train_param,
                )
            else:
                raise NotImplementedError(f"memory_update={upd!r}")
            if runtime_memory_updater is not None:
                runtime_memory_updater.base_updater = self.memory_updater
                self.memory_updater = runtime_memory_updater
            self.dim_node_input = memory_param['dim_out']

        self.layers = nn.ModuleDict()
        arch = gnn_param['arch']
        if arch == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers[f'l0h{h}'] = TransformerAttentionLayer(
                    self.dim_node_input, dim_edge, gnn_param['dim_time'],
                    gnn_param['att_head'], train_param['dropout'],
                    train_param['att_dropout'], gnn_param['dim_out'],
                    combined=combined,
                )
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers[f'l{l}h{h}'] = TransformerAttentionLayer(
                        gnn_param['dim_out'], dim_edge, gnn_param['dim_time'],
                        gnn_param['att_head'], train_param['dropout'],
                        train_param['att_dropout'], gnn_param['dim_out'],
                        combined=False,
                    )
        elif arch == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers[f'l0h{h}'] = IdentityNormLayer(self.dim_node_input)
                if gnn_param.get('time_transform') == 'JODIE':
                    self.layers[f'l0h{h}t'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError(f"arch={arch!r}")

        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if gnn_param.get('combine') == 'rnn':
            self.combiner = nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])

    def forward(
        self,
        mfgs: list,
        metadata: dict[str, Tensor],
        neg_samples: int = 1,
        mode: str = 'triplet',
        async_param: Any = None,
        memory_update_spec: Any = None,
    ) -> tuple[Tensor, Tensor]:
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0], memory_update_spec if memory_update_spec is not None else async_param)

        out = []
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers[f'l{l}h{h}'](mfgs[l][h])
                if self.gnn_param.get('time_transform') == 'JODIE':
                    rst = self.layers[f'l0h{h}t'](
                        rst,
                        mfgs[l][h].srcdata['mem_ts'],
                        mfgs[l][h].srcdata['ts'],
                    )
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        out = out[0]
        if self.gnn_param.get('dyrep'):
            out = self.memory_updater.last_updated_memory

        pos_score, neg_score = self.edge_predictor(
            out[metadata['src_pos_index']],
            out[metadata['dst_pos_index']],
            h_neg_dst=out[metadata['dst_neg_index']],
            neg_samples=neg_samples,
            mode=mode,
        )
        return pos_score, neg_score

    def forward_batch(self, batch: Batch) -> EdgePredOutput:
        """New-style entry point: accepts Batch, returns EdgePredOutput."""
        assert batch.pos_src is not None and batch.pos_dst is not None
        assert batch.neg_src is not None or batch.neg_dst is not None
        # mfgs must be stored in batch.graph by the data loader
        mfgs = batch.graph
        n = len(mfgs[0][0].dstdata.get('h', mfgs[0][0].srcdata['h']))
        # build compact index metadata from batch edge arrays
        # pos_src/pos_dst/neg_dst are indices into the mfg output node rows
        metadata = {
            'src_pos_index': batch.pos_src,
            'dst_pos_index': batch.pos_dst,
            'dst_neg_index': batch.neg_dst if batch.neg_dst is not None else batch.neg_src,
        }
        pos_score, neg_score = self.forward(mfgs, metadata)
        return EdgePredOutput(pos_score=pos_score.flatten(), neg_score=neg_score.flatten())

    @classmethod
    def from_config(
        cls,
        dim_node: int,
        dim_edge: int,
        num_nodes: int,
        config: dict,
        mailbox: Any = None,
        runtime_memory_updater: RuntimeAsyncMemoryUpdater | None = None,
    ) -> "GeneralModel":
        return cls(
            dim_node=dim_node,
            dim_edge=dim_edge,
            sample_param=config['sample'],
            memory_param=config['memory'],
            gnn_param=config['gnn'],
            train_param=config['train'],
            num_nodes=num_nodes,
            mailbox=mailbox,
            runtime_memory_updater=runtime_memory_updater,
        )
