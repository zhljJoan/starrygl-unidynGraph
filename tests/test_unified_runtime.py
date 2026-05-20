import torch

from atc_starrygl_lib.core.config import normalize_config
from atc_starrygl_lib.runtime.unified import SampledBlockGCNEncoder


def test_config_infers_snapshot_full_graph_for_dtdg_model_without_sampling() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "tgcn"},
        "task": {"name": "node_regression"},
    })

    assert cfg["graph"]["mode"] == "dtdg"
    assert cfg["runtime"]["execution_plan"] == "snapshot_full_graph"


def test_config_routes_dtdg_model_with_sampling_to_temporal_sampling() -> None:
    cfg = normalize_config({
        "graph": {"source": "x"},
        "model": {"name": "gcn"},
        "sampling": {"fanouts": [5, 5]},
        "task": {"name": "edge_prediction"},
    })

    assert cfg["graph"]["mode"] == "ctdg"
    assert cfg["runtime"]["execution_plan"] == "temporal_sampling"


def test_sampled_block_gcn_encoder_uses_sampled_block_features() -> None:
    class Block:
        is_block = True

        def __init__(self) -> None:
            self.srcdata = {"h": torch.ones(3, 2)}
            self.dstdata = {}
            self.edata = {"gcn_norm": torch.ones(2)}

        def num_src_nodes(self):
            return 3

        def num_dst_nodes(self):
            return 2

        def local_scope(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update_all(self, _msg, _reduce):
            self.dstdata["x"] = self.srcdata["x"][:2]

    encoder = SampledBlockGCNEncoder(2, 4, num_layers=1)
    out = encoder.encode([[Block()]])

    assert out.shape == (2, 4)
