import importlib
import torch


def _load_adaptive_split():
    return importlib.import_module("starry_unigraph.lib.adaptive_split_cpp")


def time_split(src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor, batch_size: int,
               graph_features=None, alpha=None, beta=None, aggl=None,
               enable_drop=False, drop_rate=0.0, window_size=1000):

    adaptive_split_cpp = _load_adaptive_split()

    result = adaptive_split_cpp.adaptive_split(
        src, dst, ts,
        base_batch_size=batch_size,
        graph_feature=graph_features,
        alpha=alpha if alpha is not None else 1.0,
        beta=beta if beta is not None else 0.5,
        aggl=aggl if aggl is not None else 0.0,
        enable_drop=enable_drop,
        drop_rate=drop_rate,
        window_size=window_size,
    )

    return result.group_index, result.keep_indices
