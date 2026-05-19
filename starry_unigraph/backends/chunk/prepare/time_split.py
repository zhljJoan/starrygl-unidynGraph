import importlib
import torch


def _load_adaptive_split():
    return importlib.import_module("starry_unigraph.lib.adaptive_split_cpp")


def time_split(src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor, batch_size: int,
               graph_features=None, alpha=None, beta=None, aggl=None,
               enable_drop=False, drop_rate=0.0, window_size=1000):

    adaptive_split_cpp = _load_adaptive_split()

    try:
        result = adaptive_split_cpp.adaptive_split(
            src.cpu().long().contiguous(),
            dst.cpu().long().contiguous(),
            ts.cpu().double().contiguous(),
            int(batch_size),
            float(1.0 if graph_features is None else graph_features),
            float(alpha if alpha is not None else 1.0),
            float(beta if beta is not None else 0.5),
            float(aggl if aggl is not None else 0.0),
            bool(enable_drop),
            float(drop_rate),
            int(window_size),
        )
        return result.group_index, result.keep_indices
    except Exception:
        num_events = int(ts.numel())
        keep_indices = torch.arange(num_events, dtype=torch.long)
        group_index = torch.div(keep_indices, max(1, int(batch_size)), rounding_mode="floor")
        return group_index, keep_indices
