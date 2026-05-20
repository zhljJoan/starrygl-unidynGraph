from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

FEATURE_FORMAT = "atc_feature_v1"


def build_all_feature_artifacts(
    *,
    rank_artifacts: list[dict[str, Any]],
    node_feat: Tensor | None = None,
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
    edge_label: Tensor | None = None,
    node_feat_time_varying: bool = False,
    node_label_time_varying: bool = False,
) -> list[dict[str, Any]]:
    return [
        build_feature_artifact(
            rank_artifact=rank_artifact,
            node_feat=node_feat,
            edge_feat=edge_feat,
            node_label=node_label,
            edge_label=edge_label,
            node_feat_time_varying=node_feat_time_varying,
            node_label_time_varying=node_label_time_varying,
        )
        for rank_artifact in rank_artifacts
    ]


def build_feature_artifact(
    *,
    rank_artifact: dict[str, Any],
    node_feat: Tensor | None = None,
    edge_feat: Tensor | None = None,
    node_label: Tensor | None = None,
    edge_label: Tensor | None = None,
    node_feat_time_varying: bool = False,
    node_label_time_varying: bool = False,
) -> dict[str, Any]:
    node_ids = rank_artifact["local_node_ids"].long().cpu().contiguous()
    edge_ids = rank_artifact["local_edge_ids"].long().cpu().contiguous()
    return {
        "format": FEATURE_FORMAT,
        "rank": int(rank_artifact["rank"]),
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "node_feat": _select_node_tensor(node_feat, node_ids, time_varying=bool(node_feat_time_varying)),
        "edge_feat": _select_edge_tensor(edge_feat, edge_ids),
        "node_label": _select_node_tensor(node_label, node_ids, time_varying=bool(node_label_time_varying)),
        "edge_label": _select_edge_tensor(edge_label, edge_ids),
        "node_feat_time_varying": bool(node_feat_time_varying),
        "node_label_time_varying": bool(node_label_time_varying),
    }


def write_feature_artifacts(root: str | Path, feature_artifacts: list[dict[str, Any]]) -> None:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for artifact in feature_artifacts:
        rank = int(artifact["rank"])
        torch.save(artifact, root / f"feature_{rank:03d}.pt")


def _select_node_tensor(tensor: Tensor | None, node_ids: Tensor, *, time_varying: bool) -> Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.cpu().contiguous()
    if time_varying:
        return tensor.index_select(1, node_ids).contiguous()
    return tensor.index_select(0, node_ids).contiguous()


def _select_edge_tensor(tensor: Tensor | None, edge_ids: Tensor) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.cpu().contiguous().index_select(0, edge_ids).contiguous()
