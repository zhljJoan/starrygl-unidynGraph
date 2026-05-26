#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor


SPLITS = ("train", "val", "test")


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    time_ranges: dict[str, dict[str, float | int]] = field(default_factory=dict)
    rank_time_ranges: list[dict[str, float | int | str | None]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_stat(self, key: str, value: int) -> None:
        self.stats[key] = self.stats.get(key, 0) + int(value)


def check_artifact_root(
    artifact_root: str | Path,
    *,
    check_ctdg: bool = True,
    check_dtdg: bool = True,
    strict_concat_order: bool = False,
    max_examples: int = 8,
    splits: Iterable[str] = SPLITS,
) -> CheckReport:
    root = Path(artifact_root)
    report = CheckReport()
    graph_path = root / "graph.pt"
    if not graph_path.exists():
        report.errors.append(f"missing graph artifact: {graph_path}")
        return report

    graph = _load_pt(graph_path)
    if not isinstance(graph, dict):
        report.errors.append(f"graph artifact must be a dict: {graph_path}")
        return report

    rank_paths = sorted(root.glob("rank_*.pt"))
    part_paths = sorted(root.glob("partition_data_*.pt"))
    report.stats["num_edges"] = _num_edges(graph)
    report.stats["num_rank_artifacts"] = len(rank_paths)
    report.stats["num_partition_data_artifacts"] = len(part_paths)

    if check_ctdg:
        if rank_paths:
            ranks = [_load_pt(path) for path in rank_paths]
            _check_rank_edge_ownership(graph=graph, ranks=ranks, report=report, max_examples=max_examples)
            _check_ctdg_batches(
                graph=graph,
                ranks=ranks,
                report=report,
                strict_concat_order=strict_concat_order,
                max_examples=max_examples,
                splits=tuple(splits),
            )
        else:
            report.warnings.append("CTDG check skipped: no rank_*.pt files found")

    if check_dtdg:
        if part_paths:
            parts = [_load_pt(path) for path in part_paths]
            _check_dtdg_partition_batches(graph=graph, parts=parts, report=report, max_examples=max_examples)
        else:
            report.warnings.append("DTDG check skipped: no partition_data_*.pt files found")

    return report


def _check_rank_edge_ownership(
    *,
    graph: dict[str, Any],
    ranks: list[dict[str, Any]],
    report: CheckReport,
    max_examples: int,
) -> None:
    expected = torch.arange(_num_edges(graph), dtype=torch.long)
    rank_edges = [_rank_local_edges(rank) for rank in ranks]
    merged = torch.cat(rank_edges, dim=0) if rank_edges else torch.empty(0, dtype=torch.long)
    report.add_stat("ctdg_rank_edge_refs", int(merged.numel()))
    if not _same_multiset(merged, expected):
        _append_multiset_error(
            report,
            label="rank local_edge_ids do not form a one-to-one cover of global event positions",
            actual=merged,
            expected=expected,
            max_examples=max_examples,
        )


def _check_ctdg_batches(
    *,
    graph: dict[str, Any],
    ranks: list[dict[str, Any]],
    report: CheckReport,
    strict_concat_order: bool,
    max_examples: int,
    splits: tuple[str, ...],
) -> None:
    rank_local_rows = [_rank_local_row(rank, graph=graph) for rank in ranks]
    for rank_idx, rank in enumerate(ranks):
        if bool(rank.get("preserve_replica_history", False)):
            report.warnings.append(
                f"CTDG rank {rank_idx} uses preserve_replica_history=True; "
                "memory-route max-timestamp uniqueness checks are skipped for that rank"
            )
    for split in splits:
        windows = _split_windows(graph, split)
        if windows.numel() == 0:
            continue
        route_indices = _split_window_global_indices(graph, split)
        if len(route_indices) != int(windows.size(0)):
            report.errors.append(
                f"CTDG {split} has {int(windows.size(0))} split windows but only "
                f"{len(route_indices)} matched graph.time_ptr_2 route slices"
            )
            route_indices = list(range(int(windows.size(0))))
        _check_ctdg_rank_batch_counts(ranks=ranks, split=split, expected_windows=int(windows.size(0)), report=report)
        split_begin = int(windows[0, 0])
        split_end = int(windows[-1, 1])
        split_ts = _positions_ts(graph, torch.arange(split_begin, split_end, dtype=torch.long))
        empty_rank_batches = 0
        report.add_stat(f"ctdg_{split}_windows", int(windows.size(0)))
        for batch_idx, (begin, end) in enumerate(windows.tolist()):
            begin_i, end_i = int(begin), int(end)
            expected_pos = torch.arange(begin_i, end_i, dtype=torch.long)
            expected_ts = _positions_ts(graph, expected_pos)
            per_rank = [
                _ctdg_rank_event_positions(rank=rank, graph=graph, split=split, batch_idx=batch_idx)
                for rank in ranks
            ]
            route_idx = int(route_indices[batch_idx])
            empty_rank_batches += sum(1 for pos in per_rank if int(pos.numel()) == 0)
            merged = torch.cat(per_rank, dim=0) if per_rank else torch.empty(0, dtype=torch.long)
            report.add_stat(f"ctdg_{split}_events", int(expected_pos.numel()))
            report.add_stat("ctdg_epoch_batches", 1)

            _check_rank_subsets_match_window(
                graph=graph,
                split=split,
                batch_idx=batch_idx,
                begin=begin_i,
                end=end_i,
                expected_ts=expected_ts,
                per_rank=per_rank,
                report=report,
            )
            for rank_idx, (rank, event_pos, local_row) in enumerate(zip(ranks, per_rank, rank_local_rows)):
                report.rank_time_ranges.append(
                    _rank_time_range_row(
                        graph=graph,
                        split=split,
                        batch_idx=batch_idx,
                        route_idx=route_idx,
                        begin=begin_i,
                        end=end_i,
                        expected_ts=expected_ts,
                        rank_idx=rank_idx,
                        event_pos=event_pos,
                    )
                )
                if not bool(rank.get("preserve_replica_history", False)) and _rank_has_memory_route(rank):
                    _check_update_nodes_and_memory_route(
                        graph=graph,
                        rank=rank,
                        rank_idx=rank_idx,
                        route_idx=route_idx,
                        event_pos=event_pos,
                        local_row=local_row,
                        report=report,
                        max_examples=max_examples,
                    )
                elif not _rank_has_memory_route(rank):
                    report.add_stat("ctdg_memory_route_skipped_slices", 1)

            if not _same_multiset(merged, expected_pos):
                _append_multiset_error(
                    report,
                    label=f"CTDG {split} batch {batch_idx} cannot reconstruct global window [{begin_i}, {end_i})",
                    actual=merged,
                    expected=expected_pos,
                    max_examples=max_examples,
                )
                continue

            expected_tuples = _event_tuples(graph, expected_pos)
            actual_tuples = _event_tuples(graph, merged)
            if Counter(actual_tuples) != Counter(expected_tuples):
                report.errors.append(f"CTDG {split} batch {batch_idx} event payload mismatch after rank merge")

            for rank_idx, pos in enumerate(per_rank):
                if not _is_non_decreasing(pos):
                    report.errors.append(f"CTDG {split} batch {batch_idx} rank {rank_idx} event positions are not sorted")

            if strict_concat_order and not torch.equal(merged, expected_pos):
                report.errors.append(
                    f"CTDG {split} batch {batch_idx} rank-concat order differs from global event order; "
                    "sorted multiset still matches"
                )

            _check_rank_local_split_counts(
                ranks=ranks,
                split=split,
                batch_idx=batch_idx,
                per_rank=per_rank,
                report=report,
            )
        report.time_ranges[f"ctdg_{split}"] = _time_range_summary(
            ts=split_ts,
            windows=int(windows.size(0)),
            empty_rank_batches=empty_rank_batches,
        )


def _rank_time_range_row(
    *,
    graph: dict[str, Any],
    split: str,
    batch_idx: int,
    route_idx: int,
    begin: int,
    end: int,
    expected_ts: Tensor,
    rank_idx: int,
    event_pos: Tensor,
) -> dict[str, float | int | str | None]:
    local_ts = _positions_ts(graph, event_pos)
    return {
        "split": split,
        "batch_idx": int(batch_idx),
        "route_idx": int(route_idx),
        "rank": int(rank_idx),
        "global_begin": int(begin),
        "global_end": int(end),
        "global_event_count": int(end) - int(begin),
        "global_min_ts": float(expected_ts.min().item()) if expected_ts.numel() else None,
        "global_max_ts": float(expected_ts.max().item()) if expected_ts.numel() else None,
        "rank_event_count": int(event_pos.numel()),
        "rank_first_event_pos": int(event_pos[0].item()) if event_pos.numel() else None,
        "rank_last_event_pos": int(event_pos[-1].item()) if event_pos.numel() else None,
        "rank_min_ts": float(local_ts.min().item()) if local_ts.numel() else None,
        "rank_max_ts": float(local_ts.max().item()) if local_ts.numel() else None,
    }


def _check_update_nodes_and_memory_route(
    *,
    graph: dict[str, Any],
    rank: dict[str, Any],
    rank_idx: int,
    route_idx: int,
    event_pos: Tensor,
    local_row: Tensor,
    report: CheckReport,
    max_examples: int,
) -> None:
    update_ptr = torch.as_tensor(rank.get("update_node_ptr"), dtype=torch.long).cpu()
    update_nodes = torch.as_tensor(rank.get("update_node_ids"), dtype=torch.long).cpu()
    update_ts = torch.as_tensor(rank.get("update_node_ts")).cpu()
    if route_idx >= max(int(update_ptr.numel()) - 1, 0):
        report.errors.append(f"CTDG rank {rank_idx} route slice {route_idx} missing from update_node_ptr")
        return
    update_begin = int(update_ptr[route_idx])
    update_end = int(update_ptr[route_idx + 1])
    actual_nodes = update_nodes[update_begin:update_end].long()
    actual_ts = update_ts[update_begin:update_end]
    expected_nodes, expected_ts = _expected_update_nodes_for_rank(
        graph=graph,
        event_pos=event_pos,
        local_row=local_row,
        report=report,
        rank_idx=rank_idx,
        route_idx=route_idx,
    )
    report.add_stat("ctdg_update_node_refs", int(actual_nodes.numel()))
    if not torch.equal(actual_nodes, expected_nodes):
        _append_multiset_error(
            report,
            label=f"CTDG rank {rank_idx} route slice {route_idx} update_node_ids mismatch",
            actual=actual_nodes,
            expected=expected_nodes,
            max_examples=max_examples,
        )
    if actual_nodes.numel() == expected_nodes.numel() and actual_nodes.numel() > 0:
        if not torch.allclose(actual_ts.float(), expected_ts.float(), rtol=0.0, atol=1e-6):
            bad = (actual_ts.float() - expected_ts.float()).abs() > 1e-6
            examples = [
                {
                    "node": int(node),
                    "actual_ts": float(a),
                    "expected_max_ts": float(e),
                }
                for node, a, e in zip(actual_nodes[bad][:max_examples].tolist(), actual_ts[bad][:max_examples].tolist(), expected_ts[bad][:max_examples].tolist())
            ]
            report.errors.append(
                f"CTDG rank {rank_idx} route slice {route_idx} update_node_ts is not per-node max timestamp: {examples}"
            )

    route = rank.get("memory_route", {})
    send_ptr = torch.as_tensor(route.get("send_ptr"), dtype=torch.long).cpu()
    if route_idx >= max(int(send_ptr.numel()) - 1, 0):
        report.errors.append(f"CTDG rank {rank_idx} route slice {route_idx} missing from memory_route.send_ptr")
        return
    send_begin = int(send_ptr[route_idx])
    send_end = int(send_ptr[route_idx + 1])
    send_pos_all = torch.as_tensor(route.get("send_update_pos"), dtype=torch.long).cpu()
    send_pos = send_pos_all[send_begin:send_end]
    report.add_stat("ctdg_memory_route_send_refs", int(send_pos.numel()))
    report.add_stat("ctdg_memory_route_checked_slices", 1)
    _check_route_slice_lengths(route=route, rank_idx=rank_idx, route_idx=route_idx, begin=send_begin, end=send_end, report=report)
    if send_pos.numel() == 0:
        return
    out_of_range = (send_pos < update_begin) | (send_pos >= update_end)
    if bool(out_of_range.any().item()):
        report.errors.append(
            f"CTDG rank {rank_idx} route slice {route_idx} send_update_pos outside current update slice "
            f"[{update_begin}, {update_end}): {send_pos[out_of_range][:max_examples].tolist()}"
        )
        return
    sent_nodes = update_nodes.index_select(0, send_pos).long()
    sent_ts = update_ts.index_select(0, send_pos)
    valid, sent_expected_ts = _lookup_expected_ts(expected_nodes, expected_ts, sent_nodes)
    if not bool(valid.all().item()):
        report.errors.append(
            f"CTDG rank {rank_idx} route slice {route_idx} sends nodes not present in current batch max-ts table: "
            f"{sent_nodes[~valid][:max_examples].tolist()}"
        )
    if sent_expected_ts.numel() > 0 and not torch.allclose(sent_ts.float(), sent_expected_ts.float(), rtol=0.0, atol=1e-6):
        bad = (sent_ts.float() - sent_expected_ts.float()).abs() > 1e-6
        examples = [
            {
                "send_update_pos": int(pos),
                "node": int(node),
                "actual_ts": float(a),
                "expected_max_ts": float(e),
            }
            for pos, node, a, e in zip(
                send_pos[bad][:max_examples].tolist(),
                sent_nodes[bad][:max_examples].tolist(),
                sent_ts[bad][:max_examples].tolist(),
                sent_expected_ts[bad][:max_examples].tolist(),
            )
        ]
        report.errors.append(
            f"CTDG rank {rank_idx} route slice {route_idx} memory_route does not point at max-ts update rows: {examples}"
        )


def _check_route_slice_lengths(
    *,
    route: dict[str, Any],
    rank_idx: int,
    route_idx: int,
    begin: int,
    end: int,
    report: CheckReport,
) -> None:
    expected = int(end) - int(begin)
    for key in ("send_rank", "send_local_row", "send_dist_index"):
        value = route.get(key)
        if value is None:
            continue
        count = int(torch.as_tensor(value).cpu()[begin:end].numel())
        if count != expected:
            report.errors.append(
                f"CTDG rank {rank_idx} route slice {route_idx} memory_route.{key} count {count} "
                f"!= send_update_pos count {expected}"
            )


def _rank_has_memory_route(rank: dict[str, Any]) -> bool:
    route = rank.get("memory_route")
    return (
        rank.get("update_node_ptr") is not None
        and rank.get("update_node_ids") is not None
        and rank.get("update_node_ts") is not None
        and isinstance(route, dict)
        and route.get("send_ptr") is not None
        and route.get("send_update_pos") is not None
    )


def _check_ctdg_rank_batch_counts(
    *,
    ranks: list[dict[str, Any]],
    split: str,
    expected_windows: int,
    report: CheckReport,
) -> None:
    expected_ptr_len = int(expected_windows) + 1
    for rank_idx, rank in enumerate(ranks):
        packed = rank.get("split_event_pos", {}).get(split)
        if isinstance(packed, dict) and packed.get("ptr") is not None:
            ptr_len = int(torch.as_tensor(packed["ptr"]).numel())
            if ptr_len != expected_ptr_len:
                report.errors.append(
                    f"CTDG {split} rank {rank_idx} split_event_pos ptr length {ptr_len} "
                    f"does not match global window count {expected_windows}"
                )
        local_windows = rank.get("split_time_ptr", {}).get(split)
        if local_windows is not None:
            local_count = int(torch.as_tensor(local_windows).size(0))
            if local_count != int(expected_windows):
                report.errors.append(
                    f"CTDG {split} rank {rank_idx} split_time_ptr has {local_count} windows, "
                    f"expected {expected_windows}"
                )


def _check_rank_subsets_match_window(
    *,
    graph: dict[str, Any],
    split: str,
    batch_idx: int,
    begin: int,
    end: int,
    expected_ts: Tensor,
    per_rank: list[Tensor],
    report: CheckReport,
) -> None:
    if expected_ts.numel() == 0:
        return
    expected_min = float(expected_ts.min().item())
    expected_max = float(expected_ts.max().item())
    for rank_idx, event_pos in enumerate(per_rank):
        if event_pos.numel() == 0:
            continue
        below = event_pos < int(begin)
        above = event_pos >= int(end)
        if bool((below | above).any().item()):
            bad = event_pos[below | above][:8].tolist()
            report.errors.append(
                f"CTDG {split} batch {batch_idx} rank {rank_idx} has events outside "
                f"global window [{begin}, {end}): {bad}"
            )
            continue
        local_ts = _positions_ts(graph, event_pos)
        local_min = float(local_ts.min().item())
        local_max = float(local_ts.max().item())
        if local_min < expected_min or local_max > expected_max:
            report.errors.append(
                f"CTDG {split} batch {batch_idx} rank {rank_idx} timestamp range "
                f"[{local_min}, {local_max}] is outside global range [{expected_min}, {expected_max}]"
            )


def _check_rank_local_split_counts(
    *,
    ranks: list[dict[str, Any]],
    split: str,
    batch_idx: int,
    per_rank: list[Tensor],
    report: CheckReport,
) -> None:
    for rank_idx, (rank, event_pos) in enumerate(zip(ranks, per_rank)):
        local_windows = rank.get("split_time_ptr", {}).get(split)
        if local_windows is None:
            continue
        local_windows = torch.as_tensor(local_windows, dtype=torch.long).cpu()
        if batch_idx >= int(local_windows.size(0)):
            expected_count = 0
        else:
            expected_count = int(local_windows[batch_idx, 1]) - int(local_windows[batch_idx, 0])
        if expected_count != int(event_pos.numel()):
            report.errors.append(
                f"CTDG {split} batch {batch_idx} rank {rank_idx} split_time_ptr count "
                f"{expected_count} != split_event_pos count {int(event_pos.numel())}"
            )


def _check_dtdg_partition_batches(
    *,
    graph: dict[str, Any],
    parts: list[dict[str, Any]],
    report: CheckReport,
    max_examples: int,
) -> None:
    time_ptr = _as_long_2d(graph.get("time_ptr_2"))
    if time_ptr.numel() == 0:
        report.warnings.append("DTDG check skipped: graph.time_ptr_2 is empty")
        return
    expected_slices = int(time_ptr.size(0))
    all_begin = int(time_ptr[0, 0])
    all_end = int(time_ptr[-1, 1])
    all_ts = _positions_ts(graph, torch.arange(all_begin, all_end, dtype=torch.long))
    empty_rank_slices = 0
    for rank_idx, part in enumerate(parts):
        part_slices = _td_len(part.get("edge_ids"))
        if part_slices != expected_slices:
            report.errors.append(
                f"DTDG rank {rank_idx} partition_data has {part_slices} slices, expected {expected_slices}"
            )

    report.add_stat("dtdg_slices", expected_slices)
    for sid, (begin, end) in enumerate(time_ptr.tolist()):
        begin_i, end_i = int(begin), int(end)
        expected_pos = torch.arange(begin_i, end_i, dtype=torch.long)
        expected = Counter(_event_tuples(graph, expected_pos, include_ts=False))
        actual_tuples: list[tuple[int, int, int]] = []
        actual_edge_ids: list[Tensor] = []
        for rank_idx, part in enumerate(parts):
            if sid >= _td_len(part.get("edge_ids")):
                continue
            tuples, edge_ids = _dtdg_partition_slice_tuples(part=part, sid=sid, rank_idx=rank_idx, report=report)
            if int(edge_ids.numel()) == 0:
                empty_rank_slices += 1
            actual_tuples.extend(tuples)
            actual_edge_ids.append(edge_ids)
        actual = Counter(actual_tuples)
        report.add_stat("dtdg_expected_edges", int(expected_pos.numel()))
        if actual != expected:
            merged_edge_ids = torch.cat(actual_edge_ids, dim=0) if actual_edge_ids else torch.empty(0, dtype=torch.long)
            expected_edge_ids = _edge_ids(graph).index_select(0, expected_pos)
            _append_multiset_error(
                report,
                label=f"DTDG slice {sid} cannot reconstruct global window [{begin_i}, {end_i})",
                actual=merged_edge_ids,
                expected=expected_edge_ids,
                max_examples=max_examples,
            )
    report.time_ranges["dtdg_all"] = _time_range_summary(
        ts=all_ts,
        windows=expected_slices,
        empty_rank_batches=empty_rank_slices,
    )


def _dtdg_partition_slice_tuples(
    *,
    part: dict[str, Any],
    sid: int,
    rank_idx: int,
    report: CheckReport,
) -> tuple[list[tuple[int, int, int]], Tensor]:
    edge_ids = _td_item(part.get("edge_ids"), sid).long()
    dst_ids = _td_item(part.get("dst_ids"), sid).long()
    src_tail = _td_item(part.get("src_ids"), sid).long()
    edge_src = _td_item(part.get("edge_src"), sid).long()
    edge_dst = _td_item(part.get("edge_dst"), sid).long()
    if not (int(edge_ids.numel()) == int(edge_src.numel()) == int(edge_dst.numel())):
        report.errors.append(
            f"DTDG rank {rank_idx} slice {sid} edge_ids/edge_src/edge_dst length mismatch: "
            f"{int(edge_ids.numel())}/{int(edge_src.numel())}/{int(edge_dst.numel())}"
        )
        return [], edge_ids
    if edge_ids.numel() == 0:
        return [], edge_ids
    combined_src_ids = torch.cat([dst_ids, src_tail], dim=0).long()
    if int(edge_src.min().item()) < 0 or int(edge_src.max().item()) >= int(combined_src_ids.numel()):
        report.errors.append(f"DTDG rank {rank_idx} slice {sid} edge_src row out of range")
        return [], edge_ids
    if int(edge_dst.min().item()) < 0 or int(edge_dst.max().item()) >= int(dst_ids.numel()):
        report.errors.append(f"DTDG rank {rank_idx} slice {sid} edge_dst row out of range")
        return [], edge_ids
    src = combined_src_ids.index_select(0, edge_src)
    dst = dst_ids.index_select(0, edge_dst)
    return [
        (int(eid), int(s), int(d))
        for eid, s, d in zip(edge_ids.tolist(), src.tolist(), dst.tolist())
    ], edge_ids


def _ctdg_rank_event_positions(*, rank: dict[str, Any], graph: dict[str, Any], split: str, batch_idx: int) -> Tensor:
    packed = rank.get("split_event_pos", {}).get(split)
    if isinstance(packed, dict) and packed.get("data") is not None and packed.get("ptr") is not None:
        data = torch.as_tensor(packed["data"], dtype=torch.long).cpu().contiguous()
        ptr = torch.as_tensor(packed["ptr"], dtype=torch.long).cpu().contiguous()
        if batch_idx >= max(int(ptr.numel()) - 1, 0):
            return torch.empty(0, dtype=torch.long)
        begin = int(ptr[batch_idx])
        end = int(ptr[batch_idx + 1])
        return data[begin:end].long().contiguous()

    windows = _split_windows(graph, split)
    if batch_idx >= int(windows.size(0)):
        return torch.empty(0, dtype=torch.long)
    begin, end = (int(v) for v in windows[batch_idx].tolist())
    local_edge_ids = torch.sort(_rank_local_edges(rank)).values
    left = torch.searchsorted(local_edge_ids, torch.tensor(begin, dtype=torch.long))
    right = torch.searchsorted(local_edge_ids, torch.tensor(end, dtype=torch.long))
    return local_edge_ids[int(left):int(right)].long().contiguous()


def _split_windows(graph: dict[str, Any], split: str) -> Tensor:
    split_time_ptr = graph.get("split_time_ptr")
    if isinstance(split_time_ptr, dict) and split in split_time_ptr and split_time_ptr[split] is not None:
        return _as_long_2d(split_time_ptr[split])
    if split == "train":
        return _as_long_2d(graph.get("time_ptr_2"))
    return torch.zeros((0, 2), dtype=torch.long)


def _split_window_global_indices(graph: dict[str, Any], split: str) -> list[int]:
    global_windows = _as_long_2d(graph.get("time_ptr_2"))
    windows = _split_windows(graph, split)
    if windows.numel() == 0:
        return []
    if not isinstance(graph.get("split_time_ptr"), dict):
        if split == "train":
            return list(range(int(windows.size(0))))
        return []
    out: list[int] = []
    used = torch.zeros(int(global_windows.size(0)), dtype=torch.bool)
    cursor = 0
    for begin, end in windows.tolist():
        found = -1
        for idx in range(cursor, int(global_windows.size(0))):
            if bool(used[idx]):
                continue
            if int(global_windows[idx, 0]) == int(begin) and int(global_windows[idx, 1]) == int(end):
                found = idx
                cursor = idx + 1
                break
        if found < 0:
            for idx in range(int(global_windows.size(0))):
                if bool(used[idx]):
                    continue
                if int(global_windows[idx, 0]) == int(begin) and int(global_windows[idx, 1]) == int(end):
                    found = idx
                    break
        if found < 0:
            break
        used[found] = True
        out.append(int(found))
    return out


def _event_tuples(
    graph: dict[str, Any],
    positions: Tensor,
    *,
    include_ts: bool = True,
) -> list[tuple[int, int, int] | tuple[int, int, int, float]]:
    positions = positions.long().cpu()
    if positions.numel() == 0:
        return []
    edge_ids = _edge_ids(graph).index_select(0, positions)
    src = torch.as_tensor(graph["src"], dtype=torch.long).cpu().index_select(0, positions)
    dst = torch.as_tensor(graph["dst"], dtype=torch.long).cpu().index_select(0, positions)
    if not include_ts:
        return [(int(e), int(s), int(d)) for e, s, d in zip(edge_ids.tolist(), src.tolist(), dst.tolist())]
    ts = torch.as_tensor(graph["ts"]).cpu().index_select(0, positions)
    return [
        (int(e), int(s), int(d), float(t))
        for e, s, d, t in zip(edge_ids.tolist(), src.tolist(), dst.tolist(), ts.tolist())
    ]


def _rank_local_row(rank: dict[str, Any], *, graph: dict[str, Any]) -> Tensor:
    raw_local_node_ids = rank.get("local_node_ids")
    local_node_ids = (
        torch.empty(0, dtype=torch.long)
        if raw_local_node_ids is None
        else torch.as_tensor(raw_local_node_ids, dtype=torch.long).cpu().contiguous()
    )
    graph_nodes = int(graph.get("num_nodes", 0))
    graph_max = 0
    if _num_edges(graph) > 0:
        src_max = int(torch.as_tensor(graph["src"], dtype=torch.long).max().item())
        dst_max = int(torch.as_tensor(graph["dst"], dtype=torch.long).max().item())
        graph_max = max(src_max, dst_max) + 1
    local_max = int(local_node_ids.max().item()) + 1 if local_node_ids.numel() else 0
    size = max(graph_nodes, graph_max, local_max)
    local_row = torch.full((size,), -1, dtype=torch.long)
    if local_node_ids.numel() > 0:
        local_row[local_node_ids] = torch.arange(int(local_node_ids.numel()), dtype=torch.long)
    return local_row


def _expected_update_nodes_for_rank(
    *,
    graph: dict[str, Any],
    event_pos: Tensor,
    local_row: Tensor,
    report: CheckReport,
    rank_idx: int,
    route_idx: int,
) -> tuple[Tensor, Tensor]:
    if event_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    src = torch.as_tensor(graph["src"], dtype=torch.long).cpu().index_select(0, event_pos.long().cpu())
    dst = torch.as_tensor(graph["dst"], dtype=torch.long).cpu().index_select(0, event_pos.long().cpu())
    ts = torch.as_tensor(graph["ts"]).cpu().index_select(0, event_pos.long().cpu())
    nodes = torch.cat([src, dst], dim=0).long()
    node_ts = torch.cat([ts, ts], dim=0)
    in_range = nodes < int(local_row.numel())
    if not bool(in_range.all().item()):
        report.errors.append(
            f"CTDG rank {rank_idx} route slice {route_idx} has endpoint nodes beyond local_row capacity: "
            f"{nodes[~in_range][:8].tolist()}"
        )
    nodes = nodes[in_range]
    node_ts = node_ts[in_range]
    rows = local_row.index_select(0, nodes)
    keep = rows >= 0
    if not bool(keep.any().item()):
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=node_ts.dtype)
    kept_nodes = nodes[keep]
    kept_ts = node_ts[keep]
    unique, inverse = torch.unique(kept_nodes.long(), sorted=True, return_inverse=True)
    max_ts = torch.full((int(unique.numel()),), -float("inf"), dtype=kept_ts.dtype)
    max_ts.scatter_reduce_(0, inverse.long(), kept_ts, reduce="amax", include_self=True)
    return unique.long().contiguous(), max_ts.contiguous()


def _lookup_expected_ts(expected_nodes: Tensor, expected_ts: Tensor, nodes: Tensor) -> tuple[Tensor, Tensor]:
    if nodes.numel() == 0:
        return torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=expected_ts.dtype)
    if expected_nodes.numel() == 0:
        return torch.zeros(int(nodes.numel()), dtype=torch.bool), torch.empty(0, dtype=expected_ts.dtype)
    pos = torch.searchsorted(expected_nodes.long(), nodes.long())
    valid_pos = pos.clamp_max(max(int(expected_nodes.numel()) - 1, 0))
    valid = (pos < int(expected_nodes.numel())) & (expected_nodes.index_select(0, valid_pos) == nodes.long())
    out = torch.empty(int(nodes.numel()), dtype=expected_ts.dtype)
    out[valid] = expected_ts.index_select(0, pos[valid])
    out[~valid] = float("nan")
    return valid.bool(), out


def _positions_ts(graph: dict[str, Any], positions: Tensor) -> Tensor:
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.as_tensor(graph["ts"]).cpu().index_select(0, positions.long().cpu())


def _time_range_summary(*, ts: Tensor, windows: int, empty_rank_batches: int) -> dict[str, float | int]:
    if ts.numel() == 0:
        return {
            "windows": int(windows),
            "events": 0,
            "empty_rank_batches": int(empty_rank_batches),
            "min_ts": 0.0,
            "max_ts": 0.0,
        }
    return {
        "windows": int(windows),
        "events": int(ts.numel()),
        "empty_rank_batches": int(empty_rank_batches),
        "min_ts": float(ts.min().item()),
        "max_ts": float(ts.max().item()),
    }


def _append_multiset_error(
    report: CheckReport,
    *,
    label: str,
    actual: Tensor,
    expected: Tensor,
    max_examples: int,
) -> None:
    actual_counter = Counter(int(v) for v in actual.long().cpu().tolist())
    expected_counter = Counter(int(v) for v in expected.long().cpu().tolist())
    missing = _counter_diff_examples(expected_counter, actual_counter, max_examples)
    extra = _counter_diff_examples(actual_counter, expected_counter, max_examples)
    duplicated = sorted([value for value, count in actual_counter.items() if count > 1])[:max_examples]
    report.errors.append(
        f"{label}: expected_count={int(expected.numel())} actual_count={int(actual.numel())} "
        f"missing={missing} extra={extra} duplicated_actual={duplicated}"
    )


def _counter_diff_examples(a: Counter[int], b: Counter[int], max_examples: int) -> list[int]:
    out: list[int] = []
    for value in sorted(a.keys()):
        missing_count = a[value] - b.get(value, 0)
        if missing_count <= 0:
            continue
        out.extend([value] * int(missing_count))
        if len(out) >= max_examples:
            return out[:max_examples]
    return out


def _same_multiset(actual: Tensor, expected: Tensor) -> bool:
    if int(actual.numel()) != int(expected.numel()):
        return False
    if actual.numel() == 0:
        return True
    return torch.equal(torch.sort(actual.long().cpu()).values, torch.sort(expected.long().cpu()).values)


def _is_non_decreasing(values: Tensor) -> bool:
    if int(values.numel()) <= 1:
        return True
    return bool((values[1:] >= values[:-1]).all().item())


def _edge_ids(graph: dict[str, Any]) -> Tensor:
    if graph.get("edge_ids") is None:
        return torch.arange(_num_edges(graph), dtype=torch.long)
    return torch.as_tensor(graph["edge_ids"], dtype=torch.long).cpu().contiguous()


def _rank_local_edges(rank: dict[str, Any]) -> Tensor:
    if rank.get("local_edge_ids") is None:
        return torch.empty(0, dtype=torch.long)
    return torch.as_tensor(rank["local_edge_ids"], dtype=torch.long).cpu().contiguous()


def _num_edges(graph: dict[str, Any]) -> int:
    if graph.get("src") is None:
        return 0
    return int(torch.as_tensor(graph["src"]).numel())


def _td_len(td: Any) -> int:
    if not isinstance(td, dict) or td.get("ptr") is None:
        return 0
    return max(int(torch.as_tensor(td["ptr"]).numel()) - 1, 0)


def _td_item(td: Any, index: int) -> Tensor:
    if not isinstance(td, dict) or td.get("ptr") is None or td.get("data") is None:
        return torch.empty(0, dtype=torch.long)
    ptr = torch.as_tensor(td["ptr"], dtype=torch.long).cpu()
    data = torch.as_tensor(td["data"]).cpu()
    begin, end = int(ptr[index]), int(ptr[index + 1])
    return data[begin:end]


def _as_long_2d(value: Any) -> Tensor:
    if value is None:
        return torch.zeros((0, 2), dtype=torch.long)
    out = torch.as_tensor(value, dtype=torch.long).cpu().contiguous()
    if out.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    if out.dim() != 2 or int(out.size(1)) != 2:
        raise ValueError(f"expected [N, 2] pointer tensor, got shape {tuple(out.shape)}")
    return out


def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _print_text_report(root: Path, report: CheckReport) -> None:
    print(f"artifact_root: {root}")
    print(f"status: {'OK' if report.ok else 'FAIL'}")
    for key in sorted(report.stats):
        print(f"{key}: {report.stats[key]}")
    if report.time_ranges:
        print("time_ranges:")
        for key in sorted(report.time_ranges):
            item = report.time_ranges[key]
            print(
                f"  - {key}: windows={int(item['windows'])} events={int(item['events'])} "
                f"ts=[{float(item['min_ts']):.6g}, {float(item['max_ts']):.6g}] "
                f"empty_rank_batches={int(item['empty_rank_batches'])}"
            )
    if report.warnings:
        print("warnings:")
        for item in report.warnings:
            print(f"  - {item}")
    if report.errors:
        print("errors:")
        for item in report.errors:
            print(f"  - {item}")


def _json_report(report: CheckReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "stats": dict(sorted(report.stats.items())),
        "time_ranges": dict(sorted(report.time_ranges.items())),
        "rank_time_ranges": report.rank_time_ranges,
        "warnings": report.warnings,
        "errors": report.errors,
    }


def _parse_checks(values: Iterable[str]) -> tuple[bool, bool]:
    selected = set(values)
    if "all" in selected:
        return True, True
    return "ctdg" in selected, "dtdg" in selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether rank-local preprocessed batches reconstruct the global graph windows."
    )
    parser.add_argument("--artifact-root", required=True, type=Path, help="Directory containing graph.pt/rank_*.pt/partition_data_*.pt")
    parser.add_argument(
        "--check",
        nargs="+",
        default=["all"],
        choices=["all", "ctdg", "dtdg"],
        help="Which artifact contracts to check.",
    )
    parser.add_argument(
        "--strict-concat-order",
        action="store_true",
        help="Also require rank0+rank1+... concatenation to match global order without sorting.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Splits to scan for CTDG epoch-style checks. Use --splits train for one training epoch.",
    )
    parser.add_argument(
        "--dump-rank-time-ranges",
        type=Path,
        default=None,
        help="Optional JSONL output with one row per checked CTDG split/batch/rank time range.",
    )
    parser.add_argument("--max-examples", type=int, default=8, help="Maximum missing/extra ids shown per error.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    check_ctdg, check_dtdg = _parse_checks(args.check)
    report = check_artifact_root(
        args.artifact_root,
        check_ctdg=check_ctdg,
        check_dtdg=check_dtdg,
        strict_concat_order=bool(args.strict_concat_order),
        max_examples=max(1, int(args.max_examples)),
        splits=tuple(args.splits),
    )
    if args.dump_rank_time_ranges is not None:
        args.dump_rank_time_ranges.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_rank_time_ranges.open("w", encoding="utf-8") as f:
            for row in report.rank_time_ranges:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report.stats["rank_time_range_rows_written"] = len(report.rank_time_ranges)
    if args.json:
        print(json.dumps(_json_report(report), indent=2, sort_keys=True))
    else:
        _print_text_report(args.artifact_root, report)
        if args.dump_rank_time_ranges is not None:
            print(f"rank_time_ranges_jsonl: {args.dump_rank_time_ranges}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
