from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel

from atc_starrygl_lib.core.types import ArtifactBundle, RuntimeContext
from atc_starrygl_lib.ctdg.runtime.backend import MemShareTemporalSamplingBackend
from atc_starrygl_lib.ctdg.train_loop import CTDGMemoryCommitHook
from atc_starrygl_lib.memory import AsyncMemoryCommitter, RuntimeAsyncMemoryUpdater
from atc_starrygl_lib.models.ctdg import GeneralModel
from atc_starrygl_lib.models.shared import EdgePredictHead
from atc_starrygl_lib.tasks import EdgePredictionTask


class SmokeEncoder(torch.nn.Module):
    def __init__(self, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(int(num_nodes), int(dim))

    def forward(self, batch: Any) -> torch.Tensor:
        node_ids = _batch_node_ids(batch)
        return self.embedding(node_ids.to(self.embedding.weight.device))


class SmokeEdgeModel(torch.nn.Module):
    def __init__(self, num_nodes: int, dim: int) -> None:
        super().__init__()
        self.encoder = SmokeEncoder(num_nodes=num_nodes, dim=dim)
        self.head = EdgePredictHead(dim=dim)

    def forward(self, batch: Any) -> Any:
        return self.head(self.encoder(batch), batch)


class GeneralEdgeModel(torch.nn.Module):
    def __init__(self, backbone: GeneralModel) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: Any) -> Any:
        return self.backbone.forward_batch(batch)


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _init_dist(args.device)
    device = _device(args.device, local_rank)
    artifact_root = Path(args.artifact_root)

    if args.prepare and rank == 0:
        ctx = _ctx(args, artifact_root=artifact_root, rank=rank, world_size=world_size, device=str(device))
        MemShareTemporalSamplingBackend().prepare(ctx)
    if dist.is_initialized():
        dist.barrier()

    graph = torch.load(artifact_root / "graph.pt", map_location="cpu", weights_only=False)
    backend = MemShareTemporalSamplingBackend()
    backend._prepared_by = "new_pipeline"
    ctx = _ctx(args, artifact_root=artifact_root, rank=rank, world_size=world_size, device=str(device))
    backend.build_runtime(ctx, _bundle(artifact_root, world_size))

    model = _build_model(args=args, graph=graph, backend=backend).to(device)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=_ddp_device_ids(device))
    task = EdgePredictionTask()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    memory_commit = CTDGMemoryCommitHook(wait_apply=True) if args.model == "general" else None

    summary: dict[str, Any] = {"rank": rank, "world_size": world_size, "epochs": []}
    for epoch in range(int(args.epochs)):
        epoch_summary = _run_epoch(
            backend=backend,
            model=model,
            task=task,
            opt=opt,
            split="train",
            device=device,
            memory_commit=memory_commit,
        )
        summary["epochs"].append({"epoch": epoch, **epoch_summary})
        if rank == 0:
            print(json.dumps(summary["epochs"][-1], sort_keys=True), flush=True)

    if rank == 0 and args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_epoch(
    *,
    backend: MemShareTemporalSamplingBackend,
    model: torch.nn.Module,
    task: EdgePredictionTask,
    opt: torch.optim.Optimizer,
    split: str,
    device: torch.device,
    memory_commit: Any = None,
) -> dict[str, float]:
    model.train()
    t0 = time.perf_counter()
    total_loss = 0.0
    total_edges = 0
    total_batches = 0
    metric_sum: dict[str, float] = {}
    join_ctx = Join([model]) if dist.is_initialized() else nullcontext()
    with join_ctx:
        for batch in backend.iter_batches(split):
            batch = _move_batch(batch, device)
            if batch.pos_src is None or batch.pos_src.numel() == 0:
                continue
            output = model(batch)
            loss = task.compute_loss(output, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if memory_commit is not None:
                memory_commit(model, batch)

            edges = int(batch.pos_src.numel()) if batch.pos_src is not None else 0
            total_loss += float(loss.detach().item())
            total_edges += edges
            total_batches += 1
            if output.pos_score.numel() > 0 and output.neg_score.numel() > 0:
                for key, value in task.compute_metrics(output, batch).items():
                    metric_sum[key] = metric_sum.get(key, 0.0) + float(value)

    elapsed = max(time.perf_counter() - t0, 1e-12)
    stats = torch.tensor([total_loss, total_edges, total_batches, elapsed], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, edge_sum, batch_sum, elapsed_sum = stats.tolist()
    out = {
        "loss": loss_sum / max(batch_sum, 1.0),
        "edges": edge_sum,
        "batches": batch_sum,
        "seconds": elapsed_sum / max(float(_world_size()), 1.0),
        "edges_per_sec": edge_sum / max(elapsed_sum / max(float(_world_size()), 1.0), 1e-12),
    }
    for key, value in metric_sum.items():
        metric = torch.tensor([value, float(total_batches)], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        out[key] = float(metric[0].item() / max(metric[1].item(), 1.0))
    return out


def _batch_node_ids(batch: Any) -> torch.Tensor:
    graph = batch.graph
    block = _first_block(graph)
    if block is not None and hasattr(block, "srcdata") and "ID" in block.srcdata:
        return block.srcdata["ID"].long()
    return batch.roots.long()


def _first_block(graph: Any) -> Any:
    if isinstance(graph, (list, tuple)):
        if not graph:
            return None
        first = graph[0]
        if isinstance(first, (list, tuple)):
            return first[0] if first else None
        return first
    return graph


def _move_batch(batch: Any, device: torch.device) -> Any:
    for name in ("roots", "timestamps", "eids", "src", "dst", "ts", "pos_src", "pos_dst", "neg_src", "neg_dst", "labels"):
        value = getattr(batch, name, None)
        if isinstance(value, torch.Tensor):
            setattr(batch, name, value.to(device))
    batch.graph = _move_graph(batch.graph, device)
    return batch


def _move_graph(graph: Any, device: torch.device) -> Any:
    if graph is None:
        return None
    if isinstance(graph, list):
        return [_move_graph(item, device) for item in graph]
    if isinstance(graph, tuple):
        return tuple(_move_graph(item, device) for item in graph)
    if hasattr(graph, "to"):
        return graph.to(device)
    return graph


def _ctx(args: argparse.Namespace, *, artifact_root: Path, rank: int, world_size: int, device: str) -> RuntimeContext:
    global_batch_size = int(args.batch_size) * int(world_size)
    memory_dim = int(args.hidden_dim)
    mailbox_msg_dim = memory_dim * 2 + int(args.edge_dim)
    return RuntimeContext(
        config={
            "graph": {"mode": "ctdg", "source": args.data},
            "task": {"name": "edge_prediction", "batch_size": args.batch_size},
            "preprocess": {
                "use_new_pipeline": True,
                "partition_algorithm": args.partition_algorithm,
                "chunks_per_rank": args.chunks_per_rank,
                "batch_size": global_batch_size,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            },
            "runtime": {
                "build_sampler": bool(args.build_sampler),
                "build_feature_runtime": False,
                "build_memory_runtime": args.model == "general",
                "build_mailbox_runtime": args.model == "general",
                "memory_dim": memory_dim,
                "mailbox_size": int(args.mailbox_size),
                "mailbox_msg_dim": mailbox_msg_dim,
                "negative_ratio": args.negative_ratio,
                "fanouts": args.fanouts,
                "num_layers": len(args.fanouts),
                "sampler_workers": args.workers,
                "graph_name": "wiki",
                "prefetch_batches": True,
            },
        },
        artifact_root=artifact_root,
        rank=rank,
        world_size=world_size,
        device=device,
    )


def _build_model(args: argparse.Namespace, graph: dict[str, Any], backend: MemShareTemporalSamplingBackend) -> torch.nn.Module:
    if args.model == "smoke":
        return SmokeEdgeModel(num_nodes=int(graph["num_nodes"]), dim=int(args.hidden_dim))
    runtime = backend._runtime
    if runtime is None or runtime.memory_runtime is None:
        raise RuntimeError("general model requires runtime memory construction")
    committer = AsyncMemoryCommitter(runtime.memory_runtime, runtime.mailbox_runtime)
    runtime_updater = RuntimeAsyncMemoryUpdater(base_updater=torch.nn.Identity(), committer=committer)
    config = {
        "sample": {
            "layer": len(args.fanouts),
            "neighbor": list(args.fanouts),
            "strategy": "recent",
            "prop_time": False,
            "history": 1,
            "duration": 0,
            "num_thread": int(args.workers),
        },
        "memory": {
            "type": "node",
            "dim_time": int(args.dim_time),
            "deliver_to": "self",
            "mail_combine": "last",
            "memory_update": "gru",
            "historical_fix": False,
            "async": True,
            "mailbox_size": int(args.mailbox_size),
            "combine_node_feature": True,
            "dim_out": int(args.hidden_dim),
        },
        "gnn": {
            "arch": "transformer_attention",
            "use_src_emb": False,
            "use_dst_emb": False,
            "layer": len(args.fanouts),
            "att_head": int(args.att_head),
            "dim_time": int(args.dim_time),
            "dim_out": int(args.hidden_dim),
        },
        "train": {
            "epoch": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "dropout": float(args.dropout),
            "att_dropout": float(args.att_dropout),
            "all_on_gpu": True,
        },
    }
    return GeneralEdgeModel(
        GeneralModel.from_config(
            dim_node=0,
            dim_edge=int(args.edge_dim),
            num_nodes=int(graph["num_nodes"]),
            config=config,
            runtime_memory_updater=runtime_updater,
        )
    )


def _bundle(root: Path, world_size: int) -> ArtifactBundle:
    files = {"graph": root / "graph.pt", "dist": root / "dist.pt", "meta": root / "meta.json"}
    for rank in range(int(world_size)):
        files[f"rank_{rank:03d}"] = root / f"rank_{rank:03d}.pt"
        feature = root / f"feature_{rank:03d}.pt"
        if feature.exists():
            files[f"feature_{rank:03d}"] = feature
    return ArtifactBundle(root=root, graph_mode="ctdg", files=files)


def _init_dist(device_name: str) -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    backend = "nccl" if str(device_name).startswith("cuda") and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _device(name: str, local_rank: int) -> torch.device:
    if name == "cuda":
        return torch.device("cuda", int(local_rank))
    return torch.device(name)


def _ddp_device_ids(device: torch.device) -> list[int] | None:
    if device.type != "cuda":
        return None
    return [int(device.index or 0)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTDG WIKI edge-prediction smoke/throughput validation.")
    parser.add_argument("--data", required=True, help="WIKI edge dataset path accepted by build_dataset.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare", action="store_true", help="Run preprocessing on rank 0 before training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--fanouts", type=int, nargs="+", default=[10])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--build-sampler", action="store_true")
    parser.add_argument("--model", choices=("general", "smoke"), default="general")
    parser.add_argument("--dim-time", type=int, default=100)
    parser.add_argument("--edge-dim", type=int, default=0)
    parser.add_argument("--mailbox-size", type=int, default=1)
    parser.add_argument("--att-head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--att-dropout", type=float, default=0.2)
    parser.add_argument("--partition-algorithm", default="speed_partition")
    parser.add_argument("--chunks-per-rank", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    main()
