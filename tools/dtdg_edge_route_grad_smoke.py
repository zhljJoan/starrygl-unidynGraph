from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.dtdg.runtime import STGraphLoader
from atc_starrygl_lib.dtdg.train_loop import prepare_edge_prediction_embeddings
from atc_starrygl_lib.preprocess.partition_data import build_all_partition_data_artifacts


def main() -> None:
    rank, world_size, local_rank = _init_dist()
    if world_size != 2:
        raise RuntimeError("dtdg_edge_route_grad_smoke expects exactly 2 ranks")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    parts = _partition_data()
    loader = STGraphLoader(partition_data=parts[rank], device=device, rank=rank, world_size=world_size)
    snapshot = loader.fetch_snapshot(0)
    graph = snapshot.graph
    dst_emb = torch.nn.Parameter(torch.full((int(graph.num_dst_nodes()), 2), float(rank + 1), device=device))
    batch = Batch(
        split="train",
        roots=snapshot.src_ids,
        graph=graph,
        eids=snapshot.edge_ids,
        pos_src=snapshot.edge_src.long(),
        pos_dst=snapshot.edge_dst.long(),
    )
    expanded = prepare_edge_prediction_embeddings(dst_emb, batch)
    if rank == 0:
        loss = expanded.index_select(0, batch.pos_src).sum()
    else:
        loss = expanded.sum() * 0.0
    loss.backward()

    grad_sum = torch.tensor([0.0 if dst_emb.grad is None else float(dst_emb.grad.abs().sum().item())], device=device)
    gathered = [torch.zeros_like(grad_sum) for _ in range(world_size)]
    dist.all_gather(gathered, grad_sum)
    if rank == 0:
        out = {"rank_grad_sums": [float(item.item()) for item in gathered]}
        print(json.dumps(out, sort_keys=True), flush=True)
        if out["rank_grad_sums"][1] <= 0.0:
            raise RuntimeError("remote endpoint embedding did not receive gradient")
    dist.barrier()
    dist.destroy_process_group()


def _partition_data() -> list[dict]:
    return build_all_partition_data_artifacts(
        rank_artifacts=[
            {
                "rank": 0,
                "local_node_ids": torch.tensor([0, 1]),
                "local_edge_ids": torch.tensor([0]),
            },
            {
                "rank": 1,
                "local_node_ids": torch.tensor([2]),
                "local_edge_ids": torch.tensor([1]),
            },
        ],
        dist_plan={
            "node_to_chunk": torch.tensor([0, 0, 1]),
            "node_master": torch.tensor([0, 0, 1]),
        },
        src=torch.tensor([2, 1]),
        dst=torch.tensor([0, 2]),
        time_ptr_2=torch.tensor([[0, 2]]),
        node_feat=torch.eye(3, dtype=torch.float32),
    )


def _init_dist() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


if __name__ == "__main__":
    main()
