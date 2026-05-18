import os

import torch
import torch.distributed as dist

from starry_unigraph.backends.chunk.data.comm import CommPipeline
from starry_unigraph.backends.chunk.data.dist_index import encode_dist_index
from starry_unigraph.backends.chunk.data.plans import FetchPlan


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")

    rows = (torch.arange(8, dtype=torch.float32, device=device).view(4, 2) + rank * 100)
    mem = (torch.arange(12, dtype=torch.float32, device=device).view(4, 3) + rank * 1000)
    remote_owner = (rank + 1) % world
    local_read = encode_dist_index(
        torch.tensor([0], device=device),
        torch.tensor([rank], device=device),
        cached=True,
    )
    remote_read = encode_dist_index(
        torch.tensor([1], device=device),
        torch.tensor([remote_owner], device=device),
    )
    plan = FetchPlan(
        block_id=0,
        placement_version=0,
        feature_node_ids=torch.empty(0, dtype=torch.long, device=device),
        feature_owners=torch.tensor([remote_owner], dtype=torch.long, device=device),
        remote_read_index=remote_read,
        local_read_index=local_read,
        memory_read_index=remote_read,
    )

    pipeline = CommPipeline(device=device)
    handle = pipeline.submit_fetch(plan, feature_rows=rows, memory_rows=mem)
    import asyncio

    result = asyncio.run(pipeline.await_handle(handle))

    expected_feat = torch.tensor([[2.0, 3.0]], device=device) + remote_owner * 100
    expected_mem = torch.tensor([[3.0, 4.0, 5.0]], device=device) + remote_owner * 1000
    assert result is not None
    assert torch.equal(result.local_features, rows[:1])
    assert torch.equal(result.remote_features, expected_feat), (rank, result.remote_features, expected_feat)
    assert torch.equal(result.local_memory, mem[:1])
    assert torch.equal(result.remote_memory, expected_mem), (rank, result.remote_memory, expected_mem)
    if rank == 0:
        print("distributed fetch all_to_all ok", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
