from __future__ import annotations

import torch

from atc_starrygl_lib.comm.layouts import MailboxReadLayout, MemoryReadLayout
from atc_starrygl_lib.ctdg.runtime.backend import _build_mailbox_runtime, _build_memory_runtime
from atc_starrygl_lib.core.types import RuntimeContext


def _ctx() -> RuntimeContext:
    return RuntimeContext(config={}, artifact_root=".", rank=0, world_size=1, device="cpu")


def test_build_memory_runtime_direct_node_id_io_reads_and_writes_by_node_id() -> None:
    runtime = _build_memory_runtime(
        rank_artifact={
            "local_node_ids": torch.tensor([10, 20], dtype=torch.long),
            "read_dist_index": torch.tensor([99, 98, 97], dtype=torch.long),
        },
        dist={
            "num_nodes": 3,
            "master_dist_index": torch.tensor([77, 76, 75], dtype=torch.long),
        },
        ctx=_ctx(),
        runtime_cfg={
            "memory_dim": 2,
            "single_machine_direct_state_io": True,
        },
    )
    assert runtime.direct_node_id_io is True
    layout = MemoryReadLayout(
        read_index=torch.tensor([2, 0], dtype=torch.long),
        read_ptr=torch.tensor([0, 2], dtype=torch.long),
        compute_to_memory=torch.tensor([1, 0], dtype=torch.long),
    )
    runtime.store.memory[0] = torch.tensor([1.0, 2.0])
    runtime.store.memory[2] = torch.tensor([5.0, 6.0])
    runtime.store.ts[0] = 10.0
    runtime.store.ts[2] = 20.0
    memory, ts = runtime.wait_read(runtime.submit_read(layout), layout)
    assert memory.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert ts.tolist() == [10.0, 20.0]

    write_layout = runtime.build_write_layout(torch.tensor([2, 0], dtype=torch.long))
    runtime.write(
        write_layout,
        torch.tensor([[9.0, 9.5], [7.0, 7.5]], dtype=torch.float32),
        torch.tensor([30.0, 40.0], dtype=torch.float32),
    ).wait_apply()
    assert runtime.store.memory[2].tolist() == [9.0, 9.5]
    assert runtime.store.memory[0].tolist() == [7.0, 7.5]
    assert runtime.store.ts[2].item() == 30.0
    assert runtime.store.ts[0].item() == 40.0


def test_build_mailbox_runtime_direct_node_id_io_reads_and_writes_by_node_id() -> None:
    runtime = _build_mailbox_runtime(
        rank_artifact={
            "local_node_ids": torch.tensor([10, 20], dtype=torch.long),
            "read_dist_index": torch.tensor([99, 98, 97], dtype=torch.long),
        },
        dist={
            "num_nodes": 3,
            "master_dist_index": torch.tensor([77, 76, 75], dtype=torch.long),
        },
        ctx=_ctx(),
        runtime_cfg={
            "mailbox_size": 1,
            "mailbox_msg_dim": 2,
            "single_machine_direct_state_io": True,
        },
    )
    assert runtime.direct_node_id_io is True
    runtime.store.mailbox[1, 0] = torch.tensor([3.0, 4.0])
    runtime.store.mailbox_ts[1, 0] = 12.0
    runtime.store.mailbox[2, 0] = torch.tensor([5.0, 6.0])
    runtime.store.mailbox_ts[2, 0] = 18.0
    layout = MailboxReadLayout(
        read_index=torch.tensor([2, 1], dtype=torch.long),
        read_ptr=torch.tensor([0, 2], dtype=torch.long),
        compute_to_mailbox=torch.tensor([1, 0], dtype=torch.long),
    )
    mailbox, mailbox_ts = runtime.wait_read(runtime.submit_read(layout), layout)
    assert mailbox.tolist() == [[[3.0, 4.0]], [[5.0, 6.0]]]
    assert mailbox_ts.tolist() == [[12.0], [18.0]]

    write_layout = runtime.build_write_layout(torch.tensor([2, 1], dtype=torch.long))
    runtime.write(
        write_layout,
        torch.tensor([[9.0, 9.5], [7.0, 7.5]], dtype=torch.float32),
        torch.tensor([30.0, 40.0], dtype=torch.float32),
    ).wait_apply()
    assert runtime.store.mailbox[2, 0].tolist() == [9.0, 9.5]
    assert runtime.store.mailbox[1, 0].tolist() == [7.0, 7.5]
