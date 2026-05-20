from __future__ import annotations

import torch

from atc_starrygl_lib.memory.async_updater import _build_mailbox_messages


def test_build_mailbox_messages_updates_both_edge_endpoints() -> None:
    nid = torch.tensor([1, 2], dtype=torch.long)
    memory = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    edge_feat = torch.tensor([[5.0]])

    msg = _build_mailbox_messages(
        nid,
        memory,
        src=torch.tensor([1], dtype=torch.long),
        dst=torch.tensor([2], dtype=torch.long),
        edge_feat=edge_feat,
    )

    assert msg.tolist() == [[10.0, 11.0, 20.0, 21.0, 5.0], [20.0, 21.0, 10.0, 11.0, 5.0]]
