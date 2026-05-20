import torch

from atc_starrygl_lib.core.types import Batch
from atc_starrygl_lib.models.shared.heads import NodeClassifyHead, NodeRegressHead


def test_node_heads_use_embedding_rows_not_global_node_ids() -> None:
    embeddings = torch.randn(2, 4)
    batch = Batch(
        split="train",
        roots=torch.tensor([10, 20], dtype=torch.long),
        timestamps=torch.tensor([1.0, 2.0]),
        node_ids=torch.tensor([10, 20], dtype=torch.long),
        labels=torch.tensor([1, 0], dtype=torch.long),
    )

    classify = NodeClassifyHead(dim=4, num_classes=3)
    regress = NodeRegressHead(dim=4)

    assert classify(embeddings, batch).logits.shape == (2, 3)
    assert regress(embeddings, batch).pred.shape == (2, 1)
