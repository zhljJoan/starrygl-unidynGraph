import torch

from starry_unigraph.backends.chunk.data import propagation_route as route_mod
from starry_unigraph.backends.chunk.data.propagation_route import ChunkPropagationRoute


class _DoneWork:
    def wait(self):
        return True


def _copy_all_to_all_single(output, input, **_kwargs):
    output.copy_(input)
    return _DoneWork()


def test_chunk_propagation_route_append_recv_backward_scatter(monkeypatch):
    monkeypatch.setattr(route_mod.dist, "all_to_all_single", _copy_all_to_all_single)
    route = ChunkPropagationRoute(
        send_sizes=[2],
        recv_sizes=[2],
        send_index=torch.tensor([1, 2]),
        append_recv=True,
    )
    ctx = route_mod._PropagationContext(route)
    x = torch.arange(8, dtype=torch.float32).view(4, 2)

    ctx.forward_send(x)
    out = ctx.forward_recv()
    assert torch.equal(out, torch.cat([x, x[[1, 2]]], dim=0))

    ctx.backward_send(torch.ones_like(out))
    grad_x = ctx.backward_recv()
    expected = torch.ones_like(x)
    expected[1] += 1
    expected[2] += 1
    assert torch.equal(grad_x, expected)


def test_chunk_propagation_route_recv_only_reorders_rows_and_grad(monkeypatch):
    monkeypatch.setattr(route_mod.dist, "all_to_all_single", _copy_all_to_all_single)
    route = ChunkPropagationRoute(
        send_sizes=[2],
        recv_sizes=[2],
        send_index=torch.tensor([1, 2]),
        recv_src_rows=torch.tensor([1, 0]),
        append_recv=False,
    )
    ctx = route_mod._PropagationContext(route)
    x = torch.arange(8, dtype=torch.float32).view(4, 2)

    ctx.forward_send(x)
    out = ctx.forward_recv()
    assert torch.equal(out, x[[2, 1]])

    grad_out = torch.tensor([[3.0, 3.0], [5.0, 5.0]])
    ctx.backward_send(grad_out)
    grad_x = ctx.backward_recv()
    expected = torch.zeros_like(x)
    expected[1] = torch.tensor([5.0, 5.0])
    expected[2] = torch.tensor([3.0, 3.0])
    assert torch.equal(grad_x, expected)
