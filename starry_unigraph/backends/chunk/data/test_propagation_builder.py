import torch

from starry_unigraph.backends.chunk.prepare.propagation_builder import build_propagation_routes


def test_propagation_routes_empty_single_rank():
    routes = build_propagation_routes(
        edge_src=torch.tensor([0, 1]),
        edge_dst=torch.tensor([1, 2]),
        time_ptr=torch.tensor([0, 2]),
        node_owner=torch.zeros(3, dtype=torch.long),
        num_parts=1,
        num_layers=2,
    )
    assert len(routes) == 1
    assert len(routes[0]) == 1
    assert len(routes[0][0]) == 2
    route = routes[0][0][0]
    assert route.send_sizes == [0]
    assert route.recv_sizes == [0]
    assert route.send_index is None


def test_propagation_routes_two_part_boundary():
    node_owner = torch.tensor([0, 0, 1, 1])
    routes = build_propagation_routes(
        edge_src=torch.tensor([0, 2, 1]),
        edge_dst=torch.tensor([2, 1, 3]),
        time_ptr=torch.tensor([0, 3]),
        node_owner=node_owner,
        num_parts=2,
        num_layers=1,
    )
    r0 = routes[0][0][0]
    r1 = routes[1][0][0]

    assert r0.send_sizes == [0, 2]
    assert r0.recv_sizes == [0, 1]
    assert torch.equal(r0.send_index, torch.tensor([0, 1]))

    assert r1.send_sizes == [1, 0]
    assert r1.recv_sizes == [2, 0]
    assert torch.equal(r1.send_index, torch.tensor([0]))
