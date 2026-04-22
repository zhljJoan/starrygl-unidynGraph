from typing import List

from torch import Tensor, dist
from dataclasses import dataclass

import torch

from starry_unigraph.runtime.route.route import Route
@dataclass  
class SpatialDeps:
    """chunk声明: 我的GNN计算需要这些远程节点的embedding"""
    #存储chunk之间的通信依赖，只考虑同时段的cluster间的通信
    send_sizes: List[Tensor]  # 每个part需要发送的embedding数量
    recv_sizes: List[Tensor]  # 每个part需要接收的embedding数量
    send_index: Tensor | None = None  # 可选的发送索引，形状 [sum(send_sizes)]，指明每个embedding对应的全局节点ID
    send_index_ptr: Tensor|None = None  # 可选的发送索引指针，形状 [num_parts + 1]，指明send_index中每个part的起止位置
    
    #根据chunk的大小建立实际的route，或者直接传入route,
    def to_route(self, chunks, in_chunks = None, num_clusters = None) -> Route:
        return Route(
            send_index=self.send_index[in_chunks] if self.send_index is not None else None,
            send_sizes=[torch.sum(self.send_sizes[i*num_clusters + chunks]).item() for i in range(dist.get_world_size())],
            recv_sizes=[torch.sum(self.recv_sizes[i*num_clusters + chunks]).item() for i in range(dist.get_world_size())],
        )

    

class RouteData:
    def __init__(self,
        send_sizes: List[List[int]],
        recv_sizes: List[List[int]],
        send_index_ind: Tensor | None,
        send_index_ptr: List[int] | None,
    ):
        assert len(send_sizes) == len(recv_sizes), "send_sizes and recv_sizes must have the same length"
        for i, (ss, rs) in enumerate(zip(send_sizes, recv_sizes)):
            assert len(ss) == len(rs), f"send_sizes[{i}] and recv_sizes[{i}] must have the same length"
        
        if send_index_ind is not None:
            assert send_index_ptr[-1] == send_index_ind.numel(), "send_index_ptr[-1] must equal send_index_ind.numel()"
        
        self.send_index_ind = send_index_ind
        self.send_index_ptr = send_index_ptr
        self.send_sizes = send_sizes
        self.recv_sizes = recv_sizes
    
    def __getstate__(self):
        return {
            "send_index_ind": self.send_index_ind,
            "send_index_ptr": self.send_index_ptr,
            "send_sizes": self.send_sizes,
            "recv_sizes": self.recv_sizes,
        }
    
    def __setstate__(self, state):
        self.send_index_ind = state["send_index_ind"]
        self.send_index_ptr = state["send_index_ptr"]
        self.send_sizes = state["send_sizes"]
        self.recv_sizes = state["recv_sizes"]
    
    def __repr__(self) -> str:
        n = len(self)
        p = len(self.send_sizes[0]) if self.send_sizes else -1
        return f"{type(self).__name__}(len={n}, parts={p})"
    
    def __len__(self):
        return len(self.send_sizes)
    
    def __getitem__(self, k: int | slice):
        if not isinstance(k, slice):
            k = slice(k, k + 1)

        if k.step is not None and k.step != 1:
            raise ValueError("Only step size of 1 is supported")
        s = 0 if k.start is None else k.start
        t = len(self) if k.stop is None else k.stop

        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
            send_index_ptr = self.send_index_ptr
        else:
            a, b = self.send_index_ptr[s], self.send_index_ptr[t]
            send_index_ind = self.send_index_ind[a:b]
            send_index_ptr = [x - a for x in self.send_index_ptr[s:t+1]]

        send_sizes = self.send_sizes[s:t]
        recv_sizes = self.recv_sizes[s:t]
        
        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )
    
    def pin_memory(self, device = None):
        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
        else:
            send_index_ind = self.send_index_ind.pin_memory(device=device)

        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=self.send_index_ptr,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
        )
    
    def to(self, device = None, dtype = None, non_blocking = False, copy = False):
        if self.send_index_ind is None:
            send_index_ind = self.send_index_ind
        else:
            send_index_ind = self.send_index_ind.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        return type(self)(
            send_index_ind=send_index_ind,
            send_index_ptr=self.send_index_ptr,
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
        )
    
    def item(self):
        routes = self.to_routes()
        assert len(routes) == 1, f"Expected 1 route, got {len(routes)}"
        return routes[0]
    
    def to_routes(self, group = None):
        routes: List[Route] = []
        for i in range(len(self)):
            if self.send_index_ind is None:
                send_index = None
            else:
                a, b = self.send_index_ptr[i], self.send_index_ptr[i+1]
                send_index = self.send_index_ind[a:b]

            r = Route(
                send_index=send_index,
                send_sizes=self.send_sizes[i],
                recv_sizes=self.recv_sizes[i],
                group=group,
            )
            routes.append(r)
        return routes
    
    @classmethod
    def from_routes(cls, routes: List[Route]):
        send_index_ind = []
        send_index_ptr = [0]
        send_sizes = []
        recv_sizes = []
        for r in routes:
            if r.send_index is None:
                assert not send_index_ind, "send_index_ind should be None if all routes have None send_index"
            else:
                send_index_ind.append(r.send_index)
                send_index_ptr.append(send_index_ptr[-1] + r.send_index.numel())

            send_sizes.append(r.send_sizes)
            recv_sizes.append(r.recv_sizes)
        
        if send_index_ind:
            assert len(send_index_ind) == len(routes), f"send_index_ind should have the same length as routes if any route has non-None send_index"
            send_index_ind = torch.cat(send_index_ind, dim=0)
        else:
            send_index_ind = None
            send_index_ptr = None

        return cls(
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )


@dataclass
class StateDeps:
    """chunk声明: 我的计算涉及的状态读写依赖"""
    
    read_boundary: Tensor       # [N_read] 可能需要读memory的远程节点, 要考虑从上一个直接获取还是从本地缓存获取的吗？
    read_from_cluster: Tensor   # [N_read] 归属cluster
    
    # 写依赖: 本chunk中哪些节点的memory更新后需要通知远程
    # = "被其他cluster的chunk当作采样邻居的本地节点"
    write_notify: Tensor        # [N_write] 需要向外广播更新的本地节点
    write_to_clusters: List[Tensor]  # 每个节点需要通知哪些cluster
    
    def stale_filter(self, mask):#过滤可以不是非得更新的特征，其他特征少收点
        torch.dist