from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .route import Route


@dataclass
class RouteData:
    send_sizes: list[list[int]]
    recv_sizes: list[list[int]]
    send_index_ind: Tensor | None
    send_index_ptr: list[int] | None

    def __post_init__(self) -> None:
        if len(self.send_sizes) != len(self.recv_sizes):
            raise ValueError("send_sizes and recv_sizes must have the same length")
        for send_row, recv_row in zip(self.send_sizes, self.recv_sizes):
            if len(send_row) != len(recv_row):
                raise ValueError("send_sizes[i] and recv_sizes[i] must have the same length")
        if self.send_index_ind is not None and self.send_index_ptr is not None and self.send_index_ptr[-1] != int(self.send_index_ind.numel()):
            raise ValueError("send_index_ptr[-1] must equal send_index_ind.numel()")

    def __len__(self) -> int:
        return len(self.send_sizes)

    def __getitem__(self, index: int | slice) -> RouteData:
        if not isinstance(index, slice):
            index = slice(index, index + 1)
        if index.step not in (None, 1):
            raise ValueError("Only step size of 1 is supported")
        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        if self.send_index_ind is None or self.send_index_ptr is None:
            send_index_ind = self.send_index_ind
            send_index_ptr = self.send_index_ptr
        else:
            offset_start, offset_stop = self.send_index_ptr[start], self.send_index_ptr[stop]
            send_index_ind = self.send_index_ind[offset_start:offset_stop]
            send_index_ptr = [value - offset_start for value in self.send_index_ptr[start : stop + 1]]
        return type(self)(
            send_sizes=self.send_sizes[start:stop],
            recv_sizes=self.recv_sizes[start:stop],
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr,
        )

    def item(self, group=None) -> Route:
        routes = self.to_routes(group=group)
        if len(routes) != 1:
            raise ValueError(f"Expected 1 route, got {len(routes)}")
        return routes[0]

    def to_routes(self, group=None) -> list[Route]:
        routes: list[Route] = []
        for index in range(len(self)):
            if self.send_index_ind is None or self.send_index_ptr is None:
                send_index = None
            else:
                start, stop = self.send_index_ptr[index], self.send_index_ptr[index + 1]
                send_index = self.send_index_ind[start:stop]
            routes.append(Route(send_index=send_index, send_sizes=self.send_sizes[index], recv_sizes=self.recv_sizes[index], group=group))
        return routes

    def pin_memory(self, device: str | None = None) -> RouteData:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.pin_memory(device=device),
            send_index_ptr=self.send_index_ptr,
        )

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> RouteData:
        return type(self)(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
            send_index_ptr=self.send_index_ptr,
        )

    @classmethod
    def from_routes(cls, routes: list[Route]) -> RouteData:
        send_index_parts: list[Tensor] = []
        send_index_ptr = [0]
        send_sizes: list[list[int]] = []
        recv_sizes: list[list[int]] = []
        for route in routes:
            if route.send_index is not None:
                send_index_parts.append(route.send_index.long())
                send_index_ptr.append(send_index_ptr[-1] + int(route.send_index.numel()))
            send_sizes.append([int(size) for size in route.send_sizes])
            recv_sizes.append([int(size) for size in route.recv_sizes])
        send_index_ind = torch.cat(send_index_parts, dim=0) if send_index_parts else None
        return cls(
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
            send_index_ind=send_index_ind,
            send_index_ptr=send_index_ptr if send_index_ind is not None else None,
        )


@dataclass
class TensorData:
    ptr: list[int]
    data: Tensor

    def __post_init__(self) -> None:
        if not self.ptr:
            raise ValueError("ptr must not be empty")
        if self.ptr[-1] != int(self.data.size(0)):
            raise ValueError(f"ptr[-1] != data.size(0): {self.ptr[-1]} != {self.data.size(0)}")

    def __len__(self) -> int:
        return len(self.ptr) - 1

    def __getitem__(self, index: int | slice) -> TensorData:
        if not isinstance(index, slice):
            index = slice(index, index + 1)
        if index.step not in (None, 1):
            raise ValueError("Only step size of 1 is supported")
        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        offset_start, offset_stop = self.ptr[start], self.ptr[stop]
        ptr = [value - offset_start for value in self.ptr[start : stop + 1]]
        return type(self)(ptr=ptr, data=self.data[offset_start:offset_stop])

    def item(self) -> Tensor:
        items = self.to_tensors()
        if len(items) != 1:
            raise ValueError(f"Expected 1 tensor, got {len(items)}")
        return items[0]

    def to_tensors(self) -> list[Tensor]:
        items: list[Tensor] = []
        for index in range(len(self)):
            start, stop = self.ptr[index], self.ptr[index + 1]
            items.append(self.data[start:stop])
        return items

    def pin_memory(self, device: str | None = None) -> TensorData:
        return type(self)(ptr=self.ptr, data=self.data.pin_memory(device=device))

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> TensorData:
        return type(self)(
            ptr=self.ptr,
            data=self.data.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy),
        )

    @classmethod
    def from_tensors(cls, tensors: list[Tensor]) -> TensorData:
        if not tensors:
            raise ValueError("tensors must not be empty")
        ptr = [0]
        for tensor in tensors:
            ptr.append(ptr[-1] + int(tensor.size(0)))
        return cls(ptr=ptr, data=torch.cat(tensors, dim=0))
