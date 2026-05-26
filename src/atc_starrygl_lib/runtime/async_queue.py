from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Deque, Generic, TypeVar

T = TypeVar("T")


class AsyncWorkQueue(Generic[T]):
    """Reusable FIFO async queue backed by a thread pool."""

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Deque[Future[T]] = deque()

    def __len__(self) -> int:
        return len(self._futures)

    def submit(self, fn: Callable[..., T], /, *args, **kwargs) -> None:
        self._futures.append(self._executor.submit(fn, *args, **kwargs))

    def pop_result(self) -> T:
        future = self._futures.popleft()
        return future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

