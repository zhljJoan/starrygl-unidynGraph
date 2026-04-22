"""CommEngine Protocol: 统一通信引擎接口。

提供 DTDG 和 CTDG 共享的通信原语：
  - 空间通信 (Spatial)：跨 rank 节点 embedding/memory 交换
  - 状态同步 (State)：增量推送与按需拉取
  - 缓存拉取 (Cache Fetch)：step_cache 预加载
  - 梯度同步 (Gradient)：all-reduce

所有通信在 comm_stream 上执行，与 compute_stream 天然重叠。
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor

from starry_unigraph.runtime.comm.spatial_plan import SpatialDeps


@runtime_checkable
class CommHandle(Protocol):
    """异步通信句柄，持有通信操作的引用，支持 wait()。"""

    def wait(self) -> None:
        """阻塞直到本次通信完成。"""
        ...

    def is_completed(self) -> bool:
        """返回本次通信是否已完成（非阻塞查询）。"""
        ...


@runtime_checkable
class CommEngine(Protocol):
    """统一通信引擎 Protocol。

    DTDG 和 CTDG 共用同一个 CommEngine，仅通信计划（plan）不同：
      - DTDG: SpatialDeps 静态预编译，all_to_all 模式
      - CTDG: SpatialDeps 按 miss 动态生成，isend/irecv 模式

    所有方法均在 comm_stream 上提交，返回 CommHandle，
    调用方在需要数据前调用 handle.wait() 同步。
    """

    # ---- 空间通信 (Spatial) ----------------------------------------

    def async_exchange(
        self,
        send_ids: Tensor,
        send_vals: Tensor,
        plan: SpatialDeps,
    ) -> CommHandle:
        """异步交换跨 rank 节点的 embedding / memory。

        Args:
            send_ids:  [N_send] 本 rank 需要发送的节点全局 ID
            send_vals: [N_send, d] 对应的 embedding/memory 向量
            plan:      SpatialDeps，描述 send/recv rank 列表及节点 ID

        Returns:
            CommHandle，wait() 后 recv 缓冲区填充完成

        Implementation notes:
            DTDG: plan 静态，使用 dist.all_to_all_single 高效批量交换
            CTDG: plan 动态，使用 isend/irecv per-peer，量化可选 (INT8)
        """
        ...

    # ---- 状态同步 (State) ------------------------------------------

    def async_push_delta(
        self,
        node_ids: Tensor,
        values: Tensor,
    ) -> CommHandle:
        """异步向 owner rank 推送节点状态增量。

        Args:
            node_ids: [N_write] 需要推送更新的节点全局 ID
            values:   [N_write, d] 增量值（embedding 更新 / memory delta）

        Returns:
            CommHandle

        Notes:
            - 调用方需在下一 chunk 的 prepare_data 之前调用 handle.wait()
            - DTDG: 推 embedding 更新（backward 后）
            - CTDG: 推 memory 增量（post_compute 后）
        """
        ...

    def try_recv_delta(self) -> Optional[Tuple[Tensor, Tensor]]:
        """非阻塞查询是否有来自其他 rank 的增量到达。

        Returns:
            (node_ids, delta_values) 若有到达，否则 None

        Notes:
            - StateManager.update() 调用此方法合并远程增量
            - 返回 None 表示当前无待处理增量，调用方继续
        """
        ...

    # ---- 缓存拉取 (Cache Fetch) ------------------------------------

    def async_fetch(
        self,
        node_ids: Tensor,
        owners: Tensor,
    ) -> CommHandle:
        """异步从 owner rank 拉取节点的 embedding/memory（用于 step_cache）。

        Args:
            node_ids: [N_fetch] 需要拉取的节点全局 ID
            owners:   [N_fetch] 对应的 owner rank

        Returns:
            CommHandle，wait() 后结果存入 CommEngine 内部 recv buffer

        Notes:
            - 主要用于 StateManager.step() 的 step_cache 预加载
            - 可选 INT8 量化：通信量减少 4x，精度损失可忽略
        """
        ...

    # ---- 梯度同步 (Gradient) ---------------------------------------

    def all_reduce_gradients(self) -> None:
        """同步所有参数的梯度（DDP all-reduce）。

        DTDG 和 CTDG 共用，在 compute_stream 上执行（非 comm_stream）。

        Notes:
            - 通常由 optimizer.step() 前调用
            - 若使用 PyTorch DDP，此方法可为 no-op（DDP 自动 all-reduce）
        """
        ...
