"""StateManager Protocol: 统一节点状态管理接口。

DTDG（DTDGStateManager）和 CTDG（CTDGStateManager）各自实现此 Protocol，
PipelineEngine 通过统一接口驱动，不感知具体实现。

四个核心方法对应 PipelineEngine 的四个阶段：
  prepare  → prepare_data 阶段：预取本 chunk 需要的节点状态
  patch    → sync_data 阶段：等待异步预取完成，补全 miss
  update   → post_compute 阶段：写回更新并异步推送增量
  step     → chunk 切换：维护缓存，预加载下一 chunk 的状态
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from torch import Tensor


class StateHandle:
    """prepare() 返回的状态句柄，持有预取 buffer 的引用。

    字段由具体实现填充，patch() 后保证所有 buffer 完整有效。
    """
    # 具体子类添加字段，例如：
    #   gathered_states: Tensor        # [N, d] 预取到的节点状态
    #   miss_mask: Tensor              # [N] bool，标记未命中节点
    #   comm_handle: CommHandle        # 异步通信句柄
    pass


@runtime_checkable
class StateManager(Protocol):
    """统一状态管理 Protocol。

    DTDG 和 CTDG 分别实现，PipelineEngine 只依赖此接口。
    """

    def prepare(self, needed_ids: Tensor, query_context: Any) -> StateHandle:
        """预取本 chunk 所需的节点状态到连续 buffer。

        Args:
            needed_ids:     [N] 本 chunk 中所有需要状态的节点全局 ID
            query_context:  chunk 上下文（如 SpatialDeps / StateDeps），
                            用于决定从本地缓存读取还是异步 fetch

        Returns:
            StateHandle，持有预取 buffer 和异步句柄引用

        Implementation notes:
            DTDG:
              1. 从 owned_embedding 中 gather 本地节点状态
              2. 对跨分区节点，调用 comm.async_fetch()，填入 handle.comm_handle
              3. 本地命中节点直接填入 handle.gathered_states
            CTDG:
              1. 先查 hot_cache → decay_cache → owned_memory（三层查找）
              2. miss 节点通过 comm.async_fetch() 异步拉取
              3. mailbox 中积压的增量一并应用
        """
        ...

    def patch(self, handle: StateHandle) -> None:
        """等待异步预取完成，将远程 fetch 结果写入 handle.gathered_states。

        Args:
            handle: prepare() 返回的句柄

        Notes:
            - 此方法对应 PipelineEngine 的 sync_data 阶段
            - 内部调用 handle.comm_handle.wait() 后补全 miss 位置
            - patch 后 handle.gathered_states 完整，可直接传入 model
        """
        ...

    def update(self, node_ids: Tensor, new_state: Tensor) -> None:
        """将 compute 阶段产出的新状态写回，并异步推送增量给其他 rank。

        Args:
            node_ids:  [N] 需要更新状态的节点全局 ID
            new_state: [N, d] 新的状态向量（embedding / memory）

        Notes:
            DTDG: 写回 owned_embedding，调用 comm.async_push_delta()
            CTDG: 写回 owned_memory，更新 hot_cache / decay_cache，
                  调用 comm.async_push_delta() 推送 memory delta
        """
        ...

    def step(self, next_chunk_idx: int) -> None:
        """chunk 切换时的缓存维护与预加载。

        Args:
            next_chunk_idx: 下一个 chunk 在序列中的索引

        Notes:
            DTDG: 可选地预加载下一快照的跨分区 embedding（step_cache）
            CTDG: 调用 decay_cache.step() 更新重用优先级，驱逐过期节点
        """
        ...
