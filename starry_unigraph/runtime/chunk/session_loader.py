"""ChunkRuntimeLoader: session.py 的 chunk 线路入口.

转发至 backends/chunk/runtime/loader.py 中的完整实现.
保留此文件供 session.py 的 build_runtime 路径使用.
"""

from starry_unigraph.backends.chunk.runtime.loader import (
    ChunkRuntimeLoader,
    redistribute_preprocessed,
    rebuild_from_scratch,
)

__all__ = ["ChunkRuntimeLoader", "redistribute_preprocessed", "rebuild_from_scratch"]
