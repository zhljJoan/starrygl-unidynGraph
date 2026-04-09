from __future__ import annotations

from starry_unigraph.types import SessionContext


def dtdg_pipeline(session_ctx: SessionContext) -> str:
    return str(session_ctx.config.get("dtdg", {}).get("pipeline", "flare_native"))
