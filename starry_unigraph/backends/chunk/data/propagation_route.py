"""Compatibility import for DTDG model-layer propagation routes."""

from starry_unigraph.models.layers import route as _route
from starry_unigraph.models.layers.route import ChunkPropagationRoute, _PropagationContext

dist = _route.dist

__all__ = ["ChunkPropagationRoute", "_PropagationContext", "dist"]
