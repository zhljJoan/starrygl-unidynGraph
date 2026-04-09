from __future__ import annotations


class ProviderRegistry:
    _providers: dict[str, type] = {}

    @classmethod
    def register(cls, graph_mode: str, provider_cls: type) -> None:
        cls._providers[graph_mode] = provider_cls

    @classmethod
    def resolve(cls, graph_mode: str) -> type:
        if graph_mode not in cls._providers:
            raise KeyError(f"No provider registered for graph_mode={graph_mode!r}")
        return cls._providers[graph_mode]
