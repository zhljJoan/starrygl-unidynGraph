from __future__ import annotations

from typing import TypeVar

from .errors import RegistryError

T = TypeVar("T")


class _Registry:
    def __init__(self) -> None:
        self._items: dict[str, object] = {}

    def register(self, name: str, item: T) -> T:
        key = name.strip().lower()
        if not key:
            raise RegistryError("registry name must be non-empty")
        if key in self._items:
            raise RegistryError(f"duplicate registry entry: {key}")
        self._items[key] = item
        return item

    def get(self, name: str) -> T:
        key = name.strip().lower()
        try:
            return self._items[key]  # type: ignore[return-value]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(f"unknown registry entry {key!r}; known: {known}") from exc


BackendRegistry = _Registry()
TaskRegistry = _Registry()
