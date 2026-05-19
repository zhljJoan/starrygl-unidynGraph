from __future__ import annotations

from typing import Protocol

from torch import Tensor

from atc_starrygl_lib.core.types import Batch


class TemporalModel(Protocol):
    def __call__(self, batch: Batch) -> dict[str, Tensor]:
        ...
