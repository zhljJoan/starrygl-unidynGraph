from __future__ import annotations

from pathlib import Path
from typing import Any

import pickle


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(payload, handle)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("rb") as handle:
        return pickle.load(handle)
