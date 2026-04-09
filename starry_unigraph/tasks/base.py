from __future__ import annotations

from typing import Any


class BaseTaskAdapter:
    task_type = "base"

    def compute_loss(self, output: dict[str, Any]) -> float:
        return float(output.get("loss", 0.0))

    def format_prediction(self, output: dict[str, Any]) -> dict[str, Any]:
        return output
