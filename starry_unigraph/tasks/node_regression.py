from __future__ import annotations

from .base import BaseTaskAdapter


class NodeRegressionTaskAdapter(BaseTaskAdapter):
    task_type = "snapshot_node_regression"

    def compute_loss(self, output: dict[str, Any]) -> float:
        return float(output.get("loss", 0.0))

    def format_prediction(self, output: dict[str, Any]) -> dict[str, Any]:
        return output