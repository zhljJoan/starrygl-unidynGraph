"""Tests for orthogonal Graph Mode × Task Type separation."""
import sys
import pytest
import torch
from torch import Tensor

# Test Phase 1: Task adapters and data structures
def test_task_registry():
    """Verify task registry has all task types."""
    from starry_unigraph.registry import TaskRegistry

    tasks = TaskRegistry.list_tasks()
    assert "edge_predict" in tasks
    assert "node_regression" in tasks
    assert "node_classification" in tasks
    assert "temporal_link_prediction" in tasks  # Backward compat


def test_edge_predict_adapter():
    """Test EdgePredictAdapter instantiation and basic methods."""
    from starry_unigraph.tasks import EdgePredictAdapter
    from starry_unigraph.data.batch_data import BatchData

    adapter = EdgePredictAdapter()
    assert adapter.task_type == "edge_predict"

    # Create dummy batch
    batch = BatchData(
        mfg=None,
        node_ids=torch.tensor([0, 1, 2, 3]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        neg_src=torch.tensor([0, 1]),
        neg_dst=torch.tensor([2, 3]),
    )

    # Test compute_loss (should handle gracefully)
    model_output = {
        "pos_score": torch.tensor([0.5, 0.7]),
        "neg_score": torch.tensor([0.2, 0.3]),
    }
    loss = adapter.compute_loss(model_output, batch)
    assert isinstance(loss, Tensor)

    # Test compute_metrics
    metrics = adapter.compute_metrics(model_output, batch)
    assert isinstance(metrics, dict)
    assert "auc" in metrics or "ap" in metrics


def test_node_regression_adapter():
    """Test NodeRegressionTaskAdapter."""
    from starry_unigraph.tasks import NodeRegressionTaskAdapter
    from starry_unigraph.data.batch_data import BatchData

    adapter = NodeRegressionTaskAdapter()
    assert adapter.task_type == "node_regression"

    batch = BatchData(
        mfg=None,
        node_ids=torch.tensor([0, 1, 2]),
        target_nodes=torch.tensor([0, 1]),
        labels=torch.tensor([[1.0], [2.0]]),
    )

    model_output = {
        "node_pred": torch.tensor([[1.1], [2.1]]),
    }

    loss = adapter.compute_loss(model_output, batch)
    assert isinstance(loss, Tensor)

    metrics = adapter.compute_metrics(model_output, batch)
    assert "mae" in metrics
    assert "rmse" in metrics


def test_node_classify_adapter():
    """Test NodeClassifyAdapter."""
    from starry_unigraph.tasks import NodeClassifyAdapter
    from starry_unigraph.data.batch_data import BatchData

    adapter = NodeClassifyAdapter()
    assert adapter.task_type == "node_classification"

    batch = BatchData(
        mfg=None,
        node_ids=torch.tensor([0, 1, 2]),
        target_nodes=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1]),
    )

    model_output = {
        "logits": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }

    loss = adapter.compute_loss(model_output, batch)
    assert isinstance(loss, Tensor)

    metrics = adapter.compute_metrics(model_output, batch)
    assert "accuracy" in metrics


# Test Phase 2: Protocols and data structures
def test_sample_config():
    """Test SampleConfig dataclass."""
    from starry_unigraph.data.sample_config import SampleConfig

    cfg = SampleConfig(
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        neg_ratio=2,
        num_neighbors=[10, 5],
    )

    assert cfg.pos_src is not None
    assert cfg.neg_ratio == 2
    assert cfg.num_neighbors == [10, 5]


def test_batch_data():
    """Test BatchData dataclass."""
    from starry_unigraph.data.batch_data import BatchData

    batch = BatchData(
        mfg=None,
        node_ids=torch.tensor([0, 1, 2]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        labels=torch.tensor([1.0, 2.0]),
    )

    assert batch.pos_src is not None
    assert batch.labels is not None


# Test Phase 3 & 4: Protocols and models
def test_task_head_edge_predict():
    """Test EdgePredictHead module."""
    from starry_unigraph.models.task_head import EdgePredictHead
    from starry_unigraph.data.batch_data import BatchData

    head = EdgePredictHead(embedding_dim=64)
    embeddings = torch.randn(10, 64)
    batch = BatchData(
        mfg=None,
        node_ids=torch.arange(10),
        pos_src=torch.tensor([0, 1, 2]),
        pos_dst=torch.tensor([1, 2, 3]),
        neg_src=torch.tensor([0, 1, 2]),
        neg_dst=torch.tensor([4, 5, 6]),
    )

    output = head(embeddings, batch)
    assert "pos_score" in output
    assert "neg_score" in output
    assert output["pos_score"].shape[0] == 3


def test_task_head_node_regression():
    """Test NodeRegressHead module."""
    from starry_unigraph.models.task_head import NodeRegressHead
    from starry_unigraph.data.batch_data import BatchData

    head = NodeRegressHead(embedding_dim=64, output_dim=1)
    embeddings = torch.randn(10, 64)
    batch = BatchData(
        mfg=None,
        node_ids=torch.arange(10),
        target_nodes=torch.tensor([0, 1, 2]),
    )

    output = head(embeddings, batch)
    assert "node_pred" in output
    assert output["node_pred"].shape == (3, 1)


def test_wrapped_model():
    """Test WrappedModel composition."""
    from starry_unigraph.models.wrapped import WrappedModel
    from starry_unigraph.models.task_head import EdgePredictHead
    from starry_unigraph.data.batch_data import BatchData

    # Create dummy backbone
    class DummyBackbone(torch.nn.Module):
        def forward(self, mfg, state):
            # Return dummy embeddings for 10 nodes
            return torch.randn(10, 64)

    backbone = DummyBackbone()
    head = EdgePredictHead(embedding_dim=64)
    model = WrappedModel(backbone, head)

    batch = BatchData(
        mfg=None,
        node_ids=torch.arange(10),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
    )

    output = model.predict({}, batch)
    assert "pos_score" in output


# Test Phase 5: Backend adapters
def test_backend_adapters():
    """Test that backend adapters can be instantiated."""
    from starry_unigraph.runtime.backend_adapters import DummyStateManager

    mgr = DummyStateManager()
    state = mgr.prepare(torch.tensor([0, 1, 2]))
    assert isinstance(state, dict)

    mgr.update({}, None)
    mgr.reset()


# Test Phase 6: PipelineEngine
def test_pipeline_engine():
    """Test PipelineEngine instantiation."""
    from starry_unigraph.runtime.engine import PipelineEngine
    from starry_unigraph.runtime.backend_adapters import DummyStateManager
    from starry_unigraph.tasks import EdgePredictAdapter

    class DummyBackend:
        def iter_batches(self, split, batch_size):
            return []
        def reset(self):
            pass
        def describe(self):
            return {}

    backend = DummyBackend()
    state_mgr = DummyStateManager()
    model = torch.nn.Identity()
    task_adapter = EdgePredictAdapter()

    engine = PipelineEngine(backend, state_mgr, model, task_adapter)
    assert engine.backend is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
