from atc_starrygl_lib.ctdg.runtime.backend import MemShareTemporalSamplingBackend


def test_legacy_prepare_config_injects_event_time_ptr_from_preprocess(monkeypatch, tmp_path) -> None:
    called = {}

    def fake_build_dataset(**kwargs):
        called.update(kwargs)
        import torch

        return {"time_ptr_2": torch.tensor([[0, 2], [2, 5]], dtype=torch.long), "split": torch.tensor([0, 0, 1, 2, 2], dtype=torch.uint8)}

    monkeypatch.setattr("atc_starrygl_lib.preprocess.dataset.build_dataset", fake_build_dataset)
    backend = MemShareTemporalSamplingBackend()
    cfg = {
        "graph": {"mode": "ctdg", "source": str(tmp_path / "edges.csv")},
        "task": {"name": "edge_pred"},
        "runtime": {"device": "cpu"},
        "preprocess": {"mode": "event", "batch_size": 2, "num_windows": 3},
    }
    out = backend._build_session_config_with_event_time_ptr(cfg, graph_cfg=cfg["graph"], prep_cfg=cfg["preprocess"])
    assert called["mode"] == "event"
    assert called["batch_size"] == 2
    assert called["num_windows"] == 3
    assert out["preprocess"]["time_ptr_2"] == [[0, 2], [2, 5]]
    assert out["preprocess"]["split"] == [0, 0, 1, 2, 2]
