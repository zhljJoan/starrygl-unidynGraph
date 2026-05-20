from pathlib import Path

from atc_starrygl_lib.core.types import RuntimeContext
from atc_starrygl_lib.ctdg.runtime.backend import MemShareTemporalSamplingBackend


def test_ctdg_prepare_uses_new_pipeline_for_event_config(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_run_preprocess_pipeline(**kwargs):
        called.update(kwargs)
        out = Path(kwargs["out_dir"])
        (out / "graph.pt").write_bytes(b"x")
        (out / "dist.pt").write_bytes(b"x")
        (out / "meta.json").write_text("{}", encoding="utf-8")
        (out / "rank_000.pt").write_bytes(b"x")
        (out / "feature_000.pt").write_bytes(b"x")
        return {"ranks": [object()], "meta": {"mode": "event"}}

    monkeypatch.setattr("atc_starrygl_lib.preprocess.pipeline.run_preprocess_pipeline", fake_run_preprocess_pipeline)
    ctx = RuntimeContext(
        config={
            "graph": {"mode": "ctdg", "source": str(tmp_path / "edges.csv")},
            "task": {"name": "edge_pred"},
            "runtime": {"device": "cpu"},
            "preprocess": {
                "use_new_pipeline": True,
                "mode": "event",
                "partition_algorithm": "speed_partition",
                "chunks_per_rank": 2,
                "batch_size": 128,
                "num_windows": 8,
            },
        },
        artifact_root=tmp_path / "artifacts",
        rank=0,
        world_size=1,
    )
    backend = MemShareTemporalSamplingBackend()
    bundle = backend.prepare(ctx)

    assert called["mode"] == "event"
    assert called["algorithm"] == "speed_partition"
    assert called["batch_size"] == 128
    assert called["num_windows"] == 8
    assert bundle.files["graph"].name == "graph.pt"
    assert bundle.files["rank_000"].name == "rank_000.pt"
