from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.make_ctdg_benchmark_config import main


def test_make_ctdg_benchmark_config_defaults_match_repo_benchmark_conventions(tmp_path: Path, monkeypatch) -> None:
    base_config = tmp_path / "base.json"
    output = tmp_path / "out.json"
    base_config.write_text(
        json.dumps(
            {
                "graph": {"source": "placeholder"},
                "model": {"name": "general"},
                "task": {"name": "edge_prediction"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_ctdg_benchmark_config.py",
            "--base-config",
            str(base_config),
            "--output",
            str(output),
            "--source",
            "/mnt/data/zlj/tgl_data/DATA/LASTFM",
            "--batch-size",
            "4000",
            "--fanout",
            "10",
            "--feature-device",
            "cuda",
            "--epochs",
            "3",
            "--dataset-name",
            "lastfm",
            "--memshare-memory-sync",
        ],
    )

    main()

    cfg = json.loads(output.read_text(encoding="utf-8"))
    assert cfg["graph"]["random_node_feat_seed"] == 0
    assert cfg["graph"]["random_edge_feat_seed"] == 0
    assert cfg["runtime"]["memory_sync_mode"] == "memshare_historical"
    assert cfg["runtime"]["negative_test_policy"] == "global"


def test_make_ctdg_benchmark_config_defaults_to_memshare_public_exact(tmp_path: Path, monkeypatch) -> None:
    base_config = tmp_path / "base.json"
    output = tmp_path / "out.json"
    base_config.write_text(
        json.dumps(
            {
                "graph": {"source": "placeholder"},
                "model": {"name": "general"},
                "task": {"name": "edge_prediction"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_ctdg_benchmark_config.py",
            "--base-config",
            str(base_config),
            "--output",
            str(output),
            "--source",
            "/mnt/data/zlj/tgl_data/DATA/WikiTalk",
            "--batch-size",
            "12000",
            "--fanout",
            "10",
            "--feature-device",
            "cuda",
            "--epochs",
            "1",
        ],
    )

    main()

    cfg = json.loads(output.read_text(encoding="utf-8"))
    assert cfg["runtime"]["memory_sync_mode"] == "memshare_public_exact"
