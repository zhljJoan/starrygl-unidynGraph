from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from starry_unigraph.distributed import finalize_distributed, initialize_distributed
from starry_unigraph.session import SchedulerSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    session = SchedulerSession.from_config(Path(args.config))
    initialize_distributed(session.ctx.dist)
    try:
        start = time.perf_counter()
        train_summary = session.run_task()
        total_elapsed = time.perf_counter() - start
        test_summary = session.evaluate(split="test")
        payload = {
            "rank": session.ctx.dist.rank,
            "world_size": session.ctx.dist.world_size,
            "train_total_s": train_summary["train_total_s"],
            "train_total_local_s": train_summary["train_total_local_s"],
            "wall_total_s": total_elapsed,
            "train_epoch_loss": train_summary["train"][-1]["loss"] if train_summary["train"] else None,
            "val_epoch_loss": train_summary["eval"][-1]["loss"] if train_summary["eval"] else None,
            "test_loss": test_summary["loss"],
            "test_steps": test_summary["steps"],
            "test_elapsed_s": test_summary["elapsed_s"],
        }
        if session.ctx.dist.rank == 0:
            print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        finalize_distributed(session.ctx.dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
