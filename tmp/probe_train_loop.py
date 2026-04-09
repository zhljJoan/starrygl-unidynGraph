from __future__ import annotations

import argparse
import json

from starry_unigraph.distributed import finalize_distributed, initialize_distributed
from starry_unigraph.session import SchedulerSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    session = SchedulerSession.from_config(args.config)
    initialize_distributed(session.ctx.dist)
    try:
        session.build_runtime()
        outputs = []
        for batch in session.provider.build_train_iterator(session.ctx, split="train"):
            outputs.append(session.provider.run_train_step(batch)["loss"])
        print(json.dumps({"rank": session.ctx.dist.rank, "count": len(outputs), "losses": outputs}))
    finally:
        finalize_distributed(session.ctx.dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
