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
        batch = next(iter(session.provider.build_train_iterator(session.ctx, split="train")))
        graph = batch.graph
        x = graph.srcdata["x"][: graph.num_dst_nodes()]
        y = graph.route.forward(x)
        payload = {
            "rank": session.ctx.dist.rank,
            "route": graph.route.describe(),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
        }
        print(json.dumps(payload, sort_keys=True))
    finally:
        finalize_distributed(session.ctx.dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
