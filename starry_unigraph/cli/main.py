from __future__ import annotations

import argparse
from pathlib import Path

from starry_unigraph import SchedulerSession
from starry_unigraph.distributed import finalize_distributed, initialize_distributed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starry-unigraph")
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare")
    subparsers.add_parser("train")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--split", default="test")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    session = SchedulerSession.from_config(Path(args.config))
    initialize_distributed(session.ctx.dist)

    try:
        if args.command == "prepare":
            session.prepare_data()
        elif args.command == "train":
            session.run_task()
        elif args.command == "predict":
            session.build_runtime()
            session.predict(split=args.split)
        elif args.command == "resume":
            session.load_checkpoint(args.checkpoint)
            session.run_task()
    finally:
        finalize_distributed(session.ctx.dist)
    return 0
