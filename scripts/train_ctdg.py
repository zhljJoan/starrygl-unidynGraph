from __future__ import annotations

from atc_starrygl_lib.core.registry import BackendRegistry
from atc_starrygl_lib.ctdg import MemShareCTDGBackend
from atc_starrygl_lib.tasks import register_builtin_tasks


BackendRegistry.register("ctdg", MemShareCTDGBackend)
register_builtin_tasks()


def main() -> None:
    raise SystemExit("wire config loading here; core TrainingSession is ready")


if __name__ == "__main__":
    main()
