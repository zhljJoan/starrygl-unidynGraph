from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_bts_sampler_module():
    return importlib.import_module("atc_starrygl_lib.lib.libstarrygl_sampler")


def is_bts_sampler_available() -> bool:
    try:
        load_bts_sampler_module()
    except Exception:
        return False
    return True
