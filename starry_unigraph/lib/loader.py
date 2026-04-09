from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_bts_sampler_module():
    return importlib.import_module("starry_unigraph.lib.libstarrygl_sampler")
