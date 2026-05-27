from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_bts_sampler_module():
    return importlib.import_module("atc_starrygl_lib.lib.libstarrygl_sampler")


@lru_cache(maxsize=1)
def load_native_utils_module():
    return importlib.import_module("atc_starrygl_lib.lib.native_utils")


@lru_cache(maxsize=1)
def load_adaptive_split_module():
    return importlib.import_module("atc_starrygl_lib.lib.adaptive_split_cpp")


def is_bts_sampler_available() -> bool:
    try:
        load_bts_sampler_module()
    except Exception:
        return False
    return True


def is_native_utils_available() -> bool:
    try:
        load_native_utils_module()
    except Exception:
        return False
    return True


def is_adaptive_split_available() -> bool:
    try:
        load_adaptive_split_module()
    except Exception:
        return False
    return True
