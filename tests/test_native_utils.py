import pytest
import torch

from atc_starrygl_lib.lib import load_native_utils_module


def test_native_utils_stable_unique_and_first_ts() -> None:
    try:
        native = load_native_utils_module()
    except Exception as exc:
        pytest.skip(f"native_utils extension is not available: {exc}")

    values = torch.tensor([3, 1, 3, 2, 1], dtype=torch.long)
    unique, inverse = native.stable_unique(values)
    assert unique.tolist() == [3, 1, 2]
    assert inverse.tolist() == [0, 1, 0, 2, 1]

    ts = torch.tensor([9, 8, 9, 7, 8], dtype=torch.long)
    unique_ts, inverse_ts, first_ts = native.stable_unique_with_ts(values, ts)
    assert unique_ts.tolist() == [3, 1, 2]
    assert inverse_ts.tolist() == [0, 1, 0, 2, 1]
    assert first_ts.tolist() == [9, 8, 7]

    out = native.first_ts_for_lids(
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
        torch.tensor([1, 0, 1, 2], dtype=torch.long),
        3,
    )
    assert out.tolist() == [6, 5, 8]
