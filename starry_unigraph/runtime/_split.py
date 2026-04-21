"""Shared train/val/test split utilities used by flare and online runtimes."""

from __future__ import annotations


def normalize_split_ratio(split_ratio: dict[str, float] | None) -> dict[str, float]:
    ratio = dict(split_ratio or {})
    train = max(0.0, float(ratio.get("train", 0.7)))
    val   = max(0.0, float(ratio.get("val",   0.15)))
    test  = max(0.0, float(ratio.get("test",  0.15)))
    total = train + val + test
    if total <= 0:
        return {"train": 0.7, "val": 0.15, "test": 0.15}
    return {"train": train / total, "val": val / total, "test": test / total}


def split_bounds(total: int, split: str, split_ratio: dict[str, float]) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    raw    = {key: total * split_ratio[key] for key in ("train", "val", "test")}
    counts = {key: int(raw[key]) for key in raw}
    remainder = total - sum(counts.values())
    if remainder > 0:
        order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
        for idx in range(remainder):
            counts[order[idx % len(order)]] += 1
    positive = [key for key in ("train", "val", "test") if split_ratio[key] > 0]
    if total >= len(positive):
        for key in positive:
            if counts[key] > 0:
                continue
            donor = max(positive, key=lambda item: counts[item])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[key]   += 1
    train_end = min(total, counts["train"])
    val_end   = min(total, train_end + counts["val"])
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, max(val_end, train_end)
    if split == "test":
        return val_end, total
    raise KeyError(f"Unsupported split: {split!r}")
