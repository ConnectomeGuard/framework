from __future__ import annotations

import copy
import random
from pathlib import Path

from neurofiber.data.subject_record import SubjectRecord
from neurofiber.registry.loader import save_dataset_index


def random_split(
    records: list[SubjectRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[SubjectRecord]:
    """Assign train / validation / test splits by shuffling all records."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.0")

    result = [copy.copy(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(result)

    n = len(result)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    for r in result[:train_end]:
        r.split = "train"
    for r in result[train_end:val_end]:
        r.split = "validation"
    for r in result[val_end:]:
        r.split = "test"

    return result


def site_heldout_split(
    records: list[SubjectRecord],
    heldout_site: str,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> list[SubjectRecord]:
    """Hold out one site entirely as test; split the remainder into train/validation."""
    heldout = [copy.copy(r) for r in records if r.site == heldout_site]
    trainval = [copy.copy(r) for r in records if r.site != heldout_site]

    if not heldout:
        raise ValueError(f"No records found for site '{heldout_site}'")

    rng = random.Random(seed)
    rng.shuffle(trainval)

    n = len(trainval)
    train_end = int(n * train_ratio)

    for r in trainval[:train_end]:
        r.split = "train"
    for r in trainval[train_end:]:
        r.split = "validation"
    for r in heldout:
        r.split = "site_heldout_test"

    return trainval + heldout


def save_splits(records: list[SubjectRecord], output_dir: str | Path) -> dict[str, Path]:
    """Write train.csv, validation.csv, and site_heldout_test.csv to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_map: dict[str, list[SubjectRecord]] = {}
    for r in records:
        split_map.setdefault(r.split, []).append(r)

    written: dict[str, Path] = {}
    for split_name, subset in split_map.items():
        dest = out / f"{split_name}.csv"
        save_dataset_index(subset, dest)
        written[split_name] = dest

    return written
