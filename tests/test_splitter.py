from __future__ import annotations

from pathlib import Path

import pytest

from neurofiber.data.subject_record import SubjectRecord
from neurofiber.registry.splitter import random_split, site_heldout_split, save_splits


def _make_records(n: int = 20, sites: list[str] | None = None) -> list[SubjectRecord]:
    sites = sites or ["NYU"] * n
    return [
        SubjectRecord(
            subject_id=f"sub-{i:03d}",
            dataset="ABIDE",
            diagnosis="ASD" if i % 2 == 0 else "CONTROL",
            age=10.0 + i,
            sex="M",
            site=sites[i % len(sites)],
            scanner="Siemens",
            modality="T1w",
            t1_path=f"/data/sub-{i:03d}/T1w.nii.gz",
        )
        for i in range(n)
    ]


def test_random_split_covers_all_records():
    records = _make_records(20)
    result = random_split(records)
    assert len(result) == 20
    assert all(r.split in {"train", "validation", "test"} for r in result)


def test_random_split_ratios_approximate():
    records = _make_records(100)
    result = random_split(records, train_ratio=0.7, val_ratio=0.15, seed=0)
    counts = {s: sum(1 for r in result if r.split == s) for s in ["train", "validation", "test"]}
    assert counts["train"] == 70
    assert counts["validation"] == 15
    assert counts["test"] == 15


def test_random_split_does_not_mutate_input():
    records = _make_records(10)
    original_splits = [r.split for r in records]
    random_split(records)
    assert [r.split for r in records] == original_splits


def test_random_split_is_deterministic():
    records = _make_records(20)
    r1 = random_split(records, seed=7)
    r2 = random_split(records, seed=7)
    assert [r.split for r in r1] == [r.split for r in r2]


def test_site_heldout_split_assigns_correct_splits():
    sites = ["NYU", "UCLA", "NYU", "UCLA", "MIT", "MIT", "MIT", "MIT", "NYU", "UCLA"]
    records = _make_records(10, sites=sites)
    result = site_heldout_split(records, heldout_site="MIT")
    heldout = [r for r in result if r.split == "site_heldout_test"]
    assert all(r.site == "MIT" for r in heldout)
    non_heldout = [r for r in result if r.split != "site_heldout_test"]
    assert all(r.site != "MIT" for r in non_heldout)


def test_site_heldout_missing_site_raises():
    records = _make_records(10)
    with pytest.raises(ValueError, match="No records found"):
        site_heldout_split(records, heldout_site="NONEXISTENT")


def test_save_splits_writes_csvs(tmp_path):
    records = _make_records(20)
    split_records = random_split(records)
    written = save_splits(split_records, tmp_path / "splits")
    assert "train" in written
    assert "validation" in written
    assert "test" in written
    for path in written.values():
        assert path.exists()
        assert path.stat().st_size > 0
