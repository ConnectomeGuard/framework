"""Tests for Phase 1.5 — Georgetown dataset exploration."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofiber.registry.eth_scanner import scan_site
from neurofiber.registry.metadata_loader import load_abide2_metadata, _read_csv_any_encoding
from neurofiber.registry.site_explorer import (
    join_scan_meta,
    make_validation_summary_df,
    make_dti_diagnostic_df,
    site_stats_from_scan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, shape: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), str(path))


def _make_gu_subject(root: Path, sid: str, *,
                     missing: list[str] | None = None) -> None:
    missing = missing or []
    sess = root / sid / "session_1"
    if "anat" not in missing:
        _write_nifti(sess / "anat_1" / "anat.nii.gz", (176, 256, 256))
    if "rest" not in missing:
        _write_nifti(sess / "rest_1" / "rest.nii.gz", (64, 64, 43, 152))


def _make_meta_csv(path: Path, rows: list[dict], encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding=encoding)
    return path


def _sample_meta_row(sub_id: int, dx: int = 1, age: float = 10.0,
                     sex: int = 1, fiq: float | None = 110.0) -> dict:
    return {
        "SITE_ID": "ABIDEII-GU_1", "SUB_ID": sub_id,
        "DX_GROUP": dx, "AGE_AT_SCAN": age, "SEX": sex,
        "FIQ": fiq if fiq is not None else float("nan"),
    }


# ---------------------------------------------------------------------------
# _read_csv_any_encoding — encoding detection
# ---------------------------------------------------------------------------

class TestReadCsvAnyEncoding:

    def test_reads_utf8(self, tmp_path):
        p = tmp_path / "meta.csv"
        p.write_text("SUB_ID,NAME\n1,Alice\n", encoding="utf-8")
        df = _read_csv_any_encoding(p)
        assert len(df) == 1

    def test_reads_latin1_without_error(self, tmp_path):
        p = tmp_path / "meta.csv"
        # Write bytes that are valid latin-1 but invalid UTF-8 (0xe9 = é)
        p.write_bytes(b"SUB_ID,NAME\n1,Caf\xe9\n")
        df = _read_csv_any_encoding(p)
        assert len(df) == 1

    def test_fallback_produces_correct_row_count(self, tmp_path):
        """A CSV written in latin-1 is fully readable via the auto-detection path."""
        rows = [{"SUB_ID": i, "DX_GROUP": 1} for i in range(5)]
        p = tmp_path / "meta.csv"
        pd.DataFrame(rows).to_csv(p, index=False, encoding="latin-1")
        df = _read_csv_any_encoding(p)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# Georgetown-style scan (no DTI, 106 subjects expected pattern)
# ---------------------------------------------------------------------------

class TestGeorgetownStyleScan:

    def test_finds_all_subjects(self, tmp_path):
        for sid in ["28741", "28742", "28743"]:
            _make_gu_subject(tmp_path, sid)
        records = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        assert len(records) == 3

    def test_valid_subject_status(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        rec = scan_site(tmp_path, site="GU", dataset="ABIDE_II")[0]
        assert rec.validation_status == "valid"
        assert rec.anat_exists
        assert rec.rest_exists
        assert not rec.dti_exists

    def test_anat_shape_recorded(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        rec = scan_site(tmp_path, site="GU", dataset="ABIDE_II")[0]
        assert rec.anat_shape == str((176, 256, 256))

    def test_rest_volume_count_152(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        rec = scan_site(tmp_path, site="GU", dataset="ABIDE_II")[0]
        assert rec.rest_volume_count == 152

    def test_dti_fields_none(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        rec = scan_site(tmp_path, site="GU", dataset="ABIDE_II")[0]
        assert rec.dti_volume_count is None
        assert rec.bval_count is None
        assert rec.recommended_action is None

    def test_missing_rest_flagged(self, tmp_path):
        _make_gu_subject(tmp_path, "28741", missing=["rest"])
        rec = scan_site(tmp_path, site="GU", dataset="ABIDE_II")[0]
        assert rec.validation_status == "missing_files"
        assert not rec.rest_exists


# ---------------------------------------------------------------------------
# load_abide2_metadata with GU-style latin-1 CSV
# ---------------------------------------------------------------------------

class TestLoadGuMetadata:

    def _meta_csv(self, tmp_path, encoding="latin-1") -> Path:
        rows = [
            _sample_meta_row(28741, dx=1, age=10.5, sex=1, fiq=112),
            _sample_meta_row(28742, dx=2, age=11.2, sex=2, fiq=None),
        ]
        return _make_meta_csv(tmp_path / "ABIDEII-GU_1.csv", rows, encoding=encoding)

    def test_loads_latin1_csv(self, tmp_path):
        csv = self._meta_csv(tmp_path, encoding="latin-1")
        meta = load_abide2_metadata(csv, site="GU")
        assert len(meta) == 2

    def test_diagnosis_asd_mapped(self, tmp_path):
        csv = self._meta_csv(tmp_path)
        meta = load_abide2_metadata(csv, site="GU")
        assert meta["28741"].diagnosis == "ASD"

    def test_diagnosis_control_mapped(self, tmp_path):
        csv = self._meta_csv(tmp_path)
        meta = load_abide2_metadata(csv, site="GU")
        assert meta["28742"].diagnosis == "CONTROL"

    def test_sex_female_mapped(self, tmp_path):
        csv = self._meta_csv(tmp_path)
        meta = load_abide2_metadata(csv, site="GU")
        assert meta["28742"].sex == "F"

    def test_null_fiq_handled(self, tmp_path):
        csv = self._meta_csv(tmp_path)
        meta = load_abide2_metadata(csv, site="GU")
        assert meta["28742"].fiq is None

    def test_age_parsed(self, tmp_path):
        csv = self._meta_csv(tmp_path)
        meta = load_abide2_metadata(csv, site="GU")
        assert abs(meta["28741"].age - 10.5) < 0.001


# ---------------------------------------------------------------------------
# join_scan_meta
# ---------------------------------------------------------------------------

class TestJoinScanMeta:

    def test_metadata_fields_populated(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        scans = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        csv = _make_meta_csv(
            tmp_path / "meta.csv",
            [_sample_meta_row(28741, dx=1, age=10.5, sex=1, fiq=112)],
        )
        meta = load_abide2_metadata(csv, site="GU")
        rows = join_scan_meta(scans, meta)
        assert rows[0]["diagnosis"] == "ASD"
        assert rows[0]["sex"] == "M"
        assert abs(rows[0]["age"] - 10.5) < 0.001

    def test_unmatched_subject_gets_none(self, tmp_path):
        _make_gu_subject(tmp_path, "99999")
        scans = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        rows = join_scan_meta(scans, {})
        assert rows[0]["diagnosis"] is None


# ---------------------------------------------------------------------------
# site_stats_from_scan for GU
# ---------------------------------------------------------------------------

class TestSiteStatsGU:

    def test_no_dti_in_stats(self, tmp_path):
        for sid in ["28741", "28742"]:
            _make_gu_subject(tmp_path, sid)
        scans = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        stats = site_stats_from_scan("GU", scans, {})
        assert stats["valid_dti_count"] == 0
        assert stats["dti_mismatch_count"] == 0
        assert stats["common_volume_count"] is None

    def test_all_modalities_count_anat_rest_only(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        scans = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        stats = site_stats_from_scan("GU", scans, {})
        assert stats["all_modalities_count"] == 1

    def test_mixed_sex_counted(self, tmp_path):
        _make_gu_subject(tmp_path, "28741")
        _make_gu_subject(tmp_path, "28742")
        scans = scan_site(tmp_path, site="GU", dataset="ABIDE_II")
        rows = [
            _sample_meta_row(28741, sex=1),   # M
            _sample_meta_row(28742, sex=2),   # F
        ]
        csv = _make_meta_csv(tmp_path / "meta.csv", rows)
        meta = load_abide2_metadata(csv, site="GU")
        stats = site_stats_from_scan("GU", scans, meta)
        assert stats["male_count"] == 1
        assert stats["female_count"] == 1


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

class TestDFBuilders:

    def _scans(self, tmp_path):
        for sid in ["28741", "28742"]:
            _make_gu_subject(tmp_path, sid)
        return scan_site(tmp_path, site="GU", dataset="ABIDE_II")

    def test_validation_summary_columns(self, tmp_path):
        df = make_validation_summary_df(self._scans(tmp_path))
        assert "validation_status" in df.columns
        assert "anat_exists" in df.columns
        assert "dti_exists" in df.columns

    def test_dti_diagnostic_all_none(self, tmp_path):
        df = make_dti_diagnostic_df(self._scans(tmp_path))
        assert df["dti_volume_count"].isna().all()
        assert df["bval_count"].isna().all()
