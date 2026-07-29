"""Tests for Phase 1.4 — EMC dataset exploration (site_explorer shared helpers)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofiber.registry.eth_scanner import scan_site
from neurofiber.registry.metadata_loader import load_abide2_metadata
from neurofiber.registry.site_explorer import (
    join_scan_meta,
    make_dti_diagnostic_df,
    make_site_manifest,
    make_validation_summary_df,
    site_stats_from_index_csv,
    site_stats_from_scan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, shape: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), str(path))


def _make_emc_subject(root: Path, sid: str, *, anat_shape=(186, 256, 256)) -> None:
    sess = root / sid / "session_1"
    _write_nifti(sess / "anat_1" / "anat.nii.gz", anat_shape)
    _write_nifti(sess / "rest_1" / "rest.nii.gz",  (64, 64, 37, 160))


def _meta_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _index_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# EMC-style scan (no DTI, pediatric, mixed sex)
# ---------------------------------------------------------------------------

class TestEmcStyleScan:

    def test_all_valid_when_anat_and_rest_present(self, tmp_path):
        for sid in ["29864", "29865", "29866"]:
            _make_emc_subject(tmp_path, sid)
        results = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        assert all(r.validation_status == "valid" for r in results)

    def test_two_anat_shape_variants_both_valid(self, tmp_path):
        _make_emc_subject(tmp_path, "sub-186", anat_shape=(186, 256, 256))
        _make_emc_subject(tmp_path, "sub-187", anat_shape=(187, 256, 256))
        results = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        shapes = {r.subject_id: r.anat_shape for r in results}
        assert shapes["sub-186"] == str((186, 256, 256))
        assert shapes["sub-187"] == str((187, 256, 256))
        assert all(r.validation_status == "valid" for r in results)

    def test_rest_volume_count(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        r = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")[0]
        assert r.rest_volume_count == 160

    def test_no_dti_fields_all_none(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        r = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")[0]
        assert r.dti_volume_count is None
        assert r.bval_count is None
        assert r.bvec_count is None
        assert r.recommended_action is None

    def test_fiq_null_in_metadata(self, tmp_path):
        csv = _meta_csv(tmp_path, [
            {"SITE_ID": "ABIDEII-EMC_1", "SUB_ID": 29864,
             "DX_GROUP": 1, "AGE_AT_SCAN": 9.01, "SEX": 1, "FIQ": float("nan")},
        ])
        meta = load_abide2_metadata(csv, site="EMC")
        assert meta["29864"].fiq is None

    def test_female_sex_mapped(self, tmp_path):
        csv = _meta_csv(tmp_path, [
            {"SITE_ID": "ABIDEII-EMC_1", "SUB_ID": 29870,
             "DX_GROUP": 2, "AGE_AT_SCAN": 8.5, "SEX": 2, "FIQ": float("nan")},
        ])
        meta = load_abide2_metadata(csv, site="EMC")
        assert meta["29870"].sex == "F"


# ---------------------------------------------------------------------------
# join_scan_meta
# ---------------------------------------------------------------------------

class TestJoinScanMeta:

    def test_joined_rows_have_meta_fields(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        scans = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        csv = _meta_csv(tmp_path, [{
            "SITE_ID": "ABIDEII-EMC_1", "SUB_ID": 29864,
            "DX_GROUP": 1, "AGE_AT_SCAN": 9.01, "SEX": 1, "FIQ": float("nan"),
        }])
        meta = load_abide2_metadata(csv, site="EMC")
        rows = join_scan_meta(scans, meta)
        assert rows[0]["diagnosis"] == "ASD"
        assert rows[0]["age"] == pytest.approx(9.01)
        assert rows[0]["sex"] == "M"

    def test_unmatched_subject_gets_none_meta(self, tmp_path):
        _make_emc_subject(tmp_path, "99999")
        scans = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        rows = join_scan_meta(scans, {})
        assert rows[0]["diagnosis"] is None


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

class TestDataFrameBuilders:

    def _scans(self, tmp_path):
        for sid in ["29864", "29865"]:
            _make_emc_subject(tmp_path, sid)
        return scan_site(tmp_path, site="EMC", dataset="ABIDE_II")

    def test_validation_summary_columns(self, tmp_path):
        df = make_validation_summary_df(self._scans(tmp_path))
        assert set(df.columns) == {
            "subject_id", "validation_status", "anat_exists",
            "dti_exists", "rest_exists", "missing_modalities",
        }

    def test_dti_diagnostic_columns(self, tmp_path):
        df = make_dti_diagnostic_df(self._scans(tmp_path))
        assert "subject_id" in df.columns
        assert "dti_volume_count" in df.columns
        assert "recommended_action" in df.columns

    def test_dti_diagnostic_all_none_for_no_dti_site(self, tmp_path):
        df = make_dti_diagnostic_df(self._scans(tmp_path))
        assert df["dti_volume_count"].isna().all()
        assert df["bval_count"].isna().all()


# ---------------------------------------------------------------------------
# site_stats_from_scan
# ---------------------------------------------------------------------------

class TestSiteStatsFromScan:

    def test_required_keys_present(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        scans = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        csv = _meta_csv(tmp_path, [{
            "SITE_ID": "ABIDEII-EMC_1", "SUB_ID": 29864,
            "DX_GROUP": 1, "AGE_AT_SCAN": 9.0, "SEX": 1, "FIQ": float("nan"),
        }])
        meta = load_abide2_metadata(csv, site="EMC")
        stats = site_stats_from_scan("EMC", scans, meta)
        for key in ["site", "subject_count", "asd_count", "control_count",
                    "valid_dti_count", "dti_mismatch_count", "common_voxel_size"]:
            assert key in stats

    def test_no_dti_counts_zero(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        scans = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        stats = site_stats_from_scan("EMC", scans, {})
        assert stats["valid_dti_count"] == 0
        assert stats["dti_mismatch_count"] == 0
        assert stats["common_volume_count"] is None


# ---------------------------------------------------------------------------
# site_stats_from_index_csv
# ---------------------------------------------------------------------------

class TestSiteStatsFromIndexCsv:

    def test_loads_stats_correctly(self, tmp_path):
        rows = [
            {"subject_id": "sub-001", "anat_exists": True, "dti_exists": False,
             "rest_exists": True, "validation_status": "valid",
             "dti_volume_count": None, "bval_count": None},
            {"subject_id": "sub-002", "anat_exists": True, "dti_exists": False,
             "rest_exists": True, "validation_status": "valid",
             "dti_volume_count": None, "bval_count": None},
        ]
        idx = _index_csv(tmp_path, rows)
        stats = site_stats_from_index_csv("ETH", idx)
        assert stats["subject_count"] == 2
        assert stats["valid_dti_count"] == 0

    def test_missing_files_count(self, tmp_path):
        rows = [
            {"subject_id": "sub-001", "anat_exists": True, "dti_exists": False,
             "rest_exists": False, "validation_status": "missing_files",
             "dti_volume_count": None, "bval_count": None},
        ]
        idx = _index_csv(tmp_path, rows)
        stats = site_stats_from_index_csv("ETH", idx)
        assert stats["missing_modality_count"] == 1


# ---------------------------------------------------------------------------
# make_site_manifest
# ---------------------------------------------------------------------------

class TestMakeSiteManifest:

    def test_manifest_structure(self, tmp_path):
        _make_emc_subject(tmp_path, "29864")
        scans = scan_site(tmp_path, site="EMC", dataset="ABIDE_II")
        csv = _meta_csv(tmp_path, [{
            "SITE_ID": "ABIDEII-EMC_1", "SUB_ID": 29864,
            "DX_GROUP": 1, "AGE_AT_SCAN": 9.0, "SEX": 1, "FIQ": float("nan"),
        }])
        meta = load_abide2_metadata(csv, site="EMC")
        m = make_site_manifest("EMC", "ABIDE_II", scans, meta)
        assert m["site"] == "EMC"
        assert m["subject_count"] == 1
        assert "demographics" in m
        assert "modality_presence" in m
        assert "dti_info" in m
        assert m["dti_info"]["dti_subjects"] == 0
