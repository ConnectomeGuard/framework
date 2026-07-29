"""
Tests for Phase 2R.2 — Tensor Estimation from processed_v2

Coverage:
  - Raw write guard
  - Input validation (missing files, bad shapes)
  - Preprocessing report gate (only status=success proceeds)
  - Tensor outputs created (FA/MD/AD/RD + report)
  - NaN/Inf handling
  - Summary CSV schema
  - Site summary schema
  - Comparison CSV schema
  - Subject failure isolation (one bad subject doesn't break others)
  - skip_if_exists resume behaviour
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tensor.tensor_estimation_v2 import (
    COMPARISON_CSV_FIELDS,
    SITE_CSV_FIELDS,
    SUBJECT_CSV_FIELDS,
    TensorRecord,
    estimate_tensors_subject,
    run_tensor_batch,
    write_site_summary,
    write_subject_summary,
    write_v1_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_dwi(tmp_path: Path, shape=(10, 10, 5, 33)) -> Path:
    """Write a minimal synthetic 4D DWI NIfTI."""
    rng  = np.random.default_rng(42)
    data = rng.random(shape, dtype=np.float32) * 1000
    # Make first volume brighter (B0)
    data[..., 0] *= 5
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "dwi_preprocessed.nii.gz"
    nib.save(img, str(p))
    return p


def _bvals(tmp_path: Path, n: int = 33) -> Path:
    vals = np.zeros(n)
    vals[1:] = 1000
    p = tmp_path / "dwi_preprocessed.bval"
    np.savetxt(str(p), vals[np.newaxis, :], fmt="%.1f")
    return p


def _bvecs(tmp_path: Path, n: int = 33) -> Path:
    rng  = np.random.default_rng(0)
    vecs = rng.standard_normal((3, n))
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    norms[norms == 0] = 1
    vecs[:, 1:] /= norms[:, 1:]
    vecs[:, 0]   = 0.0
    p = tmp_path / "dwi_preprocessed.bvec"
    np.savetxt(str(p), vecs, fmt="%.6f")
    return p


def _preproc_report(tmp_path: Path, status: str = "success") -> Path:
    p = tmp_path / "preprocessing_report.json"
    p.write_text(json.dumps({"status": status, "warning_count": 0}))
    return p


def _make_dti_dir(
    base: Path,
    site="BNI", dataset="ABIDEII-BNI_1", subject="29006",
    preproc_status="success",
    n_vols=33,
) -> Path:
    """
    Create a minimal processed_v2 dti_1 directory:
      base/abide_ii/<site_folder>/<dataset>/<subject>/session_1/dti_1/
    """
    from neurofiber.tensor.tensor_estimation_v2 import _SITE_FOLDER_MAP
    folder = _SITE_FOLDER_MAP.get(site, site.lower())
    dti_dir = base / "abide_ii" / folder / dataset / subject / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True, exist_ok=True)

    _synthetic_dwi(dti_dir, shape=(8, 8, 4, n_vols))
    _bvals(dti_dir, n_vols)
    _bvecs(dti_dir, n_vols)
    _preproc_report(dti_dir, status=preproc_status)
    return dti_dir


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_output_inside_raw(self, tmp_path):
        raw_root = tmp_path / "raw"
        dti_dir  = raw_root / "abide_ii" / "bni" / "ABIDEII-BNI_1" / "99999" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_dwi(dti_dir)
        _bvals(dti_dir)
        _bvecs(dti_dir)
        _preproc_report(dti_dir)

        with pytest.raises((ValueError, AssertionError)):
            estimate_tensors_subject(
                dti_dir=dti_dir,
                output_root=raw_root / "abide_ii",
                raw_root=raw_root,
            )


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_dwi_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "failed"
        assert "dwi_preprocessed.nii.gz" in (rec.error_message or "")

    def test_missing_bval_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.bval").unlink()

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "failed"

    def test_missing_bvec_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.bvec").unlink()

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "failed"

    def test_missing_preproc_report_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "preprocessing_report.json").unlink()

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "failed"
        assert "preprocessing_report" in (rec.error_message or "")


# ---------------------------------------------------------------------------
# TestPreprocessingGate
# ---------------------------------------------------------------------------

class TestPreprocessingGate:
    def test_skips_if_preproc_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, preproc_status="failed")

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "skipped"
        assert "failed" in (rec.warning_message or "")

    def test_processes_if_preproc_success(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, preproc_status="success")

        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "success"


# ---------------------------------------------------------------------------
# TestTensorOutputs
# ---------------------------------------------------------------------------

class TestTensorOutputs:
    def test_tensor_maps_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        tensor_dir = dti_dir / "tensor"
        for fname in ["FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz"]:
            assert (tensor_dir / fname).exists(), f"Missing {fname}"

    def test_tensor_fit_report_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        report = dti_dir / "tensor" / "tensor_fit_report.json"
        assert report.exists()
        d = json.loads(report.read_text())
        assert d["status"] == "success"

    def test_fa_values_in_valid_range(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "success"
        assert rec.fa_mean is not None
        assert 0.0 <= rec.fa_mean <= 1.0

    def test_fa_shape_matches_dwi(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        dwi = nib.load(str(dti_dir / "dwi_preprocessed.nii.gz"))
        fa  = nib.load(str(dti_dir / "tensor" / "FA.nii.gz"))
        assert fa.shape == dwi.shape[:3]

    def test_report_contains_all_metrics(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        for attr in ["fa_mean", "fa_std", "md_mean", "md_std",
                     "ad_mean", "ad_std", "rd_mean", "rd_std"]:
            assert getattr(rec, attr) is not None, f"Missing metric: {attr}"


# ---------------------------------------------------------------------------
# TestNaNHandling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_nan_count_in_record(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert isinstance(rec.nan_count, int)
        assert isinstance(rec.inf_count, int)

    def test_nan_count_non_negative(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.nan_count >= 0
        assert rec.inf_count >= 0


# ---------------------------------------------------------------------------
# TestSubjectFailureIsolation
# ---------------------------------------------------------------------------

class TestSubjectFailureIsolation:
    def test_one_bad_subject_does_not_stop_others(self, tmp_path):
        good1 = _make_dti_dir(tmp_path, subject="10001")
        good2 = _make_dti_dir(tmp_path, subject="10002")
        bad   = _make_dti_dir(tmp_path, subject="10003")
        (bad / "dwi_preprocessed.nii.gz").unlink()  # break this one

        records = run_tensor_batch(
            processed_v2_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            sites=["BNI"],
            skip_if_exists=False,
        )
        statuses = {r.subject_id: r.status for r in records}
        assert statuses.get("10001") == "success"
        assert statuses.get("10002") == "success"
        assert statuses.get("10003") == "failed"


# ---------------------------------------------------------------------------
# TestSkipIfExists
# ---------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_if_exists_true_does_not_refit(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        # First run
        rec1 = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            skip_if_exists=True,
        )
        # Corrupt the DWI to prove it won't re-read it
        (dti_dir / "dwi_preprocessed.nii.gz").write_bytes(b"corrupt")
        rec2 = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            skip_if_exists=True,
        )
        assert rec2.status == "success"
        assert rec2.fa_mean == rec1.fa_mean

    def test_skip_if_exists_false_reruns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            skip_if_exists=False,
        )
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()
        rec2 = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            skip_if_exists=False,
        )
        assert rec2.status == "failed"


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_subject_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        out = tmp_path / "summary.csv"
        write_subject_summary([rec], out)

        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SUBJECT_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_site_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        out = tmp_path / "site_summary.csv"
        write_site_summary([rec], {"BNI": 58}, out)

        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SITE_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_comparison_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        out = tmp_path / "comparison.csv"
        write_v1_comparison([rec], tmp_path / "v1", out)

        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in COMPARISON_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"


# ---------------------------------------------------------------------------
# TestV1Comparison
# ---------------------------------------------------------------------------

class TestV1Comparison:
    def test_no_v1_data_gives_no_v1_data_magnitude(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        out = tmp_path / "comparison.csv"
        write_v1_comparison([rec], tmp_path / "nonexistent_v1", out)

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["fa_change_magnitude"] == "no_v1_data"

    def test_with_v1_data_computes_delta(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, subject="29006")
        rec = estimate_tensors_subject(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        # Write a fake v1 tensor report
        v1_tensor_dir = (
            tmp_path / "v1" / "bni" / "ABIDEII-BNI_1" / "29006"
            / "session_1" / "dti_1" / "tensor"
        )
        v1_tensor_dir.mkdir(parents=True, exist_ok=True)
        (v1_tensor_dir / "tensor_fit_report.json").write_text(json.dumps({
            "fa_mean": 0.50, "fa_std": 0.20,
            "md_mean": 0.0008, "ad_mean": 0.0012, "rd_mean": 0.0006,
            "status": "success",
        }))

        out = tmp_path / "comparison.csv"
        write_v1_comparison([rec], tmp_path / "v1", out)

        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["v1_fa_mean"] == "0.5"
        assert row["delta_fa"] != ""
        assert row["fa_change_magnitude"] in (
            "minimal_change", "moderate_change", "large_change"
        )
