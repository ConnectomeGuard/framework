"""
Tests for Phase 3R.1 — FOD / Peak Direction Preparation from processed_v2b

Coverage:
  - Clean site filtering (only BNI, NYU_1, NYU_2, SDSU_1, TCD_1)
  - IP_1 explicit exclusion (no output generated)
  - Required input validation (missing DWI, bval, bvec, mask, QC report)
  - FA gate (subjects outside [0.15, 0.40] are skipped)
  - Output path generation (peaks.pam5, report, backend_used.txt)
  - Summary CSV schema (subject and site)
  - Raw-write guard
  - skip_if_exists behaviour
  - Batch runner excludes IP_1 by default and raises on explicit IP_1 request
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tractography.fod_preparation_v2b import (
    CLEAN_SITES_DISPLAY,
    EXCLUDED_SITES_DISPLAY,
    SITE_CSV_FIELDS,
    SUBJECT_CSV_FIELDS,
    _SITE_FOLDER_MAP,
    prepare_subject_fod,
    run_fod_preparation_batch,
    write_site_summary,
    write_subject_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dti_dir(
    base: Path,
    site: str = "BNI",
    dataset: str = "ABIDEII-BNI_1",
    subject: str = "29006",
    n_vols: int = 33,
    fa_mean: float = 0.25,
    md_mean: float = 0.00085,
) -> Path:
    folder  = _SITE_FOLDER_MAP.get(site, site.lower())
    dti_dir = base / "abide_ii" / folder / dataset / subject / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    shape = (12, 12, 6, n_vols)
    data  = rng.random(shape, dtype=np.float32) * 800
    data[3:9, 3:9, 1:5, :] *= 4
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dti_dir / "dwi_preprocessed.nii.gz"))

    bvals = np.zeros(n_vols); bvals[1:] = 1000
    np.savetxt(str(dti_dir / "dwi_preprocessed.bval"), bvals[np.newaxis,:], fmt="%.1f")

    bvecs = rng.standard_normal((3, n_vols))
    bvecs[:, 0] = 0
    norms = np.linalg.norm(bvecs[:, 1:], axis=0, keepdims=True)
    norms[norms == 0] = 1
    bvecs[:, 1:] /= norms
    np.savetxt(str(dti_dir / "dwi_preprocessed.bvec"), bvecs, fmt="%.6f")

    # QC dir — brain mask and tensor_qc_report
    qc_dir = dti_dir / "qc"
    qc_dir.mkdir()
    mask = np.zeros((12, 12, 6), dtype=np.uint8)
    mask[3:9, 3:9, 1:5] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(qc_dir / "brain_mask.nii.gz"))
    (qc_dir / "tensor_qc_report.json").write_text(json.dumps({
        "status": "success",
        "fa_mean": fa_mean,
        "md_mean": md_mean,
    }))

    # Tensor maps
    tensor_dir = dti_dir / "tensor"
    tensor_dir.mkdir()
    for name in ["FA", "MD", "AD", "RD"]:
        arr = rng.random((12, 12, 6), dtype=np.float32) * mask.astype(np.float32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tensor_dir / f"{name}.nii.gz"))

    return dti_dir


# ---------------------------------------------------------------------------
# TestCleanSiteFiltering
# ---------------------------------------------------------------------------

class TestCleanSiteFiltering:
    def test_clean_sites_list_correct(self):
        assert "BNI" in CLEAN_SITES_DISPLAY
        assert "NYU_1" in CLEAN_SITES_DISPLAY
        assert "NYU_2" in CLEAN_SITES_DISPLAY
        assert "SDSU_1" in CLEAN_SITES_DISPLAY
        assert "TCD_1" in CLEAN_SITES_DISPLAY

    def test_ip1_in_excluded_list(self):
        assert "IP_1" in EXCLUDED_SITES_DISPLAY

    def test_ip1_not_in_clean_list(self):
        assert "IP_1" not in CLEAN_SITES_DISPLAY

    def test_batch_raises_if_ip1_requested(self, tmp_path):
        with pytest.raises(ValueError, match="excluded"):
            run_fod_preparation_batch(
                processed_v2b_root=tmp_path / "abide_ii",
                raw_root=tmp_path / "raw",
                sites=["IP_1"],
            )


# ---------------------------------------------------------------------------
# TestIP1Exclusion
# ---------------------------------------------------------------------------

class TestIP1Exclusion:
    def test_ip1_subject_returns_excluded_status(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, site="IP_1", dataset="ABIDEII-IP_1",
                                subject="29600")
        rec = prepare_subject_fod(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "excluded"

    def test_ip1_no_peaks_file_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, site="IP_1", dataset="ABIDEII-IP_1",
                                subject="29600")
        prepare_subject_fod(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert not (dti_dir / "fod" / "peaks.pam5").exists()

    def test_ip1_no_report_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, site="IP_1", dataset="ABIDEII-IP_1",
                                subject="29600")
        prepare_subject_fod(
            dti_dir=dti_dir,
            output_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert not (dti_dir / "fod" / "fod_prep_report.json").exists()


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_dwi_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_bval_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.bval").unlink()
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_bvec_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.bvec").unlink()
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_mask_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "qc" / "brain_mask.nii.gz").unlink()
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_qc_report_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "qc" / "tensor_qc_report.json").unlink()
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"


# ---------------------------------------------------------------------------
# TestFAGate
# ---------------------------------------------------------------------------

class TestFAGate:
    def test_low_fa_subject_skipped(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, fa_mean=0.05)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "skipped"
        assert "FA" in (rec.warning_message or "")

    def test_high_fa_subject_skipped(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, fa_mean=0.70)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "skipped"

    def test_clean_fa_subject_processed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, fa_mean=0.25)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "success"


# ---------------------------------------------------------------------------
# TestOutputs
# ---------------------------------------------------------------------------

class TestOutputs:
    def test_peaks_pam5_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "success"
        assert (dti_dir / "fod" / "peaks.pam5").exists()

    def test_report_json_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert (dti_dir / "fod" / "fod_prep_report.json").exists()

    def test_backend_used_txt_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        backend_file = dti_dir / "fod" / "backend_used.txt"
        assert backend_file.exists()
        assert "dipy_csa" in backend_file.read_text()

    def test_report_has_success_status(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        d = json.loads((dti_dir / "fod" / "fod_prep_report.json").read_text())
        assert d["status"] == "success"

    def test_md_mean_in_record(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, md_mean=0.00085)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.md_mean == pytest.approx(0.00085, rel=1e-3)


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_inside_raw(self, tmp_path):
        raw_root = tmp_path / "raw"
        dti_dir  = raw_root / "abide_ii" / "bni" / "D" / "99" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        with pytest.raises((ValueError, AssertionError)):
            prepare_subject_fod(
                dti_dir=dti_dir,
                output_root=raw_root / "abide_ii",
                raw_root=raw_root,
            )


# ---------------------------------------------------------------------------
# TestSkipIfExists
# ---------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_reuses_existing_report(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec1 = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                                   skip_if_exists=True)
        (dti_dir / "dwi_preprocessed.nii.gz").write_bytes(b"corrupt")
        rec2 = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                                   skip_if_exists=True)
        assert rec2.status == "success"

    def test_no_skip_reruns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                            skip_if_exists=False)
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()
        rec2 = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                                   skip_if_exists=False)
        assert rec2.status == "failed"


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_subject_csv_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        out = tmp_path / "subj.csv"
        write_subject_summary([rec], out)
        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SUBJECT_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_site_csv_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = prepare_subject_fod(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        out = tmp_path / "site.csv"
        write_site_summary([rec], {"BNI": 58}, out)
        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SITE_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"
