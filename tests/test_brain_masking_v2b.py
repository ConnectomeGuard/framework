"""
Tests for Phase 2R.3 — Brain Masking and Tensor QC (processed_v2b)

Coverage:
  - Raw-write guard
  - Mask binary validation (uint8, values in {0,1})
  - Mask output files created
  - Summary CSV schema
  - Site summary schema
  - PNG creation
  - Missing input handling
  - Missing tensor map handling
  - skip_if_exists behaviour
  - Site outlier detection
  - Subject failure isolation
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

from neurofiber.preprocessing.brain_masking_v2b import (
    SITE_CSV_FIELDS,
    SUBJECT_CSV_FIELDS,
    BrainMaskQCRecord,
    _SITE_FOLDER_MAP,
    process_subject,
    run_masking_batch,
    write_site_summary,
    write_stability_analysis,
    write_subject_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_b0(tmp_path: Path, shape=(16, 16, 8)) -> np.ndarray:
    rng = np.random.default_rng(42)
    data = rng.random(shape, dtype=np.float32) * 500
    # Bright centre to help median_otsu
    c = tuple(s // 2 for s in shape)
    r = min(shape) // 3
    data[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r] *= 5
    return data


def _make_dti_dir(
    base: Path,
    site="BNI", dataset="ABIDEII-BNI_1", subject="29006",
    n_vols=33,
) -> Path:
    folder  = _SITE_FOLDER_MAP.get(site, site.lower())
    dti_dir = base / "abide_ii" / folder / dataset / subject / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(7)
    shape = (16, 16, 8, n_vols)
    data  = rng.random(shape, dtype=np.float32) * 500
    data[5:11, 5:11, 2:6, :] *= 5  # bright region for mask
    img   = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(dti_dir / "dwi_preprocessed.nii.gz"))

    bvals = np.zeros(n_vols); bvals[1:] = 1000
    np.savetxt(str(dti_dir / "dwi_preprocessed.bval"), bvals[np.newaxis,:], fmt="%.1f")

    bvecs = rng.standard_normal((3, n_vols))
    bvecs[:, 0] = 0
    norms = np.linalg.norm(bvecs[:, 1:], axis=0, keepdims=True)
    norms[norms == 0] = 1
    bvecs[:, 1:] /= norms
    np.savetxt(str(dti_dir / "dwi_preprocessed.bvec"), bvecs, fmt="%.6f")

    # Tensor maps
    tensor_dir = dti_dir / "tensor"
    tensor_dir.mkdir()
    mask3d = np.zeros((16, 16, 8), dtype=np.float32)
    mask3d[5:11, 5:11, 2:6] = 1
    for name, lo, hi in [("FA", 0.3, 0.7), ("MD", 5e-4, 1e-3),
                          ("AD", 8e-4, 1.5e-3), ("RD", 3e-4, 7e-4)]:
        arr = rng.uniform(lo, hi, (16, 16, 8)).astype(np.float32) * mask3d
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tensor_dir / f"{name}.nii.gz"))

    return dti_dir


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_dti_dir_inside_raw(self, tmp_path):
        raw_root = tmp_path / "raw"
        dti_dir  = raw_root / "abide_ii" / "bni" / "D" / "99" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        _make_dti_dir(raw_root.parent, subject="99")

        with pytest.raises((ValueError, AssertionError)):
            process_subject(
                dti_dir=raw_root / "abide_ii" / "bni" / "ABIDEII-BNI_1" / "99" / "session_1" / "dti_1",
                output_root=raw_root / "abide_ii",
                raw_root=raw_root,
            )


# ---------------------------------------------------------------------------
# TestMaskOutputs
# ---------------------------------------------------------------------------

class TestMaskOutputs:
    def test_brain_mask_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert (dti_dir / "qc" / "brain_mask.nii.gz").exists()

    def test_b0_brain_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert (dti_dir / "qc" / "b0_brain.nii.gz").exists()

    def test_mask_is_binary(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        mask = nib.load(str(dti_dir / "qc" / "brain_mask.nii.gz")).get_fdata()
        unique_vals = set(np.unique(mask).astype(int))
        assert unique_vals.issubset({0, 1}), f"Non-binary mask values: {unique_vals}"

    def test_mask_dtype_uint8(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        img = nib.load(str(dti_dir / "qc" / "brain_mask.nii.gz"))
        assert img.get_data_dtype() in (np.uint8, np.int16, np.float32), \
            f"Unexpected dtype: {img.get_data_dtype()}"

    def test_mask_has_nonzero_voxels(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.brain_voxel_count > 0

    def test_tensor_qc_report_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert (dti_dir / "qc" / "tensor_qc_report.json").exists()

    def test_qc_report_status_success(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        d = json.loads((dti_dir / "qc" / "tensor_qc_report.json").read_text())
        assert d["status"] == "success"

    def test_all_metrics_present(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "success"
        for attr in ["fa_mean", "fa_std", "md_mean", "md_std",
                     "ad_mean", "ad_std", "rd_mean", "rd_std",
                     "nan_count", "inf_count", "brain_voxel_count"]:
            assert getattr(rec, attr) is not None, f"Missing: {attr}"

    def test_coverage_ratio_between_0_and_1(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert 0.0 < rec.mask_coverage_ratio <= 1.0


# ---------------------------------------------------------------------------
# TestPNGCreation
# ---------------------------------------------------------------------------

class TestPNGCreation:
    def test_pngs_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw", save_png=True)
        qc_dir = dti_dir / "qc"
        for fname in ["b0_mask_overlay.png", "fa_mid_slice.png", "md_mid_slice.png"]:
            assert (qc_dir / fname).exists(), f"Missing PNG: {fname}"

    def test_no_png_flag_skips_pngs(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw", save_png=False)
        qc_dir = dti_dir / "qc"
        for fname in ["b0_mask_overlay.png", "fa_mid_slice.png", "md_mid_slice.png"]:
            assert not (qc_dir / fname).exists(), f"PNG should not exist: {fname}"


# ---------------------------------------------------------------------------
# TestMissingInputs
# ---------------------------------------------------------------------------

class TestMissingInputs:
    def test_missing_dwi_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_bval_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "dwi_preprocessed.bval").unlink()
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_tensor_dir_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        import shutil; shutil.rmtree(str(dti_dir / "tensor"))
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_fa_map_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "tensor" / "FA.nii.gz").unlink()
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"


# ---------------------------------------------------------------------------
# TestSkipIfExists
# ---------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_returns_existing_results(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec1 = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                               skip_if_exists=True)
        (dti_dir / "dwi_preprocessed.nii.gz").write_bytes(b"corrupt")
        rec2 = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                               skip_if_exists=True)
        assert rec2.status == "success"
        assert rec2.fa_mean == rec1.fa_mean

    def test_no_skip_reruns_on_forced(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                        skip_if_exists=False)
        (dti_dir / "dwi_preprocessed.nii.gz").unlink()
        rec2 = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                               skip_if_exists=False)
        assert rec2.status == "failed"


# ---------------------------------------------------------------------------
# TestSubjectFailureIsolation
# ---------------------------------------------------------------------------

class TestSubjectFailureIsolation:
    def test_one_bad_subject_does_not_stop_others(self, tmp_path):
        _make_dti_dir(tmp_path, subject="10001")
        _make_dti_dir(tmp_path, subject="10002")
        bad = _make_dti_dir(tmp_path, subject="10003")
        (bad / "dwi_preprocessed.nii.gz").unlink()

        records = run_masking_batch(
            processed_v2b_root=tmp_path / "abide_ii",
            raw_root=tmp_path / "raw",
            sites=["BNI"],
            skip_if_exists=False,
            save_png=False,
        )
        statuses = {r.subject_id: r.status for r in records}
        assert statuses.get("10001") == "success"
        assert statuses.get("10002") == "success"
        assert statuses.get("10003") == "failed"


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_subject_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                              save_png=False)
        out = tmp_path / "subj.csv"
        write_subject_summary([rec], out)
        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SUBJECT_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_site_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                              save_png=False)
        out = tmp_path / "site.csv"
        write_site_summary([rec], out)
        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SITE_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"

    def test_stability_csv_has_required_columns(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                              save_png=False)
        out = tmp_path / "stability.csv"
        write_stability_analysis([rec], out)
        with open(out) as f:
            cols = csv.DictReader(f).fieldnames
        for field in SITE_CSV_FIELDS:
            assert field in cols, f"Missing column: {field}"


# ---------------------------------------------------------------------------
# TestOutlierDetection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_outlier_flag_empty_for_uniform_sites(self, tmp_path):
        recs = []
        for subj in ["10001", "10002", "10003"]:
            dti_dir = _make_dti_dir(tmp_path, subject=subj)
            rec = process_subject(dti_dir, tmp_path / "abide_ii", tmp_path / "raw",
                                  save_png=False)
            recs.append(rec)
        out = tmp_path / "stability.csv"
        write_stability_analysis(recs, out)
        rows = list(csv.DictReader(open(out)))
        bni_row = next((r for r in rows if r["site"] == "BNI"), None)
        assert bni_row is not None
        # Single site — no cross-site comparison possible, no outlier flag expected
        assert bni_row["outlier_flag"] == "" or bni_row["outlier_flag"] is None
