"""Tests for Phase 2.3 — DTI brain masking and tensor QC."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor

from neurofiber.preprocessing.brain_masking import (
    BrainMaskQCRecord,
    _compute_qc,
    _run_median_otsu,
    process_subject,
    run_masking_batch,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_subject_tree(
    root: Path,
    sid: str,
    spatial: tuple = (12, 12, 10),
    n_dirs: int = 32,
) -> tuple[Path, Path, Path]:
    """
    Create a synthetic subject tree with realistic DTI signal and tensor maps.
    Returns (dti_path, bval_path, tensor_dir).
    """
    rng = np.random.default_rng(0)
    n_vols = n_dirs + 1

    dirs = rng.standard_normal((n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals = np.array([0.0] + [2500.0] * n_dirs)
    bvecs = np.vstack([np.zeros((1, 3)), dirs])
    gtab  = gradient_table(bvals, bvecs, b0_threshold=100)

    # Signal: interior voxels get WM-like tensor, border = 0 (background)
    data = np.zeros((*spatial, n_vols), dtype=np.float32)
    for i in range(1, spatial[0]-1):
        for j in range(1, spatial[1]-1):
            for k in range(1, spatial[2]-1):
                s = single_tensor(gtab, S0=500.0, evals=[0.0015, 0.0003, 0.0003])
                data[i, j, k] = s.astype(np.float32)

    dti_dir = root / sid / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True)

    dti_path  = dti_dir / "dti_corrected.nii.gz"
    bval_path = dti_dir / "dti.bval"
    bvec_path = dti_dir / "dti.bvec"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dti_path))
    np.savetxt(str(bval_path), bvals.reshape(1, -1), fmt="%g")
    np.savetxt(str(bvec_path), bvecs.T, fmt="%.6f")

    # Synthetic tensor maps (constant within interior)
    tensor_dir = dti_dir / "tensor"
    tensor_dir.mkdir()
    interior = np.zeros(spatial, dtype=np.float32)
    interior[1:-1, 1:-1, 1:-1] = 1.0   # mark interior voxels

    fa_data  = interior * 0.65
    md_data  = interior * 0.0012
    ad_data  = interior * 0.0020
    rd_data  = interior * 0.0007
    affine   = np.eye(4)
    for name, arr in [("FA", fa_data), ("MD", md_data), ("AD", ad_data), ("RD", rd_data)]:
        nib.save(nib.Nifti1Image(arr, affine), str(tensor_dir / f"{name}.nii.gz"))

    return dti_path, bval_path, tensor_dir


# ---------------------------------------------------------------------------
# _run_median_otsu
# ---------------------------------------------------------------------------

def _make_b0_phantom(shape=(30, 30, 20)) -> np.ndarray:
    """Return a realistic b0 phantom: background=0, sphere-like brain=500."""
    b0 = np.zeros(shape, dtype=np.float32)
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    r = min(shape) // 3
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if (i-cx)**2 + (j-cy)**2 + (k-cz)**2 < r**2:
                    b0[i, j, k] = 500.0
    return b0


class TestRunMedianOtsu:

    def test_returns_bool_mask(self):
        b0 = _make_b0_phantom()
        mask, _ = _run_median_otsu(b0)
        assert mask.dtype == bool

    def test_mask_is_binary(self):
        b0 = _make_b0_phantom()
        mask, _ = _run_median_otsu(b0)
        assert set(np.unique(mask)).issubset({False, True})

    def test_b0_brain_shape_matches_input(self):
        b0 = _make_b0_phantom()
        mask, b0_brain = _run_median_otsu(b0)
        assert b0_brain.shape == b0.shape


# ---------------------------------------------------------------------------
# _compute_qc
# ---------------------------------------------------------------------------

class TestComputeQC:

    def _uniform_maps(self, spatial=(8, 8, 6)):
        n = np.prod(spatial)
        fa = np.full(spatial, 0.35, dtype=np.float32)
        md = np.full(spatial, 0.0009, dtype=np.float32)
        ad = np.full(spatial, 0.0015, dtype=np.float32)
        rd = np.full(spatial, 0.0006, dtype=np.float32)
        mask = np.ones(spatial, dtype=bool)
        return fa, md, ad, rd, mask

    def test_fa_mean_correct(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert abs(rec.fa_mean - 0.35) < 1e-4

    def test_md_mean_correct(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert abs(rec.md_mean - 0.0009) < 1e-7

    def test_no_warnings_for_clean_data(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.warning_message is None

    def test_nan_count_detected(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        fa[0, 0, 0] = float("nan")
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.nan_count >= 1
        assert "NaN" in (rec.warning_message or "")

    def test_suspicious_fa_above_1_flagged(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        fa[1, 1, 1] = 1.5   # FA > 1
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.suspicious_fa_count >= 1

    def test_negative_md_flagged(self):
        fa, md, ad, rd, mask = self._uniform_maps()
        md[2, 2, 2] = -0.001
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.suspicious_diffusivity_count >= 1

    def test_brain_voxel_count_correct(self):
        fa, md, ad, rd, mask = self._uniform_maps(spatial=(4, 4, 4))
        mask[0, :, :] = False   # zero out one slice
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.brain_voxel_count == int(mask.sum())

    def test_high_fa_mean_triggers_warning(self):
        fa = np.full((6, 6, 6), 0.8, dtype=np.float32)   # > _FA_WARN_HIGH
        md = np.full((6, 6, 6), 0.001, dtype=np.float32)
        ad = np.full((6, 6, 6), 0.002, dtype=np.float32)
        rd = np.full((6, 6, 6), 0.0005, dtype=np.float32)
        mask = np.ones((6, 6, 6), dtype=bool)
        rec = _compute_qc("sub-001", fa, md, ad, rd, mask)
        assert rec.warning_message is not None
        assert "FA_mean" in rec.warning_message


# ---------------------------------------------------------------------------
# process_subject
# ---------------------------------------------------------------------------

class TestProcessSubject:

    def test_creates_brain_mask_file(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        rec = process_subject(dti, bval, tensor_dir, output_dir=qc_dir)
        assert (qc_dir / "brain_mask.nii.gz").exists()

    def test_creates_b0_brain_file(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        process_subject(dti, bval, tensor_dir, output_dir=qc_dir)
        assert (qc_dir / "b0_brain.nii.gz").exists()

    def test_mask_is_binary(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        process_subject(dti, bval, tensor_dir, output_dir=qc_dir)
        mask = nib.load(str(qc_dir / "brain_mask.nii.gz")).get_fdata()
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_report_json_written(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        process_subject(dti, bval, tensor_dir, output_dir=qc_dir)
        report = json.loads((qc_dir / "tensor_qc_report.json").read_text())
        assert report["status"] == "success"
        assert "fa_mean" in report
        assert "brain_voxel_count" in report

    def test_fa_within_valid_range_after_masking(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        rec = process_subject(dti, bval, tensor_dir, output_dir=qc_dir)
        assert rec.status == "success"
        assert rec.fa_mean is not None
        assert 0.0 <= rec.fa_mean <= 1.0

    def test_png_files_created(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        process_subject(dti, bval, tensor_dir, output_dir=qc_dir, save_png=True)
        for png in ("b0_mask_overlay.png", "fa_mid_slice.png", "md_mid_slice.png"):
            assert (qc_dir / png).exists(), f"{png} not created"

    def test_missing_tensor_dir_returns_failed(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        qc_dir = dti.parent / "qc"
        rec = process_subject(dti, bval, Path("/nonexistent/tensor"), output_dir=qc_dir)
        assert rec.status == "failed"

    def test_raw_write_guard_raises(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        raw.mkdir(parents=True)
        with pytest.raises(ValueError, match="SAFETY"):
            process_subject(
                dti, bval, tensor_dir,
                output_dir=raw / "sub-001_qc",
                raw_root=raw,
            )

    def test_raw_data_not_modified(self, tmp_path):
        dti, bval, tensor_dir = _make_subject_tree(tmp_path, "sub-001")
        mtime_before = dti.stat().st_mtime
        process_subject(dti, bval, tensor_dir, output_dir=dti.parent / "qc")
        assert dti.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# run_masking_batch
# ---------------------------------------------------------------------------

class TestRunMaskingBatch:

    def test_batch_processes_all_subjects(self, tmp_path):
        for sid in ["sub-001", "sub-002", "sub-003"]:
            _make_subject_tree(tmp_path, sid)
        records = run_masking_batch(tmp_path, save_png=False)
        assert len(records) == 3
        assert all(r.status == "success" for r in records)

    def test_to_summary_row_has_all_columns(self, tmp_path):
        _make_subject_tree(tmp_path, "sub-001")
        records = run_masking_batch(tmp_path, save_png=False)
        row = records[0].to_summary_row()
        required = {
            "subject_id", "brain_voxel_count", "mask_coverage_ratio",
            "fa_mean", "fa_std", "fa_min", "fa_max",
            "md_mean", "md_std", "md_min", "md_max",
            "ad_mean", "rd_mean",
            "nan_count", "inf_count",
            "suspicious_fa_count", "suspicious_diffusivity_count",
            "status", "warning_message",
        }
        assert required == set(row.keys())

    def test_missing_tensor_dir_flagged_as_failed(self, tmp_path):
        sid = "sub-bad"
        dti_dir = tmp_path / sid / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        # Create DTI + bval but no tensor/ dir
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4, 5), np.float32), np.eye(4)),
            str(dti_dir / "dti_corrected.nii.gz"),
        )
        np.savetxt(str(dti_dir / "dti.bval"), np.array([[0., 2500., 2500., 2500., 2500.]]))
        records = run_masking_batch(tmp_path, save_png=False)
        assert records[0].status == "failed"
