"""Tests for Phase 2.2 — DTI tensor estimation."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dipy.core.gradients import gradient_table
from dipy.sims.voxel import single_tensor

from neurofiber.preprocessing.tensor_estimation import (
    TensorEstimationRecord,
    estimate_tensors_subject,
    run_tensor_batch,
    B0_THRESHOLD,
    MIN_DWI_DIRECTIONS,
)


# ---------------------------------------------------------------------------
# Synthetic DTI data generator
# ---------------------------------------------------------------------------

def _make_synthetic_dti(
    tmp_path: Path,
    subject_id: str = "sub-001",
    n_dirs: int = 32,
    spatial: tuple = (8, 8, 6),
    evals: list | None = None,
) -> tuple[Path, Path, Path]:
    """
    Create a minimal but physically plausible synthetic DTI dataset.
    Returns (dti_path, bval_path, bvec_path).
    """
    if evals is None:
        evals = [0.0015, 0.0003, 0.0003]   # typical white-matter eigenvalues

    n_vols = n_dirs + 1   # 1 B0 + n_dirs DWI

    # Build gradient directions (uniform on sphere)
    rng = np.random.default_rng(42)
    dirs = rng.standard_normal((n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    bvals = np.array([0.0] + [2500.0] * n_dirs)
    bvecs = np.vstack([np.zeros((1, 3)), dirs])   # (N, 3)

    gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESHOLD)

    # Simulate signal voxel-by-voxel (outermost ring = zero/background)
    signal = np.zeros((*spatial, n_vols), dtype=np.float32)
    for i in range(spatial[0]):
        for j in range(spatial[1]):
            for k in range(spatial[2]):
                # Leave border voxels as zero to mimic background
                if i in (0, spatial[0]-1) or j in (0, spatial[1]-1) or k in (0, spatial[2]-1):
                    continue
                s = single_tensor(gtab, S0=1000.0, evals=evals, snr=None)
                signal[i, j, k] = s.astype(np.float32)

    # Subject directory structure
    dti_dir = tmp_path / subject_id / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True)

    dti_path  = dti_dir / "dti_corrected.nii.gz"
    bval_path = dti_dir / "dti.bval"
    bvec_path = dti_dir / "dti.bvec"

    nib.save(nib.Nifti1Image(signal, np.eye(4)), str(dti_path))
    # bvec in FSL convention: (3, N)
    np.savetxt(str(bval_path), bvals.reshape(1, -1), fmt="%g")
    np.savetxt(str(bvec_path), bvecs.T, fmt="%.6f")   # (3, N)

    return dti_path, bval_path, bvec_path


# ---------------------------------------------------------------------------
# estimate_tensors_subject — core tests
# ---------------------------------------------------------------------------

class TestEstimateTensorsSubject:

    def test_creates_all_four_maps(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        tensor_dir = dti.parent / "tensor"
        for name in ("FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz"):
            assert (tensor_dir / name).exists(), f"{name} not created"

    def test_output_shapes_match_spatial(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path, spatial=(8, 8, 6))
        estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        fa_img = nib.load(str(dti.parent / "tensor" / "FA.nii.gz"))
        assert fa_img.shape == (8, 8, 6)

    def test_fa_values_in_valid_range(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        fa = nib.load(str(dti.parent / "tensor" / "FA.nii.gz")).get_fdata()
        assert fa.min() >= 0.0, "FA below 0"
        assert fa.max() <= 1.0, "FA above 1"

    def test_fa_mean_is_plausible(self, tmp_path):
        """Single-tensor WM-like signal should give FA ≈ 0.7 in brain voxels."""
        dti, bval, bvec = _make_synthetic_dti(tmp_path,
                                               evals=[0.0015, 0.0003, 0.0003])
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "success"
        assert rec.fa_mean is not None
        assert 0.3 < rec.fa_mean < 1.0, f"Unexpected FA mean: {rec.fa_mean}"

    def test_record_status_success(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "success"
        assert rec.error_message is None

    def test_volume_counts_in_record(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path, n_dirs=32)
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.input_volume_count == 33
        assert rec.bval_count == 33
        assert rec.bvec_count == 33
        assert rec.b0_count == 1
        assert rec.dwi_count == 32

    def test_report_json_written(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        report_path = dti.parent / "tensor" / "tensor_fit_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["status"] == "success"
        assert "fa_mean" in report
        assert "input_dti_path" in report

    def test_missing_dti_returns_failed(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        dti.unlink()
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "failed"
        assert rec.error_message is not None

    def test_count_mismatch_returns_failed(self, tmp_path):
        """bval with wrong number of entries should fail validation."""
        dti, bval, bvec = _make_synthetic_dti(tmp_path, n_dirs=32)
        # Overwrite bval with 30 entries instead of 33
        bad_bvals = np.array([0.0] + [2500.0] * 29)
        np.savetxt(str(bval), bad_bvals.reshape(1, -1), fmt="%g")
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "failed"

    def test_too_few_directions_returns_failed(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path, n_dirs=20)
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "failed"
        assert str(MIN_DWI_DIRECTIONS) in rec.error_message

    def test_no_b0_returns_failed(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path, n_dirs=32)
        # Replace all bvals with 2500 (no b0)
        all_dwi = np.full(33, 2500.0)
        np.savetxt(str(bval), all_dwi.reshape(1, -1), fmt="%g")
        rec = estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert rec.status == "failed"

    def test_raw_write_guard_raises(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        raw.mkdir(parents=True)
        with pytest.raises(ValueError, match="SAFETY"):
            estimate_tensors_subject(
                dti, bval, bvec,
                output_dir=raw / "some_subject",
                raw_root=raw,
            )

    def test_raw_data_not_modified(self, tmp_path):
        dti, bval, bvec = _make_synthetic_dti(tmp_path)
        mtime_before = dti.stat().st_mtime
        estimate_tensors_subject(dti, bval, bvec, output_dir=dti.parent)
        assert dti.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# run_tensor_batch
# ---------------------------------------------------------------------------

class TestRunTensorBatch:

    def test_batch_processes_all_subjects(self, tmp_path):
        for sid in ["sub-001", "sub-002", "sub-003"]:
            _make_synthetic_dti(tmp_path, subject_id=sid)
        records = run_tensor_batch(tmp_path)
        assert len(records) == 3
        assert all(r.status == "success" for r in records)

    def test_batch_missing_files_returns_failed(self, tmp_path):
        # Create subject dir with no files
        (tmp_path / "sub-empty" / "session_1" / "dti_1").mkdir(parents=True)
        records = run_tensor_batch(tmp_path)
        assert len(records) == 1
        assert records[0].status == "failed"

    def test_batch_raw_guard_on_root(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        with pytest.raises(ValueError, match="SAFETY"):
            run_tensor_batch(raw, raw_root=raw)

    def test_to_summary_row_has_all_required_cols(self, tmp_path):
        _make_synthetic_dti(tmp_path, subject_id="sub-001")
        records = run_tensor_batch(tmp_path)
        row = records[0].to_summary_row()
        required_cols = {
            "subject_id", "input_volume_count", "bval_count", "bvec_count",
            "b0_count", "dwi_count", "fa_mean", "fa_std",
            "md_mean", "rd_mean", "ad_mean", "status", "error_message",
        }
        assert required_cols == set(row.keys())
