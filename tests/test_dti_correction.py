"""Tests for Phase 2.1 — BNI DTI Volume Correction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofiber.preprocessing.dti_correction import (
    correct_dti_volumes,
    run_correction_batch,
    CorrectionRecord,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_subject_tree(
    root: Path,
    subject_id: str,
    n_vols: int = 34,
    n_bvals: int = 33,
    b0_mean: float = 100.0,
    last_mean: float = 5.0,
) -> tuple[Path, Path, Path]:
    """
    Create a minimal ABIDE II BNI subject tree and return (dti, bval, bvec) paths.
    The first volume has mean signal b0_mean; the last has last_mean; others are flat.
    """
    dti_dir = root / subject_id / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True)

    # Build data: last volume at last_mean, vol 0 at b0_mean, rest at 50
    data = np.full((8, 8, 4, n_vols), 50.0, dtype=np.float32)
    data[..., 0]  = b0_mean
    data[..., -1] = last_mean
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dti_dir / "dti.nii.gz"))

    bvals = np.zeros(n_bvals)
    bvals[1:] = 2500
    np.savetxt(str(dti_dir / "dti.bval"), bvals.reshape(1, -1), fmt="%g")

    bvecs = np.zeros((3, n_bvals))
    bvecs[0, 1:] = 1.0
    np.savetxt(str(dti_dir / "dti.bvec"), bvecs, fmt="%.4f")

    return dti_dir / "dti.nii.gz", dti_dir / "dti.bval", dti_dir / "dti.bvec"


# ---------------------------------------------------------------------------
# correct_dti_volumes — single subject
# ---------------------------------------------------------------------------

class TestCorrectDtiVolumes:

    def test_volume_removed_when_ratio_low(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001", b0_mean=100, last_mean=5)
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        rec = correct_dti_volumes(dti, bval, bvec, output_dir=out)

        assert rec.action == "volume_removed"
        assert rec.status == "success"
        assert rec.original_volume_count == 34
        assert rec.corrected_volume_count == 33
        assert (out / "dti_corrected.nii.gz").exists()
        assert (out / "dti.bval").exists()
        assert (out / "dti.bvec").exists()

    def test_corrected_nifti_has_33_volumes(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001")
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        correct_dti_volumes(dti, bval, bvec, output_dir=out)

        corrected = nib.load(str(out / "dti_corrected.nii.gz"))
        assert corrected.shape[3] == 33

    def test_bval_bvec_unchanged_after_correction(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001")
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        correct_dti_volumes(dti, bval, bvec, output_dir=out)

        orig_bval = np.loadtxt(str(bval))
        copied_bval = np.loadtxt(str(out / "dti.bval"))
        np.testing.assert_array_equal(orig_bval, copied_bval)

    def test_requires_manual_review_when_ratio_high(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001", b0_mean=100, last_mean=50)
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        rec = correct_dti_volumes(dti, bval, bvec, output_dir=out)

        assert rec.action == "requires_manual_review"
        assert rec.status == "success"
        assert rec.corrected_volume_count is None
        assert not (out / "dti_corrected.nii.gz").exists()

    def test_correction_report_written(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001")
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        correct_dti_volumes(dti, bval, bvec, output_dir=out)

        report = json.loads((out / "correction_report.json").read_text())
        assert report["subject_id"] == "sub-001"
        assert report["action"] == "volume_removed"
        assert report["original_volume_count"] == 34
        assert report["corrected_volume_count"] == 33
        assert "timestamp" in report

    def test_signal_ratio_computed_correctly(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001", b0_mean=80.0, last_mean=8.0)
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        rec = correct_dti_volumes(dti, bval, bvec, output_dir=out)

        assert abs(rec.signal_ratio - 0.1) < 0.01

    def test_custom_threshold_respected(self, tmp_path):
        """Ratio of 0.25 should pass threshold=0.3 but fail threshold=0.2."""
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001", b0_mean=100, last_mean=25)
        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"

        rec_strict = correct_dti_volumes(dti, bval, bvec, output_dir=out, signal_ratio_threshold=0.2)
        assert rec_strict.action == "requires_manual_review"

        out2 = tmp_path / "processed2" / "sub-001" / "session_1" / "dti_1"
        rec_lenient = correct_dti_volumes(dti, bval, bvec, output_dir=out2, signal_ratio_threshold=0.3)
        assert rec_lenient.action == "volume_removed"

    def test_3d_nifti_returns_failed(self, tmp_path):
        raw = tmp_path / "raw"
        dti_dir = raw / "sub-bad" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), np.float32), np.eye(4)),
                 str(dti_dir / "dti.nii.gz"))
        np.savetxt(str(dti_dir / "dti.bval"), np.zeros((1, 5)), fmt="%g")
        np.savetxt(str(dti_dir / "dti.bvec"), np.zeros((3, 5)), fmt="%g")

        out = tmp_path / "processed" / "sub-bad" / "session_1" / "dti_1"
        rec = correct_dti_volumes(
            dti_dir / "dti.nii.gz", dti_dir / "dti.bval", dti_dir / "dti.bvec",
            output_dir=out,
        )
        assert rec.action == "failed"
        assert rec.status == "failed"
        assert rec.error_message is not None

    def test_volume_diff_greater_than_1_returns_failed(self, tmp_path):
        raw = tmp_path / "raw"
        # 34 volumes, 30 bvals — diff of 4, cannot auto-correct
        dti, bval, bvec = _make_subject_tree(raw, "sub-bad", n_vols=34, n_bvals=30)
        out = tmp_path / "processed" / "sub-bad" / "session_1" / "dti_1"

        rec = correct_dti_volumes(dti, bval, bvec, output_dir=out)
        assert rec.action == "failed"

    def test_consistent_data_copies_as_corrected(self, tmp_path):
        """Sites with V=B=G should copy the original DTI as dti_corrected (no-op pass-through)."""
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-ok", n_vols=33, n_bvals=33)
        out = tmp_path / "processed" / "sub-ok" / "session_1" / "dti_1"

        rec = correct_dti_volumes(dti, bval, bvec, output_dir=out)
        assert rec.action == "no_correction_needed"
        assert rec.status == "success"
        assert rec.corrected_volume_count == 33
        assert (out / "dti_corrected.nii.gz").exists()
        assert (out / "dti.bval").exists()

    def test_raw_data_not_modified(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001")
        original_dti_mtime = dti.stat().st_mtime
        original_bval_mtime = bval.stat().st_mtime

        out = tmp_path / "processed" / "sub-001" / "session_1" / "dti_1"
        correct_dti_volumes(dti, bval, bvec, output_dir=out)

        assert dti.stat().st_mtime == original_dti_mtime
        assert bval.stat().st_mtime == original_bval_mtime

    def test_output_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        dti, bval, bvec = _make_subject_tree(raw, "sub-001")
        bad_output = dti.parent  # output INSIDE the raw tree

        with pytest.raises(ValueError, match="SAFETY"):
            correct_dti_volumes(dti, bval, bvec, output_dir=bad_output)


# ---------------------------------------------------------------------------
# run_correction_batch
# ---------------------------------------------------------------------------

class TestRunCorrectionBatch:

    def test_batch_processes_all_subjects(self, tmp_path):
        raw = tmp_path / "raw"
        for sid in ["sub-001", "sub-002", "sub-003"]:
            _make_subject_tree(raw, sid)

        records = run_correction_batch(raw, tmp_path / "processed")
        assert len(records) == 3

    def test_batch_all_volume_removed(self, tmp_path):
        raw = tmp_path / "raw"
        for sid in ["sub-001", "sub-002"]:
            _make_subject_tree(raw, sid, b0_mean=100, last_mean=5)

        records = run_correction_batch(raw, tmp_path / "processed")
        assert all(r.action == "volume_removed" for r in records)

    def test_batch_output_root_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        _make_subject_tree(raw, "sub-001")

        with pytest.raises(ValueError, match="SAFETY"):
            run_correction_batch(raw, raw / "processed")

    def test_batch_missing_files_returns_failed(self, tmp_path):
        raw = tmp_path / "raw"
        # Create subject dir without any files
        (raw / "sub-empty" / "session_1" / "dti_1").mkdir(parents=True)

        records = run_correction_batch(raw, tmp_path / "processed")
        assert records[0].action == "failed"
        assert records[0].status == "failed"

    def test_to_summary_row_has_all_fields(self, tmp_path):
        raw = tmp_path / "raw"
        _make_subject_tree(raw, "sub-001")
        records = run_correction_batch(raw, tmp_path / "processed")
        row = records[0].to_summary_row()

        expected_keys = {
            "subject_id", "original_volume_count", "corrected_volume_count",
            "bval_count", "bvec_count", "b0_mean_signal", "last_volume_mean_signal",
            "signal_ratio", "action", "status",
        }
        assert expected_keys == set(row.keys())
