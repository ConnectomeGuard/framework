"""Tests for Phase 3.1 — Multi-site FOD / Orientation Preparation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from neurofiber.tractography.fod_preparation import (
    CLEAN_DTI_SITES,
    FA_MAX_CLEAN,
    FA_MIN_CLEAN,
    FODPrepRecord,
    _mrtrix3_available,
    prepare_subject_fod,
    run_fod_preparation_batch,
    save_summary_csvs,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_subject_tree(
    root: Path,
    site: str = "bni",
    dataset: str = "ABIDEII-BNI_1",
    subject_id: str = "sub-001",
    n_vols: int = 33,
    fa_mean: float = 0.25,
) -> Path:
    """
    Create a minimal Phase 2 subject tree under root/<site>/<dataset>/<subject_id>/
    and return the dti_1/ Path.

    The DWI volume is synthetic (8×8×4 × n_vols).
    """
    dti_dir = root / site / dataset / subject_id / "session_1" / "dti_1"
    qc_dir     = dti_dir / "qc"
    tensor_dir = dti_dir / "tensor"
    dti_dir.mkdir(parents=True)
    qc_dir.mkdir()
    tensor_dir.mkdir()

    # DWI NIfTI — small signal with mild variation
    rng  = np.random.default_rng(42)
    data = rng.uniform(50, 100, (8, 8, 4, n_vols)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dti_dir / "dti_corrected.nii.gz"))

    # b-values: first volume is b=0, rest are DWI
    bvals = np.zeros(n_vols)
    bvals[1:] = 2500
    np.savetxt(str(dti_dir / "dti.bval"), bvals.reshape(1, -1), fmt="%g")

    # b-vectors: FSL 3×N convention
    bvecs = np.zeros((3, n_vols))
    # Spread DWI directions on a hemisphere
    n_dwi = n_vols - 1
    for i in range(n_dwi):
        theta = np.pi * i / n_dwi
        phi   = 2 * np.pi * i / n_dwi
        bvecs[:, i + 1] = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    np.savetxt(str(dti_dir / "dti.bvec"), bvecs, fmt="%.6f")

    # Brain mask — central 4×4×2 box
    mask = np.zeros((8, 8, 4), dtype=np.uint8)
    mask[2:6, 2:6, 1:3] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(qc_dir / "brain_mask.nii.gz"))

    # Stub tensor maps
    for name in ("FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz"):
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8, 4), dtype=np.float32), np.eye(4)),
            str(tensor_dir / name),
        )

    # Tensor QC report with the requested fa_mean
    qc_report = {
        "subject_id":  subject_id,
        "fa_mean":     fa_mean,
        "fa_std":      0.1,
        "status":      "success",
        "warning_message": None,
        "timestamp":   "2026-06-03T00:00:00+00:00",
    }
    (qc_dir / "tensor_qc_report.json").write_text(json.dumps(qc_report))

    return dti_dir


# ---------------------------------------------------------------------------
# TestPrepareSingleSubject
# ---------------------------------------------------------------------------

class TestPrepareSingleSubject:

    def test_missing_dti_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        (dti_dir / "dti_corrected.nii.gz").unlink()
        out = tmp_path / "out" / "fod"

        rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw")

        assert rec.status == "failed"
        assert rec.output_created is False
        assert "dti_corrected.nii.gz" in (rec.error_message or "")

    def test_missing_bval_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        (dti_dir / "dti.bval").unlink()
        out = tmp_path / "out" / "fod"

        rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw")

        assert rec.status == "failed"
        assert "dti.bval" in (rec.error_message or "")

    def test_missing_brain_mask_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        (dti_dir / "qc" / "brain_mask.nii.gz").unlink()
        out = tmp_path / "out" / "fod"

        rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw")

        assert rec.status == "failed"
        assert "brain_mask.nii.gz" in (rec.error_message or "")

    def test_fa_below_min_returns_skipped(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed", fa_mean=0.05)
        out     = tmp_path / "out" / "fod"

        rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw")

        assert rec.status == "skipped"
        assert rec.output_created is False
        assert "FA=" in (rec.warning_message or "")

    def test_fa_above_max_returns_skipped(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed", fa_mean=0.70)
        out     = tmp_path / "out" / "fod"

        rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw")

        assert rec.status == "skipped"
        assert rec.output_created is False

    def test_fa_at_boundary_passes(self, tmp_path):
        """FA exactly at min boundary should be processed, not skipped."""
        dti_dir = _make_subject_tree(tmp_path / "processed", fa_mean=FA_MIN_CLEAN)
        out     = tmp_path / "out" / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                      use_mrtrix=False)

        assert rec.status == "success"

    def test_output_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        dti_dir = _make_subject_tree(raw)
        bad_output = dti_dir / "fod"

        with pytest.raises(ValueError, match="SAFETY"):
            prepare_subject_fod(dti_dir, bad_output, raw_root=raw)

    def test_output_dir_created(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        out     = tmp_path / "fod_out" / "nested" / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                use_mrtrix=False)

        assert out.is_dir()

    def test_report_written_on_success(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        out     = tmp_path / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                      use_mrtrix=False)

        report = json.loads((out / "fod_prep_report.json").read_text())
        assert report["status"] == "success"
        assert report["subject_id"] == rec.subject_id
        assert "timestamp" in report

    def test_backend_txt_written(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        out     = tmp_path / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                use_mrtrix=False)

        assert (out / "backend_used.txt").read_text().strip() == "dipy_csa"

    def test_gradient_counts_recorded(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed", n_vols=33)
        out     = tmp_path / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                      use_mrtrix=False)

        assert rec.input_volume_count == 33
        assert rec.b0_count == 1
        assert rec.dwi_count == 32

    def test_custom_fa_threshold_respected(self, tmp_path):
        """FA=0.35 should be skipped at max=0.30 but pass at max=0.40."""
        dti_dir = _make_subject_tree(tmp_path / "processed", fa_mean=0.35)
        out1 = tmp_path / "out1" / "fod"
        out2 = tmp_path / "out2" / "fod"

        rec_strict = prepare_subject_fod(
            dti_dir, out1, raw_root=tmp_path / "raw", fa_max=0.30, use_mrtrix=False
        )
        assert rec_strict.status == "skipped"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec_lenient = prepare_subject_fod(
                dti_dir, out2, raw_root=tmp_path / "raw", fa_max=0.40, use_mrtrix=False
            )
        assert rec_lenient.status == "success"


# ---------------------------------------------------------------------------
# TestBackendSelection
# ---------------------------------------------------------------------------

class TestBackendSelection:

    def test_no_mrtrix_defaults_to_dipy(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        out     = tmp_path / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._mrtrix3_available",
            return_value=False,
        ), patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                      use_mrtrix=None)

        assert rec.backend == "dipy_csa"

    def test_force_mrtrix_false_uses_dipy(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "processed")
        out     = tmp_path / "fod"

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            rec = prepare_subject_fod(dti_dir, out, raw_root=tmp_path / "raw",
                                      use_mrtrix=False)

        assert rec.backend == "dipy_csa"

    def test_mrtrix_available_when_binary_missing(self):
        # _mrtrix3_available() should return False if mrconvert is not on PATH.
        # This always passes in CI since MRtrix3 is not installed.
        assert _mrtrix3_available() is False


# ---------------------------------------------------------------------------
# TestSummaryCSV
# ---------------------------------------------------------------------------

class TestSummaryCSV:

    def _make_records(self) -> list[FODPrepRecord]:
        return [
            FODPrepRecord(
                site="bni", dataset="ABIDEII-BNI_1", subject_id="29001",
                fa_mean=0.25, backend="dipy_csa",
                input_volume_count=33, bval_count=33, bvec_count=33,
                b0_count=1, dwi_count=32,
                output_created=True, status="success",
            ),
            FODPrepRecord(
                site="bni", dataset="ABIDEII-BNI_1", subject_id="29002",
                fa_mean=0.72, backend="none",
                input_volume_count=0, bval_count=0, bvec_count=0,
                b0_count=0, dwi_count=0,
                output_created=False, status="skipped",
                warning_message="FA=0.7200 outside clean range [0.15, 0.40]",
            ),
            FODPrepRecord(
                site="nyu1", dataset="ABIDEII-NYU_1", subject_id="29100",
                fa_mean=0.24, backend="dipy_csa",
                input_volume_count=65, bval_count=65, bvec_count=65,
                b0_count=5, dwi_count=60,
                output_created=True, status="success",
            ),
        ]

    def test_summary_row_has_all_required_fields(self):
        rec = self._make_records()[0]
        row = rec.to_summary_row()
        expected = {
            "site", "dataset", "subject_id", "fa_mean", "backend",
            "input_volume_count", "bval_count", "bvec_count",
            "b0_count", "dwi_count", "output_created",
            "status", "warning_message", "error_message",
        }
        assert expected == set(row.keys())

    def test_run_summary_csv_written(self, tmp_path):
        records = self._make_records()
        run_csv, _ = save_summary_csvs(records, tmp_path)

        assert run_csv.exists()
        import pandas as pd
        df = pd.read_csv(run_csv)
        assert len(df) == 3
        assert "subject_id" in df.columns
        assert "status" in df.columns

    def test_site_summary_csv_written(self, tmp_path):
        records = self._make_records()
        _, site_csv = save_summary_csvs(records, tmp_path)

        assert site_csv.exists()
        import pandas as pd
        df = pd.read_csv(site_csv)
        assert set(df["site"]) == {"bni", "nyu1"}
        expected_cols = {
            "site", "total_subjects_considered", "subjects_processed",
            "subjects_skipped", "subjects_failed", "mean_fa",
            "mean_dwi_count", "backend_used", "common_b_values", "notes",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_site_summary_counts_correct(self, tmp_path):
        records = self._make_records()
        _, site_csv = save_summary_csvs(records, tmp_path)

        import pandas as pd
        df = pd.read_csv(site_csv).set_index("site")

        assert df.loc["bni", "total_subjects_considered"] == 2
        assert df.loc["bni", "subjects_processed"] == 1
        assert df.loc["bni", "subjects_skipped"] == 1
        assert df.loc["nyu1", "subjects_processed"] == 1


# ---------------------------------------------------------------------------
# TestBatchRunner
# ---------------------------------------------------------------------------

class TestBatchRunner:

    def test_batch_processes_configured_sites(self, tmp_path):
        processed = tmp_path / "processed"
        raw       = tmp_path / "raw"
        raw.mkdir()
        _make_subject_tree(processed, site="bni", subject_id="sub-001")
        _make_subject_tree(processed, site="bni", subject_id="sub-002")

        with patch(
            "neurofiber.tractography.fod_preparation._run_dipy_backend",
            return_value=("success", None, None),
        ):
            records = run_fod_preparation_batch(
                processed_root=processed,
                raw_root=raw,
                sites=["bni"],
                use_mrtrix=False,
            )

        assert len(records) == 2
        assert all(r.site == "bni" for r in records)

    def test_batch_skips_missing_site_dir(self, tmp_path):
        processed = tmp_path / "processed"
        raw       = tmp_path / "raw"
        raw.mkdir()
        processed.mkdir(parents=True)

        # "bni" directory does not exist
        records = run_fod_preparation_batch(
            processed_root=processed,
            raw_root=raw,
            sites=["bni"],
        )
        assert records == []

    def test_batch_output_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()

        with pytest.raises(ValueError, match="SAFETY"):
            run_fod_preparation_batch(
                processed_root=raw / "processed",
                raw_root=raw,
                sites=["bni"],
            )

    def test_clean_sites_constant(self):
        assert "bni"  in CLEAN_DTI_SITES
        assert "nyu1" in CLEAN_DTI_SITES
        assert "nyu2" in CLEAN_DTI_SITES
        assert "sdsu" in CLEAN_DTI_SITES
        assert "tcd"  in CLEAN_DTI_SITES
        assert "ip"   not in CLEAN_DTI_SITES  # excluded — FA inflation
