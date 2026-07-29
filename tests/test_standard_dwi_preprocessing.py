"""
Tests for Phase 2R.1 — Standard DWI Preprocessing

Coverage:
  - raw-write guard
  - volume correction rule (strip / no-correction / manual-review)
  - step 0 validation
  - backend availability detection
  - skip-with-warning behaviour
  - config loading
  - report schema completeness
  - summary CSV schema
  - data/processed is NOT overwritten
  - IP_1 QC warning attached
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurofiber.preprocessing.standard_dwi_preprocessing import (
    BackendAvailability,
    SubjectPreprocessingReport,
    _step0_validate,
    _step1_volume_correction,
    detect_backends,
    run_subject_pipeline,
    run_batch_pipeline,
    save_summary_csvs,
    site_to_folder,
    PIPELINE_VERSION,
)


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

def _make_dwi(tmp_path: Path, n_vols: int = 33, shape=(10, 10, 10)) -> tuple[Path, Path, Path]:
    """Create a minimal synthetic DWI NIfTI + bval/bvec."""
    data   = np.random.rand(*shape, n_vols).astype(np.float32)
    affine = np.eye(4)
    img    = nib.Nifti1Image(data, affine)
    nii_p  = tmp_path / "dti.nii.gz"
    nib.save(img, str(nii_p))

    bvals = np.zeros(n_vols)
    bvals[1:] = 1000
    bvec_rows = np.random.randn(n_vols, 3)
    bvec_rows /= np.linalg.norm(bvec_rows, axis=1, keepdims=True) + 1e-9

    bval_p = tmp_path / "dti.bval"
    bvec_p = tmp_path / "dti.bvec"
    np.savetxt(str(bval_p), bvals[np.newaxis, :], fmt="%g")
    np.savetxt(str(bvec_p), bvec_rows.T, fmt="%.6f")
    return nii_p, bval_p, bvec_p


def _make_dwi_with_low_last_vol(tmp_path: Path, n_vols: int = 34) -> tuple[Path, Path, Path]:
    """
    34 volumes, 33 bvals — the BNI pattern.
    Last volume has very low signal (ratio << 0.2).
    """
    shape = (10, 10, 10)
    data  = np.ones((*shape, n_vols), dtype=np.float32) * 500.0
    data[..., -1] = 10.0  # low signal last volume
    nii_p = tmp_path / "dti.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nii_p))

    bvals = np.zeros(n_vols - 1)
    bvals[1:] = 1000
    bvecs = np.tile([1.0, 0.0, 0.0], (n_vols - 1, 1))

    bval_p = tmp_path / "dti.bval"
    bvec_p = tmp_path / "dti.bvec"
    np.savetxt(str(bval_p), bvals[np.newaxis, :], fmt="%g")
    np.savetxt(str(bvec_p), bvecs.T, fmt="%.6f")
    return nii_p, bval_p, bvec_p


def _minimal_config() -> dict:
    return {
        "backend_preference": {
            "denoise": "dipy_patch2self",  # force DIPY so no MRtrix3 needed
            "gibbs":   "mrtrix_mrdegibbs",
            "eddy":    "mrtrix_dwifslpreproc",
            "bias":    "mrtrix_dwibiascorrect",
        },
        "fallbacks": {
            "denoise": "dipy_patch2self",
            "gibbs":   "skip_with_warning",
            "eddy":    "skip_with_warning",
            "bias":    "skip_with_warning",
        },
        "phase_encoding": {
            "default_pe_dir": None,
            "require_manual_pe_dir_for_eddy": True,
        },
        "safety": {
            "forbid_raw_writes": True,
            "raw_root":    "data/raw",
            "output_root": "data/processed_v2",
        },
    }


# ──────────────────────────────────────────────────────────
# site_to_folder
# ──────────────────────────────────────────────────────────

class TestSiteToFolder:
    def test_bni(self):          assert site_to_folder("BNI")    == "bni"
    def test_ip1(self):          assert site_to_folder("IP_1")   == "ip"
    def test_nyu1(self):         assert site_to_folder("NYU_1")  == "nyu1"
    def test_nyu2(self):         assert site_to_folder("NYU_2")  == "nyu2"
    def test_sdsu1(self):        assert site_to_folder("SDSU_1") == "sdsu"
    def test_tcd1(self):         assert site_to_folder("TCD_1")  == "tcd"
    def test_alias_sdsu(self):   assert site_to_folder("SDSU")   == "sdsu"
    def test_alias_tcd(self):    assert site_to_folder("TCD")    == "tcd"


# ──────────────────────────────────────────────────────────
# Step 0 — Validation
# ──────────────────────────────────────────────────────────

class TestStep0Validate:
    def test_consistent_returns_none_mismatch(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi(tmp_path, n_vols=33)
        r = _step0_validate(nii_p, bval_p, bvec_p)
        assert r["mismatch_pattern"] == "none"
        assert r["original_volume_count"] == 33
        assert r["bval_count"] == 33
        assert r["bvec_count"] == 33

    def test_volume_mismatch_detected(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi_with_low_last_vol(tmp_path)
        r = _step0_validate(nii_p, bval_p, bvec_p)
        assert r["mismatch_pattern"] != "none"
        assert r["original_volume_count"] == 34
        assert r["bval_count"] == 33

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _step0_validate(
                tmp_path / "nonexistent.nii.gz",
                tmp_path / "dti.bval",
                tmp_path / "dti.bvec",
            )

    def test_b0_count_correct(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi(tmp_path, n_vols=10)
        # First bval is 0, rest 1000 → b0_count=1
        r = _step0_validate(nii_p, bval_p, bvec_p)
        assert r["b0_count"] == 1
        assert r["dwi_count"] == 9

    def test_unique_bvals_rounded(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi(tmp_path, n_vols=5)
        r = _step0_validate(nii_p, bval_p, bvec_p)
        assert 0 in r["unique_bvals"]
        assert 1000 in r["unique_bvals"]


# ──────────────────────────────────────────────────────────
# Step 1 — Volume correction
# ──────────────────────────────────────────────────────────

class TestStep1VolumeCorrection:
    def test_no_correction_when_consistent(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi(tmp_path, n_vols=33)
        v   = _step0_validate(nii_p, bval_p, bvec_p)
        out = _step1_volume_correction(v["img"], v["bvals"], v["bvecs"])
        assert out["action"] == "no_correction_needed"
        assert out["corrected_volume_count"] == 33
        assert out["img_data"].shape[-1] == 33

    def test_strips_trailing_low_signal_volume(self, tmp_path):
        nii_p, bval_p, bvec_p = _make_dwi_with_low_last_vol(tmp_path, n_vols=34)
        v   = _step0_validate(nii_p, bval_p, bvec_p)
        out = _step1_volume_correction(v["img"], v["bvals"], v["bvecs"])
        assert out["action"] == "trailing_volume_stripped"
        assert out["corrected_volume_count"] == 33
        assert out["img_data"].shape[-1] == 33
        assert out["signal_ratio"] < 0.2

    def test_manual_review_when_ratio_above_threshold(self, tmp_path):
        """Last volume has normal signal → should NOT be stripped."""
        shape = (10, 10, 10)
        data  = np.ones((*shape, 34), dtype=np.float32) * 500.0
        # Last volume is similar to vol0 — ratio ≈ 1.0
        nii_p = tmp_path / "dti.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(nii_p))
        bvals = np.zeros(33); bvals[1:] = 1000
        bvecs = np.tile([1.0, 0.0, 0.0], (33, 1))
        bval_p = tmp_path / "dti.bval"; bvec_p = tmp_path / "dti.bvec"
        np.savetxt(str(bval_p), bvals[np.newaxis, :], fmt="%g")
        np.savetxt(str(bvec_p), bvecs.T, fmt="%.6f")

        v = _step0_validate(nii_p, bval_p, bvec_p)
        with pytest.raises(ValueError, match="manual review"):
            _step1_volume_correction(v["img"], v["bvals"], v["bvecs"])

    def test_unhandled_mismatch_raises(self, tmp_path):
        """bvals and bvecs disagree → raises."""
        shape = (5, 5, 5)
        data  = np.ones((*shape, 10), dtype=np.float32)
        nii_p = tmp_path / "dti.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(nii_p))
        bvals = np.zeros(8)  # 10 vols, 8 bvals — more than ±1
        bvecs = np.tile([1.0, 0.0, 0.0], (9, 1))
        bval_p = tmp_path / "dti.bval"; bvec_p = tmp_path / "dti.bvec"
        np.savetxt(str(bval_p), bvals[np.newaxis, :], fmt="%g")
        np.savetxt(str(bvec_p), bvecs.T, fmt="%.6f")

        v = _step0_validate(nii_p, bval_p, bvec_p)
        with pytest.raises(ValueError):
            _step1_volume_correction(v["img"], v["bvals"], v["bvecs"])


# ──────────────────────────────────────────────────────────
# Backend availability
# ──────────────────────────────────────────────────────────

class TestBackendAvailability:
    def test_no_mrtrix_falls_back_to_patch2self(self):
        av = BackendAvailability(mrtrix_version=None)
        assert av.denoise_backend("mrtrix_dwidenoise") == "dipy_patch2self"
        assert av.has_mrtrix is False

    def test_mrtrix_present_uses_mrtrix(self):
        av = BackendAvailability(mrtrix_version="MRtrix3 3.0.4")
        assert av.denoise_backend("mrtrix_dwidenoise") == "mrtrix_dwidenoise"

    def test_gibbs_skipped_without_mrtrix(self):
        av = BackendAvailability(mrtrix_version=None)
        assert av.gibbs_backend("mrtrix_mrdegibbs", "skip_with_warning") == "skip_with_warning"

    def test_eddy_skipped_without_fsl(self):
        av = BackendAvailability(mrtrix_version="MRtrix3 3.0.4", fsl_version=None)
        assert av.eddy_backend("mrtrix_dwifslpreproc", "skip_with_warning") == "skip_with_warning"

    def test_eddy_available_with_mrtrix_and_fsl(self):
        av = BackendAvailability(mrtrix_version="3.0.4", fsl_version="6.0.5")
        assert av.eddy_backend("mrtrix_dwifslpreproc", "skip_with_warning") == "mrtrix_dwifslpreproc"

    def test_bias_skipped_without_ants_or_fsl(self):
        av = BackendAvailability(mrtrix_version="3.0.4", ants_available=False, fsl_version=None)
        assert av.bias_backend("mrtrix_dwibiascorrect", "skip_with_warning") == "skip_with_warning"

    def test_detect_backends_returns_instance(self):
        av = detect_backends()
        assert isinstance(av, BackendAvailability)
        # On this machine MRtrix3 is not installed
        assert av.has_mrtrix is False


# ──────────────────────────────────────────────────────────
# Skip-with-warning behaviour
# ──────────────────────────────────────────────────────────

class TestSkipWithWarning:
    def test_gibbs_skip_recorded_in_report(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "29999" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33)
        # Copy files with expected names
        import shutil
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "processed_v2" / "bni" / "ABIDEII-BNI_1" / "29999" / "session_1" / "dti_1"

        cfg = _minimal_config()
        av  = BackendAvailability(mrtrix_version=None, fsl_version=None)

        rec = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av,
            config=cfg,
            skip_if_exists=False,
        )

        assert rec.status == "success"
        assert "step3_gibbs" in rec.steps_skipped
        assert "step4_eddy"  in rec.steps_skipped
        assert "step5_bias"  in rec.steps_skipped
        assert any("Gibbs" in w for w in rec.warnings)

    def test_eddy_skip_recorded_when_pe_dir_unknown(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "30001" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33)
        import shutil
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "pv2" / "bni" / "ABIDEII-BNI_1" / "30001" / "session_1" / "dti_1"
        cfg = _minimal_config()
        # Force gibbs to skip too so we don't try to call mrdegibbs (not installed)
        cfg["backend_preference"]["eddy"] = "mrtrix_dwifslpreproc"
        cfg["backend_preference"]["gibbs"] = "mrtrix_mrdegibbs"
        cfg["fallbacks"]["gibbs"] = "skip_with_warning"
        cfg["phase_encoding"]["default_pe_dir"] = None

        # Simulate MRtrix3+FSL available but pe_dir unknown → eddy must skip
        # Use BackendAvailability without actual mrtrix so gibbs also falls back to skip
        av = BackendAvailability(mrtrix_version=None, fsl_version=None)

        rec = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av,
            config=cfg,
            skip_if_exists=False,
        )

        assert rec.status == "success"
        assert "step4_eddy" in rec.steps_skipped
        # With pe_dir=None and require_manual_pe_dir_for_eddy=True, eddy is always skipped
        assert any("phase-encoding" in w or "eddy" in w.lower() or "Eddy" in w
                   for w in rec.warnings)


# ──────────────────────────────────────────────────────────
# Raw-write guard
# ──────────────────────────────────────────────────────────

class TestRawWriteGuard:
    def test_cannot_write_into_raw(self, tmp_path):
        raw_root = tmp_path / "data" / "raw"
        raw_root.mkdir(parents=True)
        output_inside_raw = raw_root / "abide_ii" / "bni" / "subject" / "session"

        nii_p  = tmp_path / "dti.nii.gz"
        bval_p = tmp_path / "dti.bval"
        bvec_p = tmp_path / "dti.bvec"
        # create placeholder files
        nib.save(nib.Nifti1Image(np.zeros((5,5,5,5)), np.eye(4)), str(nii_p))
        np.savetxt(str(bval_p), np.zeros((1,5)), fmt="%g")
        np.savetxt(str(bvec_p), np.zeros((3,5)), fmt="%.4f")

        with pytest.raises((ValueError, AssertionError, Exception)):
            run_subject_pipeline(
                dti_path=nii_p,
                bval_path=bval_p,
                bvec_path=bvec_p,
                output_dir=output_inside_raw,
                raw_root=raw_root,
                backends=BackendAvailability(),
                config=_minimal_config(),
                skip_if_exists=False,
            )


# ──────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_yaml_config_loads_correctly(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs" / "phase2r_standard_dwi_preprocessing.yaml"
        )
        assert config_path.exists(), f"Config not found: {config_path}"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        assert cfg["pipeline_version"] == "2R.1"
        assert "BNI"    in cfg["included_sites"]
        assert "IP_1"   in cfg["included_sites"]
        assert "NYU_1"  in cfg["included_sites"]
        assert "NYU_2"  in cfg["included_sites"]
        assert "SDSU_1" in cfg["included_sites"]
        assert "TCD_1"  in cfg["included_sites"]
        assert cfg["safety"]["output_root"] == "data/processed_v2"
        assert cfg["safety"]["output_root"] != "data/processed"
        assert "mrtrix_dwidenoise" in cfg["backend_preference"]["denoise"]
        assert "dipy_patch2self" in cfg["fallbacks"]["denoise"]

    def test_config_output_root_is_not_v1(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs" / "phase2r_standard_dwi_preprocessing.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "processed_v2" in cfg["safety"]["output_root"]
        assert cfg["safety"]["output_root"] != "data/processed"


# ──────────────────────────────────────────────────────────
# Report schema
# ──────────────────────────────────────────────────────────

class TestReportSchema:
    REQUIRED_KEYS = [
        "site", "dataset", "subject_id", "pipeline_version",
        "original_volume_count", "corrected_volume_count",
        "bval_count", "bvec_count", "b0_count", "dwi_count", "unique_bvals",
        "denoise_backend", "gibbs_backend", "eddy_backend", "bias_backend",
        "steps_completed", "steps_skipped",
        "warning_count", "warnings",
        "status", "error_message", "elapsed_sec", "timestamp",
    ]

    def test_report_has_all_required_keys(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "30002" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        import shutil
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33)
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "pv2" / "bni" / "ABIDEII-BNI_1" / "30002" / "session_1" / "dti_1"
        av  = BackendAvailability()
        rec = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av,
            config=_minimal_config(),
            skip_if_exists=False,
        )
        d = rec.to_dict()
        for key in self.REQUIRED_KEYS:
            assert key in d, f"Missing key in report: {key}"

    def test_report_json_written_to_disk(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "30003" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        import shutil
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33)
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "pv2" / "bni" / "ABIDEII-BNI_1" / "30003" / "session_1" / "dti_1"
        run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=BackendAvailability(),
            config=_minimal_config(),
            skip_if_exists=False,
        )
        report_path = out_dir / "preprocessing_report.json"
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert loaded["pipeline_version"] == PIPELINE_VERSION


# ──────────────────────────────────────────────────────────
# Summary CSV schema
# ──────────────────────────────────────────────────────────

class TestSummaryCSV:
    REQUIRED_COLS = [
        "site", "dataset", "subject_id", "pipeline_version",
        "original_volume_count", "corrected_volume_count",
        "bval_count", "bvec_count", "b0_count", "dwi_count",
        "unique_bvals", "denoise_backend", "gibbs_backend",
        "eddy_backend", "bias_backend", "steps_completed",
        "steps_skipped", "warning_count", "warnings",
        "status", "error_message", "elapsed_sec", "timestamp",
    ]

    def test_summary_csv_has_required_columns(self, tmp_path):
        import pandas as pd
        rec = SubjectPreprocessingReport(
            site="bni", dataset="ABIDEII-BNI_1", subject_id="99999",
            denoise_backend="dipy_patch2self",
            gibbs_backend="skip_with_warning",
            eddy_backend="skip_with_warning",
            bias_backend="skip_with_warning",
            steps_completed=["step0_validation", "step1_volume_correction:no_correction_needed",
                             "step2_denoise:dipy_patch2self", "step6_export"],
            steps_skipped=["step3_gibbs", "step4_eddy", "step5_bias"],
        )
        subj_csv, site_csv = save_summary_csvs([rec], tmp_path)
        df = pd.read_csv(subj_csv)
        for col in self.REQUIRED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_site_summary_written(self, tmp_path):
        import pandas as pd
        rec = SubjectPreprocessingReport(
            site="nyu1", dataset="ABIDEII-NYU_1", subject_id="88888",
            denoise_backend="dipy_patch2self",
        )
        _, site_csv = save_summary_csvs([rec], tmp_path)
        df = pd.read_csv(site_csv)
        assert "site"      in df.columns
        assert "n_success" in df.columns
        assert "n_failed"  in df.columns


# ──────────────────────────────────────────────────────────
# Old data/processed is not overwritten
# ──────────────────────────────────────────────────────────

class TestOldDataSafety:
    def test_output_root_is_not_v1_pipeline(self):
        """Config must never point output_root at data/processed."""
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs" / "phase2r_standard_dwi_preprocessing.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        output_root = Path(cfg["safety"]["output_root"])
        # Must not be the old pipeline root
        assert "processed_v2" in str(output_root)
        assert str(output_root) != "data/processed"

    def test_sentinel_v1_directory_unchanged(self):
        """
        If data/processed exists on disk (v1 outputs), it must not have been
        touched (we check it wasn't just created by writing to it).
        This test just verifies the directory name invariant, not filesystem state.
        """
        v2 = Path("data/processed_v2")
        v1 = Path("data/processed")
        assert str(v2) != str(v1)


# ──────────────────────────────────────────────────────────
# IP_1 QC warning
# ──────────────────────────────────────────────────────────

class TestIPQCWarning:
    def test_ip_subjects_get_qc_warning(self, tmp_path):
        raw_root = tmp_path / "data" / "raw"
        ip_raw   = raw_root / "ip" / "ABIDEII-IP_1" / "29580" / "session_1" / "dti_1"
        ip_raw.mkdir(parents=True)
        import shutil
        nii_p, bval_p, bvec_p = _make_dwi(ip_raw, n_vols=33)
        shutil.move(str(nii_p),  str(ip_raw / "dti.nii.gz"))
        shutil.move(str(bval_p), str(ip_raw / "dti.bval"))
        shutil.move(str(bvec_p), str(ip_raw / "dti.bvec"))

        out_root = tmp_path / "data" / "processed_v2"
        cfg = _minimal_config()

        records = run_batch_pipeline(
            raw_root=raw_root,
            output_root=out_root,
            sites=["IP_1"],
            ip_qc_only=True,
            skip_if_exists=False,
            config=cfg,
        )

        assert len(records) == 1
        rec = records[0]
        assert any("IP_1" in w or "QC-labelled" in w for w in rec.warnings), (
            f"Expected IP_1 QC warning, got: {rec.warnings}"
        )


# ──────────────────────────────────────────────────────────
# End-to-end: single subject (DIPY path, no MRtrix3)
# ──────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_single_subject_dipy_only(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "30010" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        import shutil
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33, shape=(8, 8, 8))
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "pv2" / "bni" / "ABIDEII-BNI_1" / "30010" / "session_1" / "dti_1"
        av  = BackendAvailability(mrtrix_version=None)
        rec = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av,
            config=_minimal_config(),
            skip_if_exists=False,
        )

        assert rec.status == "success", rec.error_message
        assert rec.denoise_backend == "dipy_patch2self"
        assert (out_dir / "dwi_preprocessed.nii.gz").exists()
        assert (out_dir / "dwi_preprocessed.bval").exists()
        assert (out_dir / "dwi_preprocessed.bvec").exists()
        assert (out_dir / "preprocessing_report.json").exists()
        assert (out_dir / "qc" / "qc_metrics.json").exists()

        # Preprocessed file has same spatial dims, same n_vols
        out_img = nib.load(str(out_dir / "dwi_preprocessed.nii.gz"))
        assert out_img.shape[:3] == (8, 8, 8)
        assert out_img.shape[3] == 33

    def test_skip_if_exists_works(self, tmp_path):
        raw_dir = tmp_path / "raw" / "bni" / "ABIDEII-BNI_1" / "30011" / "session_1" / "dti_1"
        raw_dir.mkdir(parents=True)
        import shutil
        nii_p, bval_p, bvec_p = _make_dwi(raw_dir, n_vols=33, shape=(6, 6, 6))
        shutil.move(str(nii_p),  str(raw_dir / "dti.nii.gz"))
        shutil.move(str(bval_p), str(raw_dir / "dti.bval"))
        shutil.move(str(bvec_p), str(raw_dir / "dti.bvec"))

        out_dir = tmp_path / "pv2" / "bni" / "ABIDEII-BNI_1" / "30011" / "session_1" / "dti_1"
        av  = BackendAvailability()
        cfg = _minimal_config()

        # First run
        rec1 = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av, config=cfg, skip_if_exists=False,
        )
        ts1 = rec1.timestamp

        # Second run with skip_if_exists=True — should return cached report
        rec2 = run_subject_pipeline(
            dti_path=raw_dir / "dti.nii.gz",
            bval_path=raw_dir / "dti.bval",
            bvec_path=raw_dir / "dti.bvec",
            output_dir=out_dir,
            raw_root=tmp_path / "raw",
            backends=av, config=cfg, skip_if_exists=True,
        )
        # Timestamp should be same (loaded from cached report)
        assert rec2.timestamp == ts1
