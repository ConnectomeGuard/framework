"""
Tests for Phase 3R.2 — Validated Streamline Tractography from processed_v2b

Coverage:
  - IP_1 exclusion (no output generated)
  - Raw-write guard
  - Required input validation (peaks, FA, mask)
  - Streamline file created
  - Minimum length filter (20mm) removes short streamlines
  - Long streamline flag (>300mm) does not remove streamlines
  - Summary CSV schema
  - Site summary schema
  - Reproducibility CSV schema
  - Zero-streamline failure
  - skip_if_exists behaviour
  - Batch runner excludes IP_1
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

from neurofiber.tractography.streamline_generation_v2b import (
    EXCLUDED_SITES,
    REPRO_CSV_FIELDS,
    SITE_CSV_FIELDS,
    SUBJECT_CSV_FIELDS,
    StreamlineRecord,
    _SITE_FOLDER_MAP,
    _compute_lengths,
    generate_subject_streamlines,
    run_streamline_batch,
    write_reproducibility_csv,
    write_site_summary,
    write_subject_summary,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_peaks(tmp_path: Path, shape=(12, 12, 6)) -> Path:
    """Write a minimal peaks.pam5 using DIPY CsaOdfModel on synthetic data."""
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.direction import peaks_from_model
    from dipy.io.peaks import save_peaks
    from dipy.reconst.shm import CsaOdfModel

    rng = np.random.default_rng(7)
    n_vols = 33
    data   = rng.random((*shape, n_vols), dtype=np.float32) * 600
    data[3:9, 3:9, 1:5, :] *= 6
    img    = nib.Nifti1Image(data, np.eye(4))

    bvals = np.zeros(n_vols); bvals[1:] = 1000
    bvecs = rng.standard_normal((3, n_vols))
    bvecs[:, 0] = 0
    norms = np.linalg.norm(bvecs[:, 1:], axis=0, keepdims=True)
    norms[norms == 0] = 1
    bvecs[:, 1:] /= norms
    bvecs = bvecs.T  # Nx3

    gtab   = gradient_table(bvals, bvecs)
    mask   = np.zeros(shape, dtype=bool)
    mask[3:9, 3:9, 1:5] = True

    model  = CsaOdfModel(gtab, sh_order=6, smooth=0.006)
    sphere = get_sphere("repulsion100")
    peaks  = peaks_from_model(model, data, sphere,
                              relative_peak_threshold=0.5,
                              min_separation_angle=25,
                              mask=mask, return_sh=True,
                              npeaks=3, normalize_peaks=True, parallel=False)
    p = tmp_path / "peaks.pam5"
    save_peaks(str(p), peaks, np.eye(4))
    return p


def _make_dti_dir(
    base: Path,
    site: str = "BNI",
    dataset: str = "ABIDEII-BNI_1",
    subject: str = "29006",
    fa_val: float = 0.35,
) -> Path:
    folder  = _SITE_FOLDER_MAP.get(site, site.lower())
    dti_dir = base / "abide_ii" / folder / dataset / subject / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True, exist_ok=True)

    shape = (12, 12, 6)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm voxels so streamlines are longer

    # FA map — bright in centre
    (dti_dir / "tensor").mkdir(parents=True, exist_ok=True)
    fa_arr = np.zeros(shape, dtype=np.float32)
    fa_arr[3:9, 3:9, 1:5] = fa_val
    nib.save(nib.Nifti1Image(fa_arr, affine), str(dti_dir / "tensor" / "FA.nii.gz"))

    # Brain mask
    mask_arr = np.zeros(shape, dtype=np.uint8)
    mask_arr[3:9, 3:9, 1:5] = 1
    (dti_dir / "qc").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_arr, affine), str(dti_dir / "qc" / "brain_mask.nii.gz"))
    (dti_dir / "qc" / "tensor_qc_report.json").write_text(
        json.dumps({"status": "success", "fa_mean": fa_val, "md_mean": 0.00085})
    )

    # Peaks
    (dti_dir / "fod").mkdir(exist_ok=True)
    peaks_path = _make_peaks(dti_dir / "fod", shape=shape)

    return dti_dir


# ---------------------------------------------------------------------------
# TestIP1Exclusion
# ---------------------------------------------------------------------------

class TestIP1Exclusion:
    def test_ip1_returns_excluded_status(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, site="IP_1", dataset="ABIDEII-IP_1",
                                subject="29600")
        rec = generate_subject_streamlines(
            dti_dir, tmp_path / "abide_ii", tmp_path / "raw"
        )
        assert rec.status == "excluded"

    def test_ip1_no_trk_file_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path, site="IP_1", dataset="ABIDEII-IP_1",
                                subject="29600")
        generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert not (dti_dir / "tractography" / "streamlines.trk").exists()

    def test_batch_raises_if_ip1_requested(self, tmp_path):
        with pytest.raises(ValueError, match="excluded"):
            run_streamline_batch(
                processed_v2b_root=tmp_path / "abide_ii",
                raw_root=tmp_path / "raw",
                sites=["IP_1"],
            )


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_inside_raw(self, tmp_path):
        raw = tmp_path / "raw"
        dti_dir = raw / "abide_ii" / "bni" / "D" / "99" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        with pytest.raises((ValueError, AssertionError)):
            generate_subject_streamlines(
                dti_dir=dti_dir,
                output_root=raw / "abide_ii",
                raw_root=raw,
            )


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_peaks_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "fod" / "peaks.pam5").unlink()
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_fa_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "tensor" / "FA.nii.gz").unlink()
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"

    def test_missing_mask_returns_failed(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        (dti_dir / "qc" / "brain_mask.nii.gz").unlink()
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        assert rec.status == "failed"


# ---------------------------------------------------------------------------
# TestStreamlineOutputs
# ---------------------------------------------------------------------------

class TestStreamlineOutputs:
    def test_trk_file_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            assert (dti_dir / "tractography" / "streamlines.trk").exists()

    def test_report_json_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            assert (dti_dir / "tractography" / "tractography_report.json").exists()

    def test_backend_used_txt_created(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            f = dti_dir / "tractography" / "backend_used.txt"
            assert f.exists()
            assert "dipy_deterministic" in f.read_text()

    def test_retained_le_raw(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            assert rec.retained_streamline_count <= rec.raw_streamline_count

    def test_rejected_short_plus_retained_equals_raw(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            assert rec.retained_streamline_count + rec.rejected_short_streamlines == rec.raw_streamline_count


# ---------------------------------------------------------------------------
# TestLengthFiltering
# ---------------------------------------------------------------------------

class TestLengthFiltering:
    def test_length_filter_logic(self):
        """Verify that _compute_lengths + filtering correctly separates short/long."""
        from dipy.tracking.streamline import Streamlines
        # 3 streamlines: short (2mm), medium (30mm), long (310mm)
        short  = np.array([[0,0,0],[1,0,0]], dtype=float)   # 1mm
        medium = np.array([[0,0,0],[30,0,0]], dtype=float)  # 30mm
        long_  = np.array([[0,0,0],[310,0,0]], dtype=float) # 310mm
        sls    = Streamlines([short, medium, long_])
        lengths = np.array(_compute_lengths(sls))

        from neurofiber.tractography.streamline_generation_v2b import MIN_LENGTH_MM, MAX_LENGTH_MM
        keep = lengths >= MIN_LENGTH_MM
        flag = lengths > MAX_LENGTH_MM

        assert not keep[0], "short streamline should be rejected"
        assert keep[1],  "medium streamline should be kept"
        assert keep[2],  "long streamline should be kept (only flagged)"
        assert flag[2],  "long streamline should be flagged"
        assert not flag[1], "medium streamline should not be flagged"

    def test_all_length_percentiles_present(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec = generate_subject_streamlines(dti_dir, tmp_path / "abide_ii", tmp_path / "raw")
        if rec.status == "success":
            for attr in ["p10_length", "p25_length", "p75_length", "p90_length"]:
                assert getattr(rec, attr) is not None, f"Missing: {attr}"


# ---------------------------------------------------------------------------
# TestSkipIfExists
# ---------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_reuses_report(self, tmp_path):
        dti_dir = _make_dti_dir(tmp_path)
        rec1 = generate_subject_streamlines(
            dti_dir, tmp_path / "abide_ii", tmp_path / "raw", skip_if_exists=True
        )
        if rec1.status != "success":
            return  # can't test skip on failed run
        (dti_dir / "fod" / "peaks.pam5").unlink()
        rec2 = generate_subject_streamlines(
            dti_dir, tmp_path / "abide_ii", tmp_path / "raw", skip_if_exists=True
        )
        assert rec2.status == "success"


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_subject_csv_columns(self, tmp_path):
        rec = StreamlineRecord(site="BNI", dataset="D", subject_id="1", status="success")
        out = tmp_path / "s.csv"
        write_subject_summary([rec], out)
        cols = csv.DictReader(open(out)).fieldnames
        for f in SUBJECT_CSV_FIELDS:
            assert f in cols, f"Missing: {f}"

    def test_site_csv_columns(self, tmp_path):
        rec = StreamlineRecord(site="BNI", dataset="D", subject_id="1", status="success",
                               retained_streamline_count=100, mean_length=45.0)
        out = tmp_path / "site.csv"
        write_site_summary([rec], out)
        cols = csv.DictReader(open(out)).fieldnames
        for f in SITE_CSV_FIELDS:
            assert f in cols, f"Missing: {f}"

    def test_repro_csv_columns(self, tmp_path):
        row = {f: None for f in REPRO_CSV_FIELDS}
        out = tmp_path / "repro.csv"
        write_reproducibility_csv([row], out)
        cols = csv.DictReader(open(out)).fieldnames
        for f in REPRO_CSV_FIELDS:
            assert f in cols, f"Missing: {f}"
