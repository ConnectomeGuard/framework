"""Tests for Phase 3.2 — Multi-site Streamline Tractography MVP."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from neurofiber.tractography.streamline_generation import (
    CLEAN_DTI_SITES_DISPLAY,
    StreamlineRecord,
    _compute_lengths_mm,
    generate_subject_streamlines,
    load_config,
    run_streamline_batch,
    save_summary_csvs,
    site_to_folder,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_peaks_pam5(dti_dir: Path, n_vols: int = 33) -> None:
    """
    Create a minimal real peaks.pam5 by running Phase 3.1 on tiny synthetic DWI.
    This is used only by integration tests that need actual tracking.
    """
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.direction import peaks_from_model
    from dipy.io.peaks import save_peaks
    from dipy.reconst.shm import CsaOdfModel

    fod_dir = dti_dir / "fod"
    fod_dir.mkdir(exist_ok=True)

    rng  = np.random.default_rng(7)
    data = rng.uniform(40, 100, (10, 10, 6, n_vols)).astype(np.float32)
    affine = np.eye(4)

    bvals = np.zeros(n_vols)
    bvals[1:] = 2500
    bvecs = np.zeros((3, n_vols))
    for i in range(n_vols - 1):
        theta = np.pi * i / (n_vols - 1)
        phi   = 2 * np.pi * i / (n_vols - 1)
        bvecs[:, i + 1] = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            abs(np.cos(theta)) + 0.01,
        ]
        bvecs[:, i + 1] /= np.linalg.norm(bvecs[:, i + 1]) + 1e-9

    gtab   = gradient_table(bvals, bvecs)
    model  = CsaOdfModel(gtab, sh_order=6, smooth=0.006)
    sphere = get_sphere("repulsion724")
    mask   = np.ones((10, 10, 6), dtype=bool)
    peaks  = peaks_from_model(model, data, sphere,
                               relative_peak_threshold=0.5,
                               min_separation_angle=25,
                               mask=mask, return_sh=True,
                               npeaks=5, normalize_peaks=True,
                               parallel=False)
    save_peaks(str(fod_dir / "peaks.pam5"), peaks, affine)


def _make_subject_tree(
    root: Path,
    site: str = "bni",
    dataset: str = "ABIDEII-BNI_1",
    subject_id: str = "sub-001",
    fa_mean_val: float = 0.25,
    create_peaks: bool = False,
) -> Path:
    """
    Create a minimal Phase 2+3.1 subject tree and return the dti_1/ Path.
    """
    dti_dir    = root / site / dataset / subject_id / "session_1" / "dti_1"
    tensor_dir = dti_dir / "tensor"
    qc_dir     = dti_dir / "qc"
    dti_dir.mkdir(parents=True)
    tensor_dir.mkdir()
    qc_dir.mkdir()

    # FA map — uniform value inside a central brain mask
    fa_data = np.zeros((10, 10, 6), dtype=np.float32)
    fa_data[2:8, 2:8, 1:5] = fa_mean_val
    nib.save(nib.Nifti1Image(fa_data, np.eye(4)), str(tensor_dir / "FA.nii.gz"))

    # Brain mask — same central box
    mask = np.zeros((10, 10, 6), dtype=np.uint8)
    mask[2:8, 2:8, 1:5] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(qc_dir / "brain_mask.nii.gz"))

    # Tensor QC report
    (qc_dir / "tensor_qc_report.json").write_text(
        json.dumps({"subject_id": subject_id, "fa_mean": fa_mean_val, "status": "success"})
    )

    if create_peaks:
        _make_peaks_pam5(dti_dir)

    return dti_dir


def _make_config(tmp_path: Path, **overrides) -> Path:
    import yaml
    cfg = {
        "included_sites": ["BNI", "NYU_1"],
        "excluded_sites": ["IP_1"],
        "fa_threshold": 0.15,
        "seeds_per_subject": 200,
        "step_size": 0.5,
        "max_angle": 30.0,
        "max_cross": 1,
        "random_seed": 42,
        "qc": {
            "min_streamline_count": 500,
            "min_mean_length_mm": 20.0,
            "max_file_size_mb": 500.0,
        },
    }
    cfg.update(overrides)
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg_path


# ---------------------------------------------------------------------------
# TestSiteNormalisation
# ---------------------------------------------------------------------------

class TestSiteNormalisation:

    def test_display_names_mapped_correctly(self):
        assert site_to_folder("BNI")    == "bni"
        assert site_to_folder("NYU_1")  == "nyu1"
        assert site_to_folder("NYU_2")  == "nyu2"
        assert site_to_folder("SDSU_1") == "sdsu"
        assert site_to_folder("TCD_1")  == "tcd"
        assert site_to_folder("IP_1")   == "ip"

    def test_folder_names_pass_through(self):
        assert site_to_folder("bni")  == "bni"
        assert site_to_folder("nyu1") == "nyu1"

    def test_clean_sites_constant(self):
        assert "BNI"    in CLEAN_DTI_SITES_DISPLAY
        assert "NYU_1"  in CLEAN_DTI_SITES_DISPLAY
        assert "SDSU_1" in CLEAN_DTI_SITES_DISPLAY
        assert "IP_1"   not in CLEAN_DTI_SITES_DISPLAY


# ---------------------------------------------------------------------------
# TestConfigLoading
# ---------------------------------------------------------------------------

class TestConfigLoading:

    def test_load_valid_config(self, tmp_path):
        cfg_path = _make_config(tmp_path)
        cfg = load_config(cfg_path)
        assert cfg["fa_threshold"] == 0.15
        assert "BNI" in cfg["included_sites"]
        assert cfg["seeds_per_subject"] == 200

    def test_missing_required_key_raises(self, tmp_path):
        import yaml
        cfg = {"included_sites": ["BNI"]}  # missing fa_threshold etc.
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump(cfg))
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(bad)


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_missing_fa_map_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "proc", create_peaks=False)
        (dti_dir / "tensor" / "FA.nii.gz").unlink()

        rec = generate_subject_streamlines(
            dti_dir, dti_dir / "tractography", raw_root=tmp_path / "raw"
        )
        assert rec.status == "failed"
        assert "FA.nii.gz" in (rec.error_message or "")

    def test_missing_mask_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "proc", create_peaks=False)
        (dti_dir / "qc" / "brain_mask.nii.gz").unlink()

        rec = generate_subject_streamlines(
            dti_dir, dti_dir / "tractography", raw_root=tmp_path / "raw"
        )
        assert rec.status == "failed"
        assert "brain_mask.nii.gz" in (rec.error_message or "")

    def test_missing_peaks_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path / "proc", create_peaks=False)
        # fod/ dir not created, peaks.pam5 absent

        rec = generate_subject_streamlines(
            dti_dir, dti_dir / "tractography", raw_root=tmp_path / "raw"
        )
        assert rec.status == "failed"
        assert "peaks.pam5" in (rec.error_message or "")

    def test_output_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        dti_dir = _make_subject_tree(raw, create_peaks=False)
        bad_output = dti_dir / "tractography"

        with pytest.raises(ValueError, match="SAFETY"):
            generate_subject_streamlines(dti_dir, bad_output, raw_root=raw)


# ---------------------------------------------------------------------------
# TestStreamlineLengthHelper
# ---------------------------------------------------------------------------

class TestStreamlineLengthHelper:

    def test_straight_streamline_length(self):
        # 11 points, 1mm apart → length = 10mm
        s = np.array([[0, 0, i] for i in range(11)], dtype=float)
        lengths = _compute_lengths_mm([s])
        assert abs(lengths[0] - 10.0) < 1e-6

    def test_empty_streamline_length_is_zero(self):
        s = np.array([[0, 0, 0]], dtype=float)
        lengths = _compute_lengths_mm([s])
        assert lengths[0] == 0.0

    def test_multiple_streamlines(self):
        s1 = np.array([[0, 0, i] for i in range(6)], dtype=float)   # 5mm
        s2 = np.array([[0, 0, i] for i in range(11)], dtype=float)  # 10mm
        lengths = _compute_lengths_mm([s1, s2])
        assert abs(lengths[0] - 5.0)  < 1e-6
        assert abs(lengths[1] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# TestSummaryCSV
# ---------------------------------------------------------------------------

class TestSummaryCSV:

    def _make_records(self) -> list[StreamlineRecord]:
        return [
            StreamlineRecord(
                site="bni", dataset="ABIDEII-BNI_1", subject_id="29001",
                fa_mean=0.25, fa_threshold=0.15,
                seed_count=5000, streamline_count=1200,
                mean_streamline_length=45.2, median_streamline_length=42.0,
                min_streamline_length=5.0,  max_streamline_length=180.0,
                output_file_size_mb=2.3, status="success",
            ),
            StreamlineRecord(
                site="bni", dataset="ABIDEII-BNI_1", subject_id="29002",
                fa_mean=0.24, fa_threshold=0.15,
                seed_count=5000, streamline_count=0,
                mean_streamline_length=None, median_streamline_length=None,
                min_streamline_length=None, max_streamline_length=None,
                output_file_size_mb=None, status="failed",
                error_message="Zero streamlines generated",
            ),
        ]

    def test_summary_row_has_all_required_fields(self):
        rec = self._make_records()[0]
        row = rec.to_summary_row()
        expected = {
            "site", "dataset", "subject_id", "fa_mean", "fa_threshold",
            "seed_count", "streamline_count",
            "mean_streamline_length", "median_streamline_length",
            "min_streamline_length", "max_streamline_length",
            "output_file_size_mb", "status", "warning_message", "error_message",
        }
        assert expected == set(row.keys())

    def test_run_csv_written(self, tmp_path):
        records = self._make_records()
        run_csv, _ = save_summary_csvs(records, tmp_path)
        assert run_csv.exists()
        import pandas as pd
        df = pd.read_csv(run_csv)
        assert len(df) == 2
        assert "streamline_count" in df.columns

    def test_site_csv_written(self, tmp_path):
        records = self._make_records()
        _, site_csv = save_summary_csvs(records, tmp_path)
        assert site_csv.exists()
        import pandas as pd
        df = pd.read_csv(site_csv)
        expected_cols = {
            "site", "subjects_considered", "subjects_processed", "subjects_failed",
            "mean_streamline_count", "mean_streamline_length", "mean_fa", "notes",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 1  # only "bni"


# ---------------------------------------------------------------------------
# TestBatchRunner
# ---------------------------------------------------------------------------

class TestBatchRunner:

    def test_subject_failure_does_not_crash_batch(self, tmp_path):
        """One subject with missing peaks should not stop the other."""
        proc = tmp_path / "proc"
        raw  = tmp_path / "raw"
        raw.mkdir()

        # sub-001 — valid (peaks created)
        dti1 = _make_subject_tree(proc, subject_id="sub-001", create_peaks=True)
        # sub-002 — missing peaks
        _make_subject_tree(proc, subject_id="sub-002", create_peaks=False)

        records = run_streamline_batch(
            processed_root=proc,
            raw_root=raw,
            sites=["bni"],
            fa_threshold=0.15,
            seeds_per_subject=30,
        )
        assert len(records) == 2
        statuses = {r.subject_id: r.status for r in records}
        assert statuses["sub-001"] == "success"
        assert statuses["sub-002"] == "failed"

    def test_batch_skips_missing_site_dir(self, tmp_path):
        proc = tmp_path / "proc"
        proc.mkdir()
        raw  = tmp_path / "raw"
        raw.mkdir()

        records = run_streamline_batch(
            processed_root=proc,
            raw_root=raw,
            sites=["bni"],
        )
        assert records == []

    def test_batch_output_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()

        with pytest.raises(ValueError, match="SAFETY"):
            run_streamline_batch(
                processed_root=raw / "processed",
                raw_root=raw,
                sites=["bni"],
            )

    def test_output_path_created(self, tmp_path):
        proc = tmp_path / "proc"
        raw  = tmp_path / "raw"
        raw.mkdir()

        dti_dir = _make_subject_tree(proc, create_peaks=True)
        records = run_streamline_batch(
            processed_root=proc,
            raw_root=raw,
            sites=["bni"],
            seeds_per_subject=30,
        )
        assert records[0].status == "success"
        assert (dti_dir / "tractography" / "streamlines.trk").exists()
        assert (dti_dir / "tractography" / "tractography_report.json").exists()
        assert (dti_dir / "tractography" / "backend_used.txt").exists()

    def test_backend_txt_content(self, tmp_path):
        proc = tmp_path / "proc"
        raw  = tmp_path / "raw"
        raw.mkdir()
        dti_dir = _make_subject_tree(proc, create_peaks=True)

        run_streamline_batch(
            processed_root=proc, raw_root=raw, sites=["bni"], seeds_per_subject=30
        )
        content = (dti_dir / "tractography" / "backend_used.txt").read_text().strip()
        assert content == "dipy_deterministic"

    def test_report_json_schema(self, tmp_path):
        proc = tmp_path / "proc"
        raw  = tmp_path / "raw"
        raw.mkdir()
        dti_dir = _make_subject_tree(proc, create_peaks=True)

        run_streamline_batch(
            processed_root=proc, raw_root=raw, sites=["bni"], seeds_per_subject=30
        )
        import json
        report = json.loads(
            (dti_dir / "tractography" / "tractography_report.json").read_text()
        )
        for key in ("site", "subject_id", "seed_count", "streamline_count",
                    "mean_streamline_length", "status", "timestamp"):
            assert key in report

    def test_display_name_sites_resolved(self, tmp_path):
        """Batch should accept display names like 'BNI' and resolve to folder 'bni'."""
        proc = tmp_path / "proc"
        raw  = tmp_path / "raw"
        raw.mkdir()
        _make_subject_tree(proc, site="bni", create_peaks=True)

        records = run_streamline_batch(
            processed_root=proc, raw_root=raw,
            sites=["BNI"],   # display name
            seeds_per_subject=30,
        )
        assert len(records) == 1
        assert records[0].status == "success"
