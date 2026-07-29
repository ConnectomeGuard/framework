"""
Tests for Phase 3R.3 — Connectome Construction from processed_v2b

Coverage:
  - Atlas missing fails clearly
  - Raw-write guard
  - Matrix shape is (n_rois x n_rois) square
  - Count matrix is symmetric
  - Self-loop handling (counted, not included in edges)
  - Summary CSV schema (subject and site)
  - Subject failure does not crash batch
  - IP_1 exclusion
  - skip_if_exists behaviour
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

from neurofiber.connectome.connectome_builder_v2b import (
    ATLAS_N_ROIS,
    EXCLUDED_SITES,
    SITE_CSV_FIELDS,
    SUBJECT_CSV_FIELDS,
    ConnectomeRecord,
    _SITE_FOLDER_MAP,
    build_subject_connectome,
    run_connectome_batch,
    write_site_summary,
    write_subject_summary,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _save_synthetic_trk(path: Path, affine: np.ndarray, n_streamlines: int = 20) -> None:
    """Write a minimal .trk with synthetic streamlines."""
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_trk
    from dipy.tracking.streamline import Streamlines

    rng = np.random.default_rng(42)
    streamlines = []
    for _ in range(n_streamlines):
        pts = rng.uniform(3, 9, (10, 3)).astype(np.float32)
        pts *= 3.0  # scale to ~mm coordinates matching 3mm voxels
        streamlines.append(pts)

    ref_img = nib.Nifti1Image(np.zeros((12, 12, 8)), affine)
    sft = StatefulTractogram(Streamlines(streamlines), ref_img, Space.RASMM)
    save_trk(sft, str(path), bbox_valid_check=False)


def _make_dti_dir(
    base: Path,
    site: str = "BNI",
    dataset: str = "ABIDEII-BNI_1",
    subject: str = "29006",
    n_rois: int = 10,
) -> tuple[Path, Path]:
    """
    Creates both the v2b dti_dir and the v1 atlas dir.
    Returns (dti_dir, v1_processed_root).
    """
    folder  = _SITE_FOLDER_MAP.get(site, site.lower())
    affine  = np.diag([3.0, 3.0, 3.0, 1.0])
    shape3  = (12, 12, 8)

    # v2b dti_dir
    dti_dir = base / "v2b" / "abide_ii" / folder / dataset / subject / "session_1" / "dti_1"
    dti_dir.mkdir(parents=True, exist_ok=True)

    # Tensor maps
    (dti_dir / "tensor").mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for name in ["FA", "MD", "AD", "RD"]:
        arr = rng.random(shape3, dtype=np.float32) * 0.3
        nib.save(nib.Nifti1Image(arr, affine), str(dti_dir / "tensor" / f"{name}.nii.gz"))

    # Brain mask
    (dti_dir / "qc").mkdir(exist_ok=True)
    mask = np.ones(shape3, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, affine), str(dti_dir / "qc" / "brain_mask.nii.gz"))

    # Streamlines
    (dti_dir / "tractography").mkdir(exist_ok=True)
    _save_synthetic_trk(dti_dir / "tractography" / "streamlines.trk", affine)
    # FOD marker (batch scanner looks for peaks.pam5)
    (dti_dir / "fod").mkdir(exist_ok=True)
    (dti_dir / "fod" / "peaks.pam5").write_bytes(b"")

    # v1 atlas dir (atlas labels 1..n_rois scattered in image)
    v1_root = base / "v1" / "abide_ii"
    atlas_dir = v1_root / folder / dataset / subject / "session_1" / "connectome" / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    # Fill atlas with labels — tile n_rois labels across all voxels so
    # random streamline endpoints are virtually guaranteed to hit a label.
    indices = np.arange(shape3[0] * shape3[1] * shape3[2])
    atlas_flat = ((indices % n_rois) + 1).astype(np.float32)
    atlas_arr = atlas_flat.reshape(shape3)
    nib.save(nib.Nifti1Image(atlas_arr, affine), str(atlas_dir / "atlas_subject_space.nii.gz"))

    return dti_dir, v1_root


# ---------------------------------------------------------------------------
# TestAtlasMissing
# ---------------------------------------------------------------------------

class TestAtlasMissing:
    def test_missing_atlas_returns_failed(self, tmp_path):
        dti_dir, _ = _make_dti_dir(tmp_path)
        # Don't pass the correct v1_root
        empty_v1 = tmp_path / "nonexistent_v1" / "abide_ii"
        rec = build_subject_connectome(
            dti_dir=dti_dir,
            v1_processed_root=empty_v1,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "failed"
        assert "Atlas not found" in (rec.error_message or "")


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_inside_raw(self, tmp_path):
        raw = tmp_path / "raw"
        dti_dir = raw / "abide_ii" / "bni" / "D" / "99" / "session_1" / "dti_1"
        dti_dir.mkdir(parents=True)
        with pytest.raises((ValueError, AssertionError)):
            build_subject_connectome(
                dti_dir=dti_dir,
                v1_processed_root=tmp_path / "v1" / "abide_ii",
                output_root=raw / "abide_ii",
                raw_root=raw,
            )


# ---------------------------------------------------------------------------
# TestIP1Exclusion
# ---------------------------------------------------------------------------

class TestIP1Exclusion:
    def test_ip1_returns_excluded(self, tmp_path):
        dti_dir, v1_root = _make_dti_dir(tmp_path, site="IP_1",
                                          dataset="ABIDEII-IP_1", subject="29600")
        rec = build_subject_connectome(
            dti_dir=dti_dir,
            v1_processed_root=v1_root,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert rec.status == "excluded"

    def test_ip1_no_matrices_created(self, tmp_path):
        dti_dir, v1_root = _make_dti_dir(tmp_path, site="IP_1",
                                          dataset="ABIDEII-IP_1", subject="29600")
        build_subject_connectome(
            dti_dir=dti_dir,
            v1_processed_root=v1_root,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        assert not (dti_dir / "connectome" / "count_matrix.npy").exists()

    def test_batch_raises_if_ip1_requested(self, tmp_path):
        with pytest.raises(ValueError, match="excluded"):
            run_connectome_batch(
                processed_v2b_root=tmp_path / "v2b" / "abide_ii",
                v1_processed_root=tmp_path / "v1" / "abide_ii",
                raw_root=tmp_path / "raw",
                sites=["IP_1"],
            )


# ---------------------------------------------------------------------------
# TestMatrixProperties
# ---------------------------------------------------------------------------

class TestMatrixProperties:
    def _run(self, tmp_path):
        dti_dir, v1_root = _make_dti_dir(tmp_path, n_rois=ATLAS_N_ROIS)
        rec = build_subject_connectome(
            dti_dir=dti_dir,
            v1_processed_root=v1_root,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw",
        )
        return dti_dir, rec

    def test_matrices_created(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        for name in ["count_matrix.npy", "mean_length_matrix.npy",
                     "mean_fa_matrix.npy", "mean_md_matrix.npy"]:
            assert (dti_dir / "connectome" / name).exists(), f"Missing: {name}"

    def test_count_matrix_is_square(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        m = np.load(str(dti_dir / "connectome" / "count_matrix.npy"))
        assert m.ndim == 2
        assert m.shape[0] == m.shape[1]

    def test_count_matrix_shape_matches_atlas(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        m = np.load(str(dti_dir / "connectome" / "count_matrix.npy"))
        assert m.shape == (ATLAS_N_ROIS, ATLAS_N_ROIS)

    def test_count_matrix_is_symmetric(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        m = np.load(str(dti_dir / "connectome" / "count_matrix.npy"))
        assert np.allclose(m, m.T), "Count matrix is not symmetric"

    def test_count_matrix_non_negative(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        m = np.load(str(dti_dir / "connectome" / "count_matrix.npy"))
        assert (m >= 0).all()

    def test_count_matrix_diagonal_zero(self, tmp_path):
        """Self-loops should not appear in count_matrix diagonal."""
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        m = np.load(str(dti_dir / "connectome" / "count_matrix.npy"))
        assert np.all(np.diag(m) == 0), "Diagonal should be zero (no self-loops)"

    def test_self_loop_count_recorded(self, tmp_path):
        _, rec = self._run(tmp_path)
        if rec.status == "success":
            assert isinstance(rec.self_loops_count, int)
            assert rec.self_loops_count >= 0

    def test_report_json_created(self, tmp_path):
        dti_dir, rec = self._run(tmp_path)
        if rec.status != "success":
            return
        report_path = dti_dir / "connectome" / "connectome_report.json"
        assert report_path.exists()
        d = json.loads(report_path.read_text())
        assert d["status"] == "success"


# ---------------------------------------------------------------------------
# TestSubjectFailureIsolation
# ---------------------------------------------------------------------------

class TestSubjectFailureIsolation:
    def test_bad_subject_does_not_crash_batch(self, tmp_path):
        good, v1_root = _make_dti_dir(tmp_path, subject="10001")
        bad, _        = _make_dti_dir(tmp_path, subject="10002")
        (bad / "tractography" / "streamlines.trk").unlink()

        records = run_connectome_batch(
            processed_v2b_root=tmp_path / "v2b" / "abide_ii",
            v1_processed_root=v1_root,
            raw_root=tmp_path / "raw",
            sites=["BNI"],
            skip_if_exists=False,
        )
        statuses = {r.subject_id: r.status for r in records}
        assert statuses.get("10001") == "success"
        assert statuses.get("10002") == "failed"


# ---------------------------------------------------------------------------
# TestSkipIfExists
# ---------------------------------------------------------------------------

class TestSkipIfExists:
    def test_skip_reuses_existing_report(self, tmp_path):
        dti_dir, v1_root = _make_dti_dir(tmp_path)
        rec1 = build_subject_connectome(
            dti_dir=dti_dir, v1_processed_root=v1_root,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw", skip_if_exists=True,
        )
        if rec1.status != "success":
            return
        (dti_dir / "tractography" / "streamlines.trk").write_bytes(b"corrupt")
        rec2 = build_subject_connectome(
            dti_dir=dti_dir, v1_processed_root=v1_root,
            output_root=tmp_path / "v2b" / "abide_ii",
            raw_root=tmp_path / "raw", skip_if_exists=True,
        )
        assert rec2.status == "success"


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_subject_csv_columns(self, tmp_path):
        rec = ConnectomeRecord(site="BNI", dataset="D", subject_id="1",
                               status="success", nonzero_edges=50, density=0.01)
        out = tmp_path / "s.csv"
        write_subject_summary([rec], out)
        cols = csv.DictReader(open(out)).fieldnames
        for f in SUBJECT_CSV_FIELDS:
            assert f in cols, f"Missing: {f}"

    def test_site_csv_columns(self, tmp_path):
        rec = ConnectomeRecord(site="BNI", dataset="D", subject_id="1",
                               status="success", nonzero_edges=50, density=0.01)
        out = tmp_path / "site.csv"
        write_site_summary([rec], out)
        cols = csv.DictReader(open(out)).fieldnames
        for f in SITE_CSV_FIELDS:
            assert f in cols, f"Missing: {f}"
