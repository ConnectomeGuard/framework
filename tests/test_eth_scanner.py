"""Tests for Phase 1.3 — ETH site scanner and metadata loader."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofiber.registry.eth_scanner import scan_site, SubjectScanResult, MismatchType
from neurofiber.registry.metadata_loader import load_abide2_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_nifti(path: Path, shape: tuple, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=dtype), np.eye(4)), str(path))


def _make_bval(path: Path, bvals: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), np.array(bvals).reshape(1, -1), fmt="%g")


def _make_bvec(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vecs = np.zeros((3, n))
    vecs[0, 1:] = 1.0
    np.savetxt(str(path), vecs, fmt="%.4f")


def _make_eth_subject(root: Path, sid: str, *, missing: list[str] | None = None) -> None:
    """Build a full ETH-style subject tree (anat + rest, no DTI by default)."""
    missing = missing or []
    sess = root / sid / "session_1"
    if "anat" not in missing:
        _make_nifti(sess / "anat_1" / "anat.nii.gz", (256, 256, 180))
    if "rest" not in missing:
        _make_nifti(sess / "rest_1" / "rest.nii.gz", (80, 80, 40, 210))


def _make_dti_subject(root: Path, sid: str, n_vols: int = 33,
                      n_bvals: int = 33, n_bvecs: int = 33,
                      b0_signal: float = 100.0, last_signal: float = 5.0) -> None:
    """Build a subject tree WITH DTI.  Vol 0 = b0_signal, vol[-1] = last_signal."""
    sess = root / sid / "session_1"
    _make_nifti(sess / "anat_1" / "anat.nii.gz", (64, 64, 30))
    _make_nifti(sess / "rest_1" / "rest.nii.gz",  (32, 32, 20, 50))
    # Build DTI with realistic signal levels
    data = np.full((8, 8, 4, n_vols), 30.0, dtype=np.float32)
    data[..., 0]  = b0_signal
    data[..., -1] = last_signal
    dti_path = sess / "dti_1" / "dti.nii.gz"
    dti_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dti_path))
    bvals = [0.0] + [2500.0] * (n_bvals - 1)
    _make_bval(sess / "dti_1" / "dti.bval", bvals)
    _make_bvec(sess / "dti_1" / "dti.bvec", n_bvecs)


def _make_metadata_csv(path: Path, subjects: list[dict]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subjects).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# scan_site — ETH-style (no DTI)
# ---------------------------------------------------------------------------

class TestScanSiteNoDTI:

    def test_finds_all_subjects(self, tmp_path):
        for sid in ["29057", "29058", "29059"]:
            _make_eth_subject(tmp_path, sid)
        results = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")
        assert len(results) == 3

    def test_valid_subject_status(self, tmp_path):
        _make_eth_subject(tmp_path, "29057")
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.validation_status == "valid"
        assert r.anat_exists
        assert r.rest_exists
        assert not r.dti_exists
        assert r.missing_modalities == "dti, bval, bvec"

    def test_anat_shape_recorded(self, tmp_path):
        _make_eth_subject(tmp_path, "29057")
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.anat_shape == str((256, 256, 180))

    def test_rest_volume_count_recorded(self, tmp_path):
        _make_eth_subject(tmp_path, "29057")
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.rest_volume_count == 210

    def test_missing_rest_detected(self, tmp_path):
        _make_eth_subject(tmp_path, "29057", missing=["rest"])
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.validation_status == "missing_files"
        assert not r.rest_exists
        assert "rest" in r.missing_modalities

    def test_missing_anat_status(self, tmp_path):
        _make_eth_subject(tmp_path, "29057", missing=["anat"])
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.validation_status == "missing_anat"

    def test_dti_fields_none_when_no_dti(self, tmp_path):
        _make_eth_subject(tmp_path, "29057")
        r = scan_site(tmp_path, site="ETH", dataset="ABIDE_II")[0]
        assert r.dti_volume_count is None
        assert r.bval_count is None
        assert r.bvec_count is None
        assert r.signal_ratio is None
        assert r.recommended_action is None

    def test_nonexistent_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_site(tmp_path / "no_such_dir", site="ETH", dataset="ABIDE_II")


# ---------------------------------------------------------------------------
# scan_site — WITH DTI (consistency tests)
# ---------------------------------------------------------------------------

class TestScanSiteWithDTI:

    def test_consistent_dti_valid(self, tmp_path):
        _make_dti_subject(tmp_path, "sub-001", n_vols=33, n_bvals=33, n_bvecs=33)
        r = scan_site(tmp_path, site="BNI", dataset="ABIDE_II")[0]
        assert r.dti_consistent is True
        assert r.validation_status == "valid"
        assert r.dti_volume_count == 33

    def test_extra_trailing_volume_classified(self, tmp_path):
        _make_dti_subject(tmp_path, "sub-001", n_vols=34, n_bvals=33, n_bvecs=33)
        r = scan_site(tmp_path, site="BNI", dataset="ABIDE_II")[0]
        assert r.dti_consistent is False
        assert r.dti_mismatch_type == MismatchType.EXTRA_TRAILING_VOLUME
        assert r.validation_status == "dti_mismatch"
        assert r.signal_ratio is not None

    def test_bval_bvec_counts_recorded(self, tmp_path):
        _make_dti_subject(tmp_path, "sub-001", n_vols=33)
        r = scan_site(tmp_path, site="BNI", dataset="ABIDE_II")[0]
        assert r.bval_count == 33
        assert r.bvec_count == 33

    def test_b0_and_dwi_counts(self, tmp_path):
        _make_dti_subject(tmp_path, "sub-001", n_vols=33)
        r = scan_site(tmp_path, site="BNI", dataset="ABIDE_II")[0]
        assert r.n_b0_volumes == 1
        assert r.n_dwi_volumes == 32

    def test_strip_last_volume_recommended(self, tmp_path):
        """Signal ratio < 0.2 should recommend strip_last_volume."""
        sess = tmp_path / "sub-001" / "session_1"
        data = np.full((8, 8, 4, 34), 50.0, np.float32)
        data[..., 0]  = 100.0   # B0
        data[..., -1] = 5.0     # trailing (ratio=0.05 < 0.2)
        sess.mkdir(parents=True)
        _make_nifti(sess / "anat_1" / "anat.nii.gz", (8, 8, 4))
        path = sess / "dti_1" / "dti.nii.gz"
        path.parent.mkdir()
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        _make_bval(sess / "dti_1" / "dti.bval", [0] + [2500]*32)
        _make_bvec(sess / "dti_1" / "dti.bvec", 33)
        _make_nifti(sess / "rest_1" / "rest.nii.gz", (8, 8, 4, 20))

        r = scan_site(tmp_path, site="BNI", dataset="ABIDE_II")[0]
        assert r.recommended_action == "strip_last_volume"


# ---------------------------------------------------------------------------
# load_abide2_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:

    def _eth_csv(self, tmp_path: Path) -> Path:
        data = [
            {"SITE_ID": "ABIDEII-ETH_1", "SUB_ID": 29057, "DX_GROUP": 1,
             "AGE_AT_SCAN": 27.25, "SEX": 1, "FIQ": 112},
            {"SITE_ID": "ABIDEII-ETH_1", "SUB_ID": 29058, "DX_GROUP": 2,
             "AGE_AT_SCAN": 18.5, "SEX": 1, "FIQ": 98},
        ]
        path = tmp_path / "meta.csv"
        _make_metadata_csv(path, data)
        return path

    def test_load_returns_correct_count(self, tmp_path):
        meta = load_abide2_metadata(self._eth_csv(tmp_path), site="ETH")
        assert len(meta) == 2

    def test_diagnosis_mapped(self, tmp_path):
        meta = load_abide2_metadata(self._eth_csv(tmp_path), site="ETH")
        assert meta["29057"].diagnosis == "ASD"
        assert meta["29058"].diagnosis == "CONTROL"

    def test_sex_mapped(self, tmp_path):
        meta = load_abide2_metadata(self._eth_csv(tmp_path), site="ETH")
        assert meta["29057"].sex == "M"

    def test_age_parsed(self, tmp_path):
        meta = load_abide2_metadata(self._eth_csv(tmp_path), site="ETH")
        assert meta["29057"].age == pytest.approx(27.25)

    def test_fiq_loaded(self, tmp_path):
        meta = load_abide2_metadata(self._eth_csv(tmp_path), site="ETH")
        assert meta["29057"].fiq == 112.0

    def test_missing_col_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        import pandas as pd
        pd.DataFrame([{"SUB_ID": 1, "DX_GROUP": 1}]).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_abide2_metadata(bad, site="ETH")


# ---------------------------------------------------------------------------
# MismatchType.classify
# ---------------------------------------------------------------------------

class TestMismatchClassify:

    def test_extra_trailing(self):
        assert MismatchType.classify(34, 33, 33) == MismatchType.EXTRA_TRAILING_VOLUME

    def test_missing_bval_bvec(self):
        assert MismatchType.classify(34, None, None) == MismatchType.MISSING_BVAL_BVEC

    def test_unknown(self):
        assert MismatchType.classify(34, 30, 33) == MismatchType.UNKNOWN
