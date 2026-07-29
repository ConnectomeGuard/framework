from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofiber.registry.dti_validator import validate_dti


def _write_nifti(path: Path, shape: tuple) -> None:
    data = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _write_bval(path: Path, n: int) -> None:
    vals = np.zeros(n)
    vals[0] = 0      # b0
    vals[1:] = 1000  # DWI volumes
    np.savetxt(str(path), vals.reshape(1, -1), fmt="%g")


def _write_bvec(path: Path, n: int) -> None:
    # FSL convention: 3 rows × N cols
    vecs = np.random.default_rng(0).standard_normal((3, n))
    np.savetxt(str(path), vecs, fmt="%.6f")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dti_triplet(tmp_path):
    """Return (dti_path, bval_path, bvec_path) for a valid 30-volume dataset."""
    n = 30
    dti  = tmp_path / "dti.nii.gz"
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_nifti(dti, (64, 64, 30, n))
    _write_bval(bval, n)
    _write_bvec(bvec, n)
    return dti, bval, bvec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_dti_returns_correct_counts(dti_triplet):
    dti, bval, bvec = dti_triplet
    result = validate_dti(dti, bval, bvec)
    assert result.status == "valid"
    assert result.volume_count == 30
    assert result.bval_count == 30
    assert result.bvec_count == 30
    assert result.errors == []


def test_bval_mismatch_detected(tmp_path):
    dti  = tmp_path / "dti.nii.gz"
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_nifti(dti, (64, 64, 30, 30))
    _write_bval(bval, 28)          # wrong count
    _write_bvec(bvec, 30)
    result = validate_dti(dti, bval, bvec)
    assert result.status == "dti_mismatch"
    assert any("bval" in e for e in result.errors)


def test_bvec_mismatch_detected(tmp_path):
    dti  = tmp_path / "dti.nii.gz"
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_nifti(dti, (64, 64, 30, 30))
    _write_bval(bval, 30)
    _write_bvec(bvec, 25)          # wrong count
    result = validate_dti(dti, bval, bvec)
    assert result.status == "dti_mismatch"
    assert any("bvec" in e for e in result.errors)


def test_missing_dti_file_returns_failed(tmp_path):
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_bval(bval, 30)
    _write_bvec(bvec, 30)
    result = validate_dti(tmp_path / "dti.nii.gz", bval, bvec)
    assert result.status == "failed"
    assert result.volume_count is None


def test_3d_nifti_counts_as_one_volume(tmp_path):
    dti  = tmp_path / "dti.nii.gz"
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_nifti(dti, (64, 64, 30))   # 3D
    _write_bval(bval, 1)
    np.savetxt(str(bvec), np.array([[0.0], [0.0], [0.0]]), fmt="%.1f")
    result = validate_dti(dti, bval, bvec)
    assert result.volume_count == 1
    assert result.status == "valid"


def test_bvec_transposed_format(tmp_path):
    """Accept bvec stored as N×3 (transposed) instead of 3×N."""
    n = 20
    dti  = tmp_path / "dti.nii.gz"
    bval = tmp_path / "dti.bval"
    bvec = tmp_path / "dti.bvec"
    _write_nifti(dti, (10, 10, 10, n))
    _write_bval(bval, n)
    # Write as N×3 (transposed — some tools use this)
    vecs = np.random.default_rng(1).standard_normal((n, 3))
    np.savetxt(str(bvec), vecs, fmt="%.6f")
    result = validate_dti(dti, bval, bvec)
    assert result.bvec_count == n
    assert result.status == "valid"
