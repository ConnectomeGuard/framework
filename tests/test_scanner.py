from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofiber.registry.scanner import scan_subjects


# ---------------------------------------------------------------------------
# Helpers — build fake ABIDE II directory trees
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, shape=(10, 10, 10, 5)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), str(path))


def _write_bval(path: Path, n: int) -> None:
    np.savetxt(str(path), np.zeros((1, n)), fmt="%g")


def _write_bvec(path: Path, n: int) -> None:
    np.savetxt(str(path), np.zeros((3, n)), fmt="%.1f")


def _make_subject(root: Path, subject_id: str, *, missing: list[str] | None = None) -> Path:
    """
    Create a full ABIDE II subject tree under *root*.
    Pass *missing* to omit specific files: "anat", "dti", "bval", "bvec", "rest".
    """
    missing = missing or []
    session = root / subject_id / "session_1"

    if "anat" not in missing:
        _write_nifti(session / "anat_1" / "anat.nii.gz", (10, 10, 10))

    n = 5
    if "dti" not in missing:
        _write_nifti(session / "dti_1" / "dti.nii.gz", (10, 10, 10, n))
    if "bval" not in missing:
        (session / "dti_1").mkdir(parents=True, exist_ok=True)
        _write_bval(session / "dti_1" / "dti.bval", n)
    if "bvec" not in missing:
        (session / "dti_1").mkdir(parents=True, exist_ok=True)
        _write_bvec(session / "dti_1" / "dti.bvec", n)

    if "rest" not in missing:
        _write_nifti(session / "rest_1" / "rest.nii.gz", (10, 10, 10, 50))

    return root / subject_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scan_finds_all_subjects(tmp_path):
    for sid in ["sub-001", "sub-002", "sub-003"]:
        _make_subject(tmp_path, sid)
    records = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")
    assert len(records) == 3
    assert {r.subject_id for r in records} == {"sub-001", "sub-002", "sub-003"}


def test_valid_subject_has_correct_status(tmp_path):
    _make_subject(tmp_path, "sub-001")
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.validation_status == "valid"
    assert rec.anat_exists
    assert rec.dti_exists
    assert rec.bval_exists
    assert rec.bvec_exists
    assert rec.rest_exists


def test_dti_counts_populated_for_valid_subject(tmp_path):
    _make_subject(tmp_path, "sub-001")
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.dti_volume_count == 5
    assert rec.bval_count == 5
    assert rec.bvec_count == 5


def test_missing_anat_detected(tmp_path):
    _make_subject(tmp_path, "sub-001", missing=["anat"])
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.validation_status == "missing_files"
    assert not rec.anat_exists
    assert rec.dti_exists


def test_missing_rest_detected(tmp_path):
    _make_subject(tmp_path, "sub-001", missing=["rest"])
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.validation_status == "missing_files"
    assert not rec.rest_exists


def test_missing_bval_and_bvec_detected(tmp_path):
    _make_subject(tmp_path, "sub-001", missing=["bval", "bvec"])
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.validation_status == "missing_files"
    assert not rec.bval_exists
    assert not rec.bvec_exists
    assert rec.dti_volume_count is None   # DTI check skipped


def test_dti_mismatch_detected(tmp_path):
    session = tmp_path / "sub-bad" / "session_1"
    _write_nifti(session / "anat_1" / "anat.nii.gz", (10, 10, 10))
    _write_nifti(session / "dti_1"  / "dti.nii.gz",  (10, 10, 10, 5))
    _write_bval(session / "dti_1" / "dti.bval", 3)   # wrong count
    _write_bvec(session / "dti_1" / "dti.bvec", 5)
    _write_nifti(session / "rest_1" / "rest.nii.gz", (10, 10, 10, 50))
    rec = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")[0]
    assert rec.validation_status == "dti_mismatch"
    assert rec.dti_volume_count == 5
    assert rec.bval_count == 3


def test_site_and_dataset_fields_set(tmp_path):
    _make_subject(tmp_path, "sub-001")
    rec = scan_subjects(tmp_path, site="MIT", dataset="ABIDE_II")[0]
    assert rec.site == "MIT"
    assert rec.dataset == "ABIDE_II"


def test_nonexistent_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_subjects(tmp_path / "does_not_exist", site="BNI", dataset="ABIDE_II")


def test_mixed_subjects_summary(tmp_path):
    _make_subject(tmp_path, "sub-valid")
    _make_subject(tmp_path, "sub-no-rest",  missing=["rest"])
    _make_subject(tmp_path, "sub-no-anat",  missing=["anat"])
    records = scan_subjects(tmp_path, site="BNI", dataset="ABIDE_II")
    statuses = {r.subject_id: r.validation_status for r in records}
    assert statuses["sub-valid"]   == "valid"
    assert statuses["sub-no-rest"] == "missing_files"
    assert statuses["sub-no-anat"] == "missing_files"
