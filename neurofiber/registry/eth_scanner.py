"""
Phase 1.3 — ABIDE II Site Scanner (ETH + general)

Scans any ABIDE II site directory, inspects every modality's NIfTI properties,
validates DTI consistency when present, and returns one SubjectScanResult per
subject.

Key design choice: DTI is treated as *optional* — sites without DTI (e.g. ETH)
are fully supported and will carry None for all DTI fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mismatch classification
# ---------------------------------------------------------------------------

class MismatchType:
    EXTRA_TRAILING_VOLUME = "extra_trailing_volume"   # volumes == bvals+1 == bvecs+1
    MISSING_BVAL_BVEC     = "missing_bval_bvec"       # DTI NIfTI exists, bval/bvec missing
    MALFORMED_BVEC        = "malformed_bvec"           # bvec rows ≠ 3 and ≠ N
    UNKNOWN               = "unknown_mismatch"

    @classmethod
    def classify(
        cls,
        n_vols: int,
        n_bvals: Optional[int],
        n_bvecs: Optional[int],
    ) -> str:
        if n_bvals is None or n_bvecs is None:
            return cls.MISSING_BVAL_BVEC
        if n_vols == n_bvals + 1 and n_bvals == n_bvecs:
            return cls.EXTRA_TRAILING_VOLUME
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubjectScanResult:
    subject_id: str
    site: str
    dataset: str
    # --- modality presence ---
    anat_exists: bool
    dti_exists: bool
    bval_exists: bool
    bvec_exists: bool
    rest_exists: bool
    # --- anat ---
    anat_shape: Optional[str] = field(default=None)
    anat_voxel_size_mm: Optional[str] = field(default=None)
    # --- resting-state ---
    rest_shape: Optional[str] = field(default=None)
    rest_volume_count: Optional[int] = field(default=None)
    rest_voxel_size_mm: Optional[str] = field(default=None)
    # --- DTI ---
    dti_shape: Optional[str] = field(default=None)
    dti_volume_count: Optional[int] = field(default=None)
    dti_voxel_size_mm: Optional[str] = field(default=None)
    bval_count: Optional[int] = field(default=None)
    bvec_count: Optional[int] = field(default=None)
    unique_bvalues: Optional[str] = field(default=None)
    n_b0_volumes: Optional[int] = field(default=None)
    n_dwi_volumes: Optional[int] = field(default=None)
    # --- DTI consistency ---
    dti_consistent: Optional[bool] = field(default=None)
    dti_mismatch_type: Optional[str] = field(default=None)
    b0_mean_signal: Optional[float] = field(default=None)
    last_volume_mean_signal: Optional[float] = field(default=None)
    signal_ratio: Optional[float] = field(default=None)
    recommended_action: Optional[str] = field(default=None)
    # --- overall ---
    validation_status: str = field(default="unknown")
    missing_modalities: str = field(default="")   # comma-separated

    def to_dict(self) -> dict:
        return {
            "subject_id":              self.subject_id,
            "site":                    self.site,
            "dataset":                 self.dataset,
            "anat_exists":             self.anat_exists,
            "dti_exists":              self.dti_exists,
            "bval_exists":             self.bval_exists,
            "bvec_exists":             self.bvec_exists,
            "rest_exists":             self.rest_exists,
            "anat_shape":              self.anat_shape,
            "anat_voxel_size_mm":      self.anat_voxel_size_mm,
            "rest_shape":              self.rest_shape,
            "rest_volume_count":       self.rest_volume_count,
            "rest_voxel_size_mm":      self.rest_voxel_size_mm,
            "dti_shape":               self.dti_shape,
            "dti_volume_count":        self.dti_volume_count,
            "dti_voxel_size_mm":       self.dti_voxel_size_mm,
            "bval_count":              self.bval_count,
            "bvec_count":              self.bvec_count,
            "unique_bvalues":          self.unique_bvalues,
            "n_b0_volumes":            self.n_b0_volumes,
            "n_dwi_volumes":           self.n_dwi_volumes,
            "dti_consistent":          self.dti_consistent,
            "dti_mismatch_type":       self.dti_mismatch_type,
            "b0_mean_signal":          self.b0_mean_signal,
            "last_volume_mean_signal": self.last_volume_mean_signal,
            "signal_ratio":            self.signal_ratio,
            "recommended_action":      self.recommended_action,
            "validation_status":       self.validation_status,
            "missing_modalities":      self.missing_modalities,
        }


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_site(
    dataset_root: Path | str,
    site: str,
    dataset: str,
    session: str = "session_1",
) -> list[SubjectScanResult]:
    """Scan every subject directory under *dataset_root* and return one result each."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    logger.info("Found %d subject directories in %s", len(subject_dirs), root)

    results: list[SubjectScanResult] = []
    for subject_dir in subject_dirs:
        r = _scan_one(subject_dir, site=site, dataset=dataset, session=session)
        results.append(r)
        logger.debug("  %-10s  status=%-20s  missing=%s",
                     r.subject_id, r.validation_status, r.missing_modalities or "none")
    return results


# ---------------------------------------------------------------------------
# Single-subject helpers
# ---------------------------------------------------------------------------

def _scan_one(
    subject_dir: Path,
    site: str,
    dataset: str,
    session: str,
) -> SubjectScanResult:
    sid = subject_dir.name
    sess = subject_dir / session

    anat_path = sess / "anat_1" / "anat.nii.gz"
    dti_path  = sess / "dti_1"  / "dti.nii.gz"
    bval_path = sess / "dti_1"  / "dti.bval"
    bvec_path = sess / "dti_1"  / "dti.bvec"
    rest_path = sess / "rest_1" / "rest.nii.gz"

    r = SubjectScanResult(
        subject_id=sid, site=site, dataset=dataset,
        anat_exists=anat_path.exists(),
        dti_exists=dti_path.exists(),
        bval_exists=bval_path.exists(),
        bvec_exists=bvec_path.exists(),
        rest_exists=rest_path.exists(),
    )

    missing = [
        name for name, flag in [
            ("anat", r.anat_exists), ("dti", r.dti_exists),
            ("bval", r.bval_exists), ("bvec", r.bvec_exists),
            ("rest", r.rest_exists),
        ] if not flag
    ]
    r.missing_modalities = ", ".join(missing)

    # --- inspect existing modalities ---
    if r.anat_exists:
        _fill_anat(r, anat_path)
    if r.rest_exists:
        _fill_rest(r, rest_path)
    if r.dti_exists:
        _fill_dti(r, dti_path, bval_path if r.bval_exists else None,
                  bvec_path if r.bvec_exists else None)

    # --- set overall validation_status ---
    r.validation_status = _derive_status(r)
    return r


def _fill_anat(r: SubjectScanResult, path: Path) -> None:
    try:
        img = nib.load(str(path))
        r.anat_shape = str(img.shape)
        r.anat_voxel_size_mm = _fmt_zooms(img.header.get_zooms())
    except Exception as exc:
        logger.warning("[%s] anat load error: %s", r.subject_id, exc)


def _fill_rest(r: SubjectScanResult, path: Path) -> None:
    try:
        img = nib.load(str(path))
        r.rest_shape = str(img.shape)
        r.rest_volume_count = img.shape[3] if img.ndim == 4 else 1
        r.rest_voxel_size_mm = _fmt_zooms(img.header.get_zooms()[:3])
    except Exception as exc:
        logger.warning("[%s] rest load error: %s", r.subject_id, exc)


def _fill_dti(
    r: SubjectScanResult,
    dti_path: Path,
    bval_path: Optional[Path],
    bvec_path: Optional[Path],
) -> None:
    # --- NIfTI ---
    try:
        img = nib.load(str(dti_path))
        r.dti_shape = str(img.shape)
        r.dti_volume_count = img.shape[3] if img.ndim == 4 else 1
        r.dti_voxel_size_mm = _fmt_zooms(img.header.get_zooms()[:3])
    except Exception as exc:
        logger.warning("[%s] dti load error: %s", r.subject_id, exc)
        r.dti_mismatch_type = "failed_to_load"
        r.validation_status = "dti_failed"
        return

    # --- bval ---
    if bval_path:
        try:
            bvals = np.loadtxt(str(bval_path))
            r.bval_count = int(bvals.size)
            unique_bv = sorted(set(int(round(v, -1)) for v in bvals.flat))
            r.unique_bvalues = str(unique_bv)
            r.n_b0_volumes  = int(np.sum(bvals < 100))
            r.n_dwi_volumes = int(np.sum(bvals >= 100))
        except Exception as exc:
            logger.warning("[%s] bval load error: %s", r.subject_id, exc)

    # --- bvec ---
    if bvec_path:
        try:
            bvecs = np.loadtxt(str(bvec_path))
            if bvecs.ndim == 1:
                r.bvec_count = 1
            elif bvecs.shape[0] == 3:
                r.bvec_count = int(bvecs.shape[1])   # FSL: 3×N
            else:
                r.bvec_count = int(bvecs.shape[0])   # N×3
        except Exception as exc:
            logger.warning("[%s] bvec load error: %s", r.subject_id, exc)

    # --- consistency ---
    n_vols = r.dti_volume_count
    n_bvals = r.bval_count
    n_bvecs = r.bvec_count

    if n_bvals is not None and n_bvecs is not None:
        r.dti_consistent = (n_vols == n_bvals == n_bvecs)
        if not r.dti_consistent:
            r.dti_mismatch_type = MismatchType.classify(n_vols, n_bvals, n_bvecs)
            _compute_signal_ratio(r, img)
            r.recommended_action = _recommend(r.dti_mismatch_type, r.signal_ratio)
    else:
        r.dti_consistent = False
        r.dti_mismatch_type = MismatchType.MISSING_BVAL_BVEC


def _compute_signal_ratio(r: SubjectScanResult, img: nib.Nifti1Image) -> None:
    try:
        fdata = img.get_fdata(dtype=np.float32)
        b0_mean   = float(fdata[..., 0].mean())
        last_mean = float(fdata[..., -1].mean())
        r.b0_mean_signal          = round(b0_mean, 4)
        r.last_volume_mean_signal = round(last_mean, 4)
        r.signal_ratio            = round(last_mean / b0_mean, 6) if b0_mean > 0 else None
    except Exception as exc:
        logger.warning("[%s] signal ratio computation failed: %s", r.subject_id, exc)


def _recommend(mismatch_type: Optional[str], signal_ratio: Optional[float]) -> str:
    if mismatch_type == MismatchType.EXTRA_TRAILING_VOLUME:
        if signal_ratio is not None and signal_ratio < 0.2:
            return "strip_last_volume"
        return "manual_review_signal_ratio_above_threshold"
    if mismatch_type == MismatchType.MISSING_BVAL_BVEC:
        return "locate_or_reconstruct_bval_bvec"
    if mismatch_type == MismatchType.MALFORMED_BVEC:
        return "inspect_bvec_format"
    return "manual_review"


def _derive_status(r: SubjectScanResult) -> str:
    if not r.anat_exists:
        return "missing_anat"
    if r.dti_exists:
        if r.dti_consistent is True:
            return "valid"
        if r.dti_consistent is False:
            return "dti_mismatch"
        return "dti_failed"
    # DTI absent — this site may not have a DTI protocol.
    # Only flag missing_files if rest is also absent (rest is universal).
    if not r.rest_exists:
        return "missing_files"
    return "valid"


def _fmt_zooms(zooms) -> str:
    return "×".join(f"{z:.3f}" for z in zooms)
