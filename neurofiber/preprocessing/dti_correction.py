"""
Phase 2.1 — BNI DTI Volume Correction

The ABIDE II BNI dataset stores 34 DTI volumes per subject but only 33 bval/bvec
entries. The 34th volume is a low-signal trailing calibration/noise frame that must
be removed before tensor fitting or tractography.

Correction logic:
  - If signal_ratio = mean(last_vol) / mean(vol_0) < threshold (default 0.2):
      strip last volume, save dti_corrected.nii.gz, copy bval/bvec unchanged.
  - If signal_ratio >= threshold:
      mark requires_manual_review (do not modify data).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_RAW_DIR_MARKER = "data/raw"


def _guard_no_raw_write(output_dir: Path, raw_path: Path) -> None:
    """Raise immediately if output_dir is inside (or equal to) the raw data tree."""
    try:
        output_dir.resolve().relative_to(raw_path.resolve())
        raise ValueError(
            f"SAFETY: output_dir '{output_dir}' is inside the raw data tree "
            f"'{raw_path}'. Raw data must never be overwritten. "
            f"Use a path under data/processed/ instead."
        )
    except ValueError as exc:
        if "SAFETY" in str(exc):
            raise
        # relative_to() raised because output_dir is NOT inside raw_path — safe
        pass

DEFAULT_SIGNAL_RATIO_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRecord:
    subject_id: str
    original_volume_count: int
    corrected_volume_count: Optional[int]
    bval_count: int
    bvec_count: int
    b0_mean_signal: float
    last_volume_mean_signal: float
    signal_ratio: float
    action: str          # "volume_removed" | "requires_manual_review" | "failed"
    status: str          # "success" | "failed"
    original_dti_path: str
    corrected_dti_path: Optional[str]
    error_message: Optional[str] = field(default=None)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        return {
            "subject_id":               self.subject_id,
            "original_volume_count":    self.original_volume_count,
            "corrected_volume_count":   self.corrected_volume_count,
            "bval_count":               self.bval_count,
            "bvec_count":               self.bvec_count,
            "b0_mean_signal":           round(self.b0_mean_signal, 4),
            "last_volume_mean_signal":  round(self.last_volume_mean_signal, 4),
            "signal_ratio":             round(self.signal_ratio, 6),
            "action":                   self.action,
            "status":                   self.status,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class DTICorrectionError(Exception):
    pass


def _load_and_validate(
    dti_path: Path,
    bval_path: Path,
    bvec_path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, int, int, int]:
    """Load and validate inputs. Returns (img, bvals, bvecs, n_vols, n_bvals, n_bvecs)."""
    img = nib.load(str(dti_path))

    if img.ndim != 4:
        raise DTICorrectionError(f"DTI must be 4D, got shape {img.shape}")

    n_vols = img.shape[3]

    bvals = np.loadtxt(str(bval_path))
    n_bvals = int(bvals.size)

    bvecs = np.loadtxt(str(bvec_path))
    if bvecs.ndim == 1:
        n_bvecs = 1
    elif bvecs.shape[0] == 3:
        n_bvecs = int(bvecs.shape[1])  # FSL: 3 × N
    else:
        n_bvecs = int(bvecs.shape[0])  # N × 3

    if n_bvals != n_bvecs:
        raise DTICorrectionError(
            f"bval count ({n_bvals}) ≠ bvec count ({n_bvecs})"
        )
    # Accept both clean data (V=B=G) and BNI-style trailing volume (V=B+1=G+1)
    if n_vols == n_bvals:
        pass   # consistent — no correction needed
    elif n_vols == n_bvals + 1:
        pass   # trailing calibration volume — will check signal ratio
    else:
        raise DTICorrectionError(
            f"Volume count ({n_vols}) vs bval/bvec count ({n_bvals}) "
            f"differs by more than 1 — cannot auto-correct"
        )

    return img, bvals, bvecs, n_vols, n_bvals, n_bvecs


# ---------------------------------------------------------------------------
# Core correction function
# ---------------------------------------------------------------------------

def correct_dti_volumes(
    dti_path: Path,
    bval_path: Path,
    bvec_path: Path,
    output_dir: Path,
    signal_ratio_threshold: float = DEFAULT_SIGNAL_RATIO_THRESHOLD,
) -> CorrectionRecord:
    """
    Validate, correct, and save a single subject's DTI data.

    Writes to output_dir:
      dti_corrected.nii.gz  (if volume_removed)
      dti.bval              (copy of original)
      dti.bvec              (copy of original)
      correction_report.json

    Returns a CorrectionRecord describing what was done.
    """
    subject_id = dti_path.parent.parent.parent.name
    _guard_no_raw_write(output_dir, dti_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load & validate ---
    try:
        img, bvals, bvecs, n_vols, n_bvals, n_bvecs = _load_and_validate(
            dti_path, bval_path, bvec_path
        )
    except Exception as exc:
        rec = _failed_record(subject_id, str(dti_path), str(exc))
        _write_report(rec, output_dir)
        return rec

    # --- already consistent: copy original as dti_corrected for pipeline continuity ---
    if n_vols == n_bvals:
        corrected_path = output_dir / "dti_corrected.nii.gz"
        shutil.copy2(dti_path,  corrected_path)
        shutil.copy2(bval_path, output_dir / "dti.bval")
        shutil.copy2(bvec_path, output_dir / "dti.bvec")
        action            = "no_correction_needed"
        corrected_vol_cnt = n_vols
        b0_mean = last_mean = ratio = 0.0
        logger.info("[%s] volumes=%d  bvals=%d  already consistent — copied as dti_corrected",
                    subject_id, n_vols, n_bvals)
    else:
        # --- trailing volume: check signal ratio ---
        fdata = img.get_fdata(dtype=np.float32)
        b0_mean   = float(fdata[..., 0].mean())
        last_mean = float(fdata[..., -1].mean())
        ratio     = last_mean / b0_mean if b0_mean > 0 else float("inf")

        logger.info(
            "[%s] volumes=%d  bvals=%d  B0_mean=%.2f  last_mean=%.2f  ratio=%.4f",
            subject_id, n_vols, n_bvals, b0_mean, last_mean, ratio,
        )

        if ratio < signal_ratio_threshold:
            corrected_path = output_dir / "dti_corrected.nii.gz"
            _strip_last_volume(img, corrected_path)
            shutil.copy2(bval_path, output_dir / "dti.bval")
            shutil.copy2(bvec_path, output_dir / "dti.bvec")
            action            = "volume_removed"
            corrected_vol_cnt = n_vols - 1
            logger.info("[%s] → volume_removed  saved %s", subject_id, corrected_path)
        else:
            corrected_path    = None
            action            = "requires_manual_review"
            corrected_vol_cnt = None
            logger.warning("[%s] → requires_manual_review  ratio=%.4f >= %.2f",
                           subject_id, ratio, signal_ratio_threshold)

    rec = CorrectionRecord(
        subject_id=subject_id,
        original_volume_count=n_vols,
        corrected_volume_count=corrected_vol_cnt,
        bval_count=n_bvals,
        bvec_count=n_bvecs,
        b0_mean_signal=b0_mean,
        last_volume_mean_signal=last_mean,
        signal_ratio=ratio,
        action=action,
        status="success",
        original_dti_path=str(dti_path),
        corrected_dti_path=str(corrected_path) if corrected_path else None,
    )
    _write_report(rec, output_dir)
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_correction_batch(
    dataset_root: Path,
    output_root: Path,
    session: str = "session_1",
    signal_ratio_threshold: float = DEFAULT_SIGNAL_RATIO_THRESHOLD,
) -> list[CorrectionRecord]:
    """
    Run volume correction on every subject directory under dataset_root.

    Expected input layout:
      <dataset_root>/<subject_id>/<session>/dti_1/
        dti.nii.gz  dti.bval  dti.bvec

    Output layout mirrors the input under output_root.
    """
    _guard_no_raw_write(output_root, dataset_root)
    subject_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    logger.info("Processing %d subjects in %s …", len(subject_dirs), dataset_root)

    records: list[CorrectionRecord] = []
    for subject_dir in subject_dirs:
        subject_id  = subject_dir.name
        session_dir = subject_dir / session / "dti_1"

        dti_path  = session_dir / "dti.nii.gz"
        bval_path = session_dir / "dti.bval"
        bvec_path = session_dir / "dti.bvec"

        if not all(p.exists() for p in [dti_path, bval_path, bvec_path]):
            missing = [p.name for p in [dti_path, bval_path, bvec_path] if not p.exists()]
            logger.warning("[%s] skipping — missing files: %s", subject_id, missing)
            records.append(_failed_record(
                subject_id, str(dti_path),
                f"Missing input files: {missing}",
            ))
            continue

        out_dir = output_root / subject_id / session / "dti_1"
        rec = correct_dti_volumes(
            dti_path, bval_path, bvec_path,
            output_dir=out_dir,
            signal_ratio_threshold=signal_ratio_threshold,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_last_volume(img: nib.Nifti1Image, output_path: Path) -> None:
    raw = np.asanyarray(img.dataobj)[..., :-1]
    corrected = nib.Nifti1Image(raw, img.affine, img.header)
    nib.save(corrected, str(output_path))


def _write_report(rec: CorrectionRecord, output_dir: Path) -> None:
    report_path = output_dir / "correction_report.json"
    report_path.write_text(json.dumps(rec.to_dict(), indent=2))


def _failed_record(subject_id: str, dti_path: str, error: str) -> CorrectionRecord:
    logger.error("[%s] failed: %s", subject_id, error)
    return CorrectionRecord(
        subject_id=subject_id,
        original_volume_count=0,
        corrected_volume_count=None,
        bval_count=0,
        bvec_count=0,
        b0_mean_signal=0.0,
        last_volume_mean_signal=0.0,
        signal_ratio=0.0,
        action="failed",
        status="failed",
        original_dti_path=dti_path,
        corrected_dti_path=None,
        error_message=error,
    )
