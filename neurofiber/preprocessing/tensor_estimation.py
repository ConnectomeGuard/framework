"""
Phase 2.2 — DTI Tensor Estimation

Fits the diffusion tensor model (WLS) to Phase 2.1 corrected BNI data and
produces voxel-wise scalar maps: FA, MD, AD, RD.

Pipeline per subject:
  1. Load dti_corrected.nii.gz + dti.bval + dti.bvec
  2. Validate (4D, counts match, ≥1 B0, ≥30 DWI directions)
  3. Build DIPY gradient table (transpose bvecs from FSL 3×N → DIPY N×3)
  4. Derive a threshold mask from the mean B0 signal (30th percentile)
  5. Fit TensorModel (WLS) within the mask
  6. Save FA.nii.gz  MD.nii.gz  AD.nii.gz  RD.nii.gz
  7. Write tensor_fit_report.json and return TensorEstimationRecord

Only BNI subjects (which have DTI) should be processed.
ETH and EMC must not be passed to this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# Minimum number of diffusion-weighted directions required before fitting
MIN_DWI_DIRECTIONS = 30
B0_THRESHOLD       = 100.0   # bval below this → B0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TensorEstimationRecord:
    subject_id: str
    input_volume_count: int
    bval_count: int
    bvec_count: int
    b0_count: int
    dwi_count: int
    fa_mean: Optional[float]
    fa_std: Optional[float]
    md_mean: Optional[float]
    rd_mean: Optional[float]
    ad_mean: Optional[float]
    status: str                          # "success" | "failed" | "skipped"
    error_message: Optional[str] = field(default=None)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_summary_row(self) -> dict:
        return {
            "subject_id":          self.subject_id,
            "input_volume_count":  self.input_volume_count,
            "bval_count":          self.bval_count,
            "bvec_count":          self.bvec_count,
            "b0_count":            self.b0_count,
            "dwi_count":           self.dwi_count,
            "fa_mean":             _r(self.fa_mean),
            "fa_std":              _r(self.fa_std),
            "md_mean":             _r(self.md_mean),
            "rd_mean":             _r(self.rd_mean),
            "ad_mean":             _r(self.ad_mean),
            "status":              self.status,
            "error_message":       self.error_message,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_tensors_subject(
    dti_path: Path,
    bval_path: Path,
    bvec_path: Path,
    output_dir: Path,
    raw_root: Optional[Path] = None,
) -> TensorEstimationRecord:
    """
    Fit tensor maps for one subject and save outputs under *output_dir/tensor/*.

    Parameters
    ----------
    dti_path:   corrected DTI NIfTI (e.g. dti_corrected.nii.gz)
    bval_path:  matching bval file
    bvec_path:  matching bvec file (FSL convention: 3×N)
    output_dir: parent directory; tensor/ subdirectory is created automatically
    raw_root:   if given, guard that output_dir is not inside raw data tree
    """
    # dti_path = .../ABIDEII-BNI_1/<subject_id>/session_1/dti_1/dti_corrected.nii.gz
    subject_id = dti_path.parent.parent.parent.name

    if raw_root:
        guard_no_raw_write(output_dir, raw_root)

    tensor_dir = output_dir / "tensor"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    # --- load & validate ---
    try:
        img, bvals, bvecs, n_vols, n_bvals, n_bvecs = _load_and_validate(
            dti_path, bval_path, bvec_path
        )
    except Exception as exc:
        return _failed(subject_id, 0, 0, 0, 0, 0, str(exc))

    b0_count  = int(np.sum(bvals < B0_THRESHOLD))
    dwi_count = int(np.sum(bvals >= B0_THRESHOLD))

    # --- fit ---
    try:
        fa, md, ad, rd, mask = _fit_tensor(img, bvals, bvecs)
    except Exception as exc:
        return _failed(subject_id, n_vols, n_bvals, n_bvecs, b0_count, dwi_count, str(exc))

    # --- statistics within brain mask ---
    brain_mask = mask & (fa > 0)
    fa_mean = _r(float(fa[brain_mask].mean()))  if brain_mask.any() else None
    fa_std  = _r(float(fa[brain_mask].std()))   if brain_mask.any() else None
    md_mean = _r(float(md[brain_mask].mean()))  if brain_mask.any() else None
    ad_mean = _r(float(ad[brain_mask].mean()))  if brain_mask.any() else None
    rd_mean = _r(float(rd[brain_mask].mean()))  if brain_mask.any() else None

    # --- save maps ---
    affine  = img.affine
    header  = img.header
    _save_map(fa, affine, header, tensor_dir / "FA.nii.gz")
    _save_map(md, affine, header, tensor_dir / "MD.nii.gz")
    _save_map(ad, affine, header, tensor_dir / "AD.nii.gz")
    _save_map(rd, affine, header, tensor_dir / "RD.nii.gz")

    rec = TensorEstimationRecord(
        subject_id=subject_id,
        input_volume_count=n_vols,
        bval_count=n_bvals,
        bvec_count=n_bvecs,
        b0_count=b0_count,
        dwi_count=dwi_count,
        fa_mean=fa_mean, fa_std=fa_std,
        md_mean=md_mean, ad_mean=ad_mean, rd_mean=rd_mean,
        status="success",
    )
    _write_report(rec, tensor_dir, str(dti_path))

    logger.info(
        "[%s] success  FA_mean=%.4f  MD_mean=%.6f",
        subject_id, fa_mean or 0, md_mean or 0,
    )
    return rec


def run_tensor_batch(
    processed_root: Path,
    raw_root: Optional[Path] = None,
    session: str = "session_1",
) -> list[TensorEstimationRecord]:
    """
    Run tensor estimation for every subject under *processed_root*.

    Expected layout:
      <processed_root>/<subject_id>/<session>/dti_1/
        dti_corrected.nii.gz   dti.bval   dti.bvec
    """
    if raw_root:
        guard_no_raw_write(processed_root, raw_root)

    subject_dirs = sorted(p for p in processed_root.iterdir() if p.is_dir())
    logger.info("Tensor estimation: %d subjects in %s", len(subject_dirs), processed_root)

    records: list[TensorEstimationRecord] = []
    for subject_dir in subject_dirs:
        sid      = subject_dir.name
        dti_dir  = subject_dir / session / "dti_1"
        dti_path  = dti_dir / "dti_corrected.nii.gz"
        bval_path = dti_dir / "dti.bval"
        bvec_path = dti_dir / "dti.bvec"

        if not all(p.exists() for p in [dti_path, bval_path, bvec_path]):
            missing = [p.name for p in [dti_path, bval_path, bvec_path] if not p.exists()]
            logger.warning("[%s] skipping — missing: %s", sid, missing)
            records.append(_failed(sid, 0, 0, 0, 0, 0, f"Missing inputs: {missing}"))
            continue

        rec = estimate_tensors_subject(
            dti_path, bval_path, bvec_path,
            output_dir=dti_dir,
            raw_root=raw_root,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class TensorValidationError(Exception):
    pass


def _load_and_validate(
    dti_path: Path,
    bval_path: Path,
    bvec_path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, int, int, int]:
    img   = nib.load(str(dti_path))
    if img.ndim != 4:
        raise TensorValidationError(f"Expected 4D DTI, got shape {img.shape}")

    n_vols = img.shape[3]

    bvals  = np.loadtxt(str(bval_path))
    n_bvals = int(bvals.size)

    bvecs  = np.loadtxt(str(bvec_path))
    if bvecs.ndim == 1:
        n_bvecs = 1
    elif bvecs.shape[0] == 3:
        n_bvecs = int(bvecs.shape[1])   # FSL: 3×N
    elif bvecs.shape[1] == 3:
        n_bvecs = int(bvecs.shape[0])   # N×3
    else:
        raise TensorValidationError(f"Unexpected bvec shape: {bvecs.shape}")

    if n_vols != n_bvals:
        raise TensorValidationError(
            f"Volume count ({n_vols}) ≠ bval count ({n_bvals})"
        )
    if n_vols != n_bvecs:
        raise TensorValidationError(
            f"Volume count ({n_vols}) ≠ bvec count ({n_bvecs})"
        )

    b0_count  = int(np.sum(bvals < B0_THRESHOLD))
    dwi_count = int(np.sum(bvals >= B0_THRESHOLD))

    if b0_count < 1:
        raise TensorValidationError("No B0 volumes found (bval < 100)")
    if dwi_count < MIN_DWI_DIRECTIONS:
        raise TensorValidationError(
            f"Only {dwi_count} DWI directions — need ≥ {MIN_DWI_DIRECTIONS}"
        )

    return img, bvals, bvecs, n_vols, n_bvals, n_bvecs


def _fit_tensor(
    img: nib.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit WLS tensor and return (FA, MD, AD, RD, mask)."""
    # DIPY wants bvecs as (N, 3); FSL stores (3, N)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T   # → (N, 3)

    gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESHOLD)

    data = img.get_fdata(dtype=np.float32)

    # Threshold mask from mean B0 signal to skip background
    b0_idx  = np.where(gtab.b0s_mask)[0]
    b0_mean = data[..., b0_idx].mean(axis=-1) if len(b0_idx) > 1 else data[..., b0_idx[0]]
    thr  = float(np.percentile(b0_mean, 30))
    mask = b0_mean > thr
    if not mask.any():
        # Fallback for edge cases (e.g. uniform synthetic data): include all
        # voxels with any positive b0 signal.
        mask = b0_mean > 0

    tenfit = TensorModel(gtab).fit(data, mask=mask)

    fa = np.clip(tenfit.fa.astype(np.float32), 0.0, 1.0)
    md = tenfit.md.astype(np.float32)
    ad = tenfit.ad.astype(np.float32)
    rd = tenfit.rd.astype(np.float32)

    # Zero out non-mask voxels
    for arr in (fa, md, ad, rd):
        arr[~mask] = 0.0

    return fa, md, ad, rd, mask


def _save_map(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: Path,
) -> None:
    img = nib.Nifti1Image(data, affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def _write_report(
    rec: TensorEstimationRecord,
    tensor_dir: Path,
    input_dti_path: str,
) -> None:
    report = asdict(rec)
    report["input_dti_path"] = input_dti_path
    (tensor_dir / "tensor_fit_report.json").write_text(json.dumps(report, indent=2))


def _failed(
    subject_id: str,
    n_vols: int,
    n_bvals: int,
    n_bvecs: int,
    b0_count: int,
    dwi_count: int,
    error: str,
) -> TensorEstimationRecord:
    logger.error("[%s] failed: %s", subject_id, error)
    return TensorEstimationRecord(
        subject_id=subject_id,
        input_volume_count=n_vols,
        bval_count=n_bvals,
        bvec_count=n_bvecs,
        b0_count=b0_count,
        dwi_count=dwi_count,
        fa_mean=None, fa_std=None,
        md_mean=None, rd_mean=None, ad_mean=None,
        status="failed",
        error_message=error,
    )


def _r(v: Optional[float], decimals: int = 6) -> Optional[float]:
    return round(v, decimals) if v is not None else None
