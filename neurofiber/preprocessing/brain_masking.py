"""
Phase 2.3 — DTI Brain Masking and Tensor QC

Applies DIPY median_otsu brain extraction to the B0 image and recalculates
tensor quality metrics (FA, MD, AD, RD) within the resulting anatomical mask.

The median_otsu mask replaces the Phase 2.2 B0-percentile mask, which was
biased toward white matter and inflated mean FA (~0.55 → corrected ~0.24).

Per-subject outputs (written to dti_1/qc/):
  brain_mask.nii.gz         binary brain mask (uint8)
  b0_brain.nii.gz           skull-stripped B0 image
  tensor_qc_report.json     QC metrics and warnings
  b0_mask_overlay.png       B0 mid-axial slice with mask boundary
  fa_mid_slice.png          FA mid-axial slice (brain mask applied)
  md_mid_slice.png          MD mid-axial slice (brain mask applied)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from dipy.segment.mask import median_otsu

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# median_otsu parameters
_OTSU_RADIUS  = 3
_OTSU_NUMPASS = 2
_OTSU_DILATE  = 1

# QC thresholds
_FA_WARN_LOW  = 0.10
_FA_WARN_HIGH = 0.50


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BrainMaskQCRecord:
    subject_id: str
    brain_voxel_count: int
    mask_coverage_ratio: float
    fa_mean: Optional[float]
    fa_std: Optional[float]
    fa_min: Optional[float]
    fa_max: Optional[float]
    md_mean: Optional[float]
    md_std: Optional[float]
    md_min: Optional[float]
    md_max: Optional[float]
    ad_mean: Optional[float]
    rd_mean: Optional[float]
    nan_count: int
    inf_count: int
    suspicious_fa_count: int
    suspicious_diffusivity_count: int
    status: str                          # "success" | "failed"
    warning_message: Optional[str] = field(default=None)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_summary_row(self) -> dict:
        return {
            "subject_id":                  self.subject_id,
            "brain_voxel_count":           self.brain_voxel_count,
            "mask_coverage_ratio":         _r(self.mask_coverage_ratio, 4),
            "fa_mean":                     _r(self.fa_mean),
            "fa_std":                      _r(self.fa_std),
            "fa_min":                      _r(self.fa_min),
            "fa_max":                      _r(self.fa_max),
            "md_mean":                     _r(self.md_mean),
            "md_std":                      _r(self.md_std),
            "md_min":                      _r(self.md_min),
            "md_max":                      _r(self.md_max),
            "ad_mean":                     _r(self.ad_mean),
            "rd_mean":                     _r(self.rd_mean),
            "nan_count":                   self.nan_count,
            "inf_count":                   self.inf_count,
            "suspicious_fa_count":         self.suspicious_fa_count,
            "suspicious_diffusivity_count": self.suspicious_diffusivity_count,
            "status":                      self.status,
            "warning_message":             self.warning_message,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_subject(
    dti_path: Path,
    bval_path: Path,
    tensor_dir: Path,
    output_dir: Path,
    raw_root: Optional[Path] = None,
    save_png: bool = True,
) -> BrainMaskQCRecord:
    """
    Generate brain mask and tensor QC for one subject.

    Parameters
    ----------
    dti_path:   corrected DTI NIfTI (dti_corrected.nii.gz)
    bval_path:  matching bval file
    tensor_dir: directory containing FA/MD/AD/RD NIfTIs
    output_dir: where to write brain_mask.nii.gz, b0_brain.nii.gz, PNGs, report
    raw_root:   if given, guard that output_dir is not inside raw data tree
    save_png:   whether to write PNG QC images
    """
    subject_id = dti_path.parent.parent.parent.name

    if raw_root:
        guard_no_raw_write(output_dir, raw_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load B0 ---
    try:
        b0_vol, img_ref = _load_b0(dti_path, bval_path)
    except Exception as exc:
        return _failed(subject_id, str(exc))

    # --- brain mask ---
    try:
        mask, b0_brain = _run_median_otsu(b0_vol)
    except Exception as exc:
        return _failed(subject_id, f"median_otsu failed: {exc}")

    # --- save mask and b0_brain ---
    try:
        _save_nifti(mask.astype(np.uint8), img_ref, output_dir / "brain_mask.nii.gz")
        _save_nifti(b0_brain.astype(np.float32), img_ref, output_dir / "b0_brain.nii.gz")
    except Exception as exc:
        return _failed(subject_id, f"save mask/b0: {exc}")

    # --- load tensor maps ---
    try:
        fa, md, ad, rd = _load_tensor_maps(tensor_dir)
    except Exception as exc:
        return _failed(subject_id, f"load tensor maps: {exc}")

    # --- QC statistics ---
    rec = _compute_qc(subject_id, fa, md, ad, rd, mask)

    # --- optional QC PNGs ---
    if save_png:
        try:
            _save_qc_pngs(b0_vol, b0_brain, mask, fa, md, output_dir)
        except Exception as exc:
            logger.warning("[%s] PNG generation failed (non-fatal): %s", subject_id, exc)

    # --- write JSON report ---
    report = asdict(rec)
    report["input_dti_path"]  = str(dti_path)
    report["tensor_dir"]      = str(tensor_dir)
    (output_dir / "tensor_qc_report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "[%s] mask=%d vox (%.1f%%)  FA=%.4f±%.4f  warnings=%s",
        subject_id,
        rec.brain_voxel_count,
        rec.mask_coverage_ratio * 100,
        rec.fa_mean or 0,
        rec.fa_std  or 0,
        rec.warning_message or "none",
    )
    return rec


def run_masking_batch(
    processed_root: Path,
    raw_root: Optional[Path] = None,
    session: str = "session_1",
    save_png: bool = True,
) -> list[BrainMaskQCRecord]:
    """Run brain masking and tensor QC for every subject under *processed_root*."""
    if raw_root:
        guard_no_raw_write(processed_root, raw_root)

    subject_dirs = sorted(p for p in processed_root.iterdir() if p.is_dir())
    logger.info("Brain masking: %d subjects in %s", len(subject_dirs), processed_root)

    records: list[BrainMaskQCRecord] = []
    for subject_dir in subject_dirs:
        sid       = subject_dir.name
        dti_dir   = subject_dir / session / "dti_1"
        dti_path  = dti_dir / "dti_corrected.nii.gz"
        bval_path = dti_dir / "dti.bval"
        tensor_dir = dti_dir / "tensor"
        qc_dir    = dti_dir / "qc"

        missing = [p.name for p in [dti_path, bval_path] if not p.exists()]
        if missing or not tensor_dir.exists():
            logger.warning("[%s] skipping — missing: %s", sid,
                           missing + (["tensor/"] if not tensor_dir.exists() else []))
            records.append(_failed(sid, f"Missing inputs: {missing}"))
            continue

        rec = process_subject(
            dti_path, bval_path, tensor_dir,
            output_dir=qc_dir,
            raw_root=raw_root,
            save_png=save_png,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_b0(
    dti_path: Path,
    bval_path: Path,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    img   = nib.load(str(dti_path))
    bvals = np.loadtxt(str(bval_path))
    b0_indices = np.where(bvals < 100)[0]
    if len(b0_indices) == 0:
        raise ValueError("No B0 volumes found (all bvals ≥ 100)")
    data = img.get_fdata(dtype=np.float32)
    b0 = data[..., b0_indices].mean(axis=-1) if len(b0_indices) > 1 else data[..., b0_indices[0]]
    return b0, img


def _run_median_otsu(
    b0_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask_f, b0_brain = median_otsu(
            b0_vol,
            median_radius=_OTSU_RADIUS,
            numpass=_OTSU_NUMPASS,
            dilate=_OTSU_DILATE,
        )
    mask = mask_f.astype(bool)
    if not mask.any():
        raise ValueError("median_otsu returned empty mask")
    return mask, b0_brain.astype(np.float32)


def _load_tensor_maps(
    tensor_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _load(name: str) -> np.ndarray:
        p = tensor_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found in {tensor_dir}")
        return nib.load(str(p)).get_fdata(dtype=np.float32)

    return _load("FA.nii.gz"), _load("MD.nii.gz"), _load("AD.nii.gz"), _load("RD.nii.gz")


def _compute_qc(
    subject_id: str,
    fa: np.ndarray,
    md: np.ndarray,
    ad: np.ndarray,
    rd: np.ndarray,
    mask: np.ndarray,
) -> BrainMaskQCRecord:
    brain_voxels = int(mask.sum())
    total_voxels = mask.size

    # NaN / Inf counts across all maps
    all_maps = np.stack([fa, md, ad, rd], axis=-1)
    nan_count = int(np.isnan(all_maps).sum())
    inf_count = int(np.isinf(all_maps).sum())

    # Replace NaN/Inf with 0 for masked stats
    fa_c = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
    md_c = np.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
    ad_c = np.nan_to_num(ad, nan=0.0, posinf=0.0, neginf=0.0)
    rd_c = np.nan_to_num(rd, nan=0.0, posinf=0.0, neginf=0.0)

    suspicious_fa   = int(((fa_c[mask] < 0) | (fa_c[mask] > 1)).sum())
    suspicious_diff = int(((md_c[mask] < 0) | (ad_c[mask] < 0) | (rd_c[mask] < 0)).sum())

    def _stats(arr: np.ndarray) -> tuple[float, float, float, float]:
        m = arr[mask]
        return float(m.mean()), float(m.std()), float(m.min()), float(m.max())

    fa_mean, fa_std, fa_min, fa_max = _stats(fa_c)
    md_mean, md_std, md_min, md_max = _stats(md_c)
    ad_mean, _, _, _                = _stats(ad_c)
    rd_mean, _, _, _                = _stats(rd_c)

    # Clamp FA stats to [0, 1]
    fa_min = max(0.0, fa_min)
    fa_max = min(1.0, fa_max)

    # Build warning message
    warnings_list: list[str] = []
    if fa_mean < _FA_WARN_LOW:
        warnings_list.append(f"FA_mean={fa_mean:.3f} below {_FA_WARN_LOW} — poor mask or tissue coverage")
    if fa_mean > _FA_WARN_HIGH:
        warnings_list.append(f"FA_mean={fa_mean:.3f} above {_FA_WARN_HIGH} — mask may overrepresent WM")
    if nan_count > 0:
        warnings_list.append(f"{nan_count} NaN values across tensor maps")
    if inf_count > 0:
        warnings_list.append(f"{inf_count} Inf values across tensor maps")
    if suspicious_fa > 0:
        warnings_list.append(f"{suspicious_fa} FA voxels outside [0,1]")
    if suspicious_diff > 0:
        warnings_list.append(f"{suspicious_diff} negative diffusivity voxels")

    return BrainMaskQCRecord(
        subject_id=subject_id,
        brain_voxel_count=brain_voxels,
        mask_coverage_ratio=round(brain_voxels / total_voxels, 6),
        fa_mean=_r(fa_mean), fa_std=_r(fa_std),
        fa_min=_r(fa_min),   fa_max=_r(fa_max),
        md_mean=_r(md_mean), md_std=_r(md_std),
        md_min=_r(md_min),   md_max=_r(md_max),
        ad_mean=_r(ad_mean), rd_mean=_r(rd_mean),
        nan_count=nan_count,
        inf_count=inf_count,
        suspicious_fa_count=suspicious_fa,
        suspicious_diffusivity_count=suspicious_diff,
        status="success",
        warning_message="; ".join(warnings_list) if warnings_list else None,
    )


def _save_qc_pngs(
    b0_vol: np.ndarray,
    b0_brain: np.ndarray,
    mask: np.ndarray,
    fa: np.ndarray,
    md: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt

    z_mid = b0_vol.shape[2] // 2

    # 1. B0 + mask boundary overlay
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(b0_vol[:, :, z_mid].T, cmap="gray", origin="lower",
              vmin=0, vmax=np.percentile(b0_vol, 99))
    ax.contour(mask[:, :, z_mid].T, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title("B0 + mask boundary", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "b0_mask_overlay.png"), dpi=120)
    plt.close(fig)

    # 2. FA mid-axial (masked)
    fa_masked = np.where(mask, fa, np.nan)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fa_masked[:, :, z_mid].T, cmap="RdYlGn",
                   origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("FA (brain mask)", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "fa_mid_slice.png"), dpi=120)
    plt.close(fig)

    # 3. MD mid-axial (masked, scaled to µm²/ms for readability)
    md_masked = np.where(mask, md * 1000, np.nan)   # mm²/s → ×10³ mm²/s
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(md_masked[:, :, z_mid].T, cmap="viridis",
                   origin="lower", vmin=0, vmax=2.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("MD ×10³ mm²/s (brain mask)", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "md_mid_slice.png"), dpi=120)
    plt.close(fig)


def _save_nifti(data: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    nib.save(nib.Nifti1Image(data, ref.affine, ref.header), str(path))


def _failed(subject_id: str, error: str) -> BrainMaskQCRecord:
    logger.error("[%s] failed: %s", subject_id, error)
    return BrainMaskQCRecord(
        subject_id=subject_id,
        brain_voxel_count=0,
        mask_coverage_ratio=0.0,
        fa_mean=None, fa_std=None, fa_min=None, fa_max=None,
        md_mean=None, md_std=None, md_min=None, md_max=None,
        ad_mean=None, rd_mean=None,
        nan_count=0, inf_count=0,
        suspicious_fa_count=0, suspicious_diffusivity_count=0,
        status="failed",
        warning_message=error,
    )


def _r(v: Optional[float], decimals: int = 6) -> Optional[float]:
    return round(v, decimals) if v is not None else None
