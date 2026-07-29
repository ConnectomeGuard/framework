"""
NeuroFiber Phase 2R.3 — Brain Masking and Tensor QC from processed_v2b

Generates DIPY median_otsu brain masks and tensor QC reports for all
277 valid DTI subjects in the processed_v2b canonical pipeline.

Input layout per subject:
  data/processed_v2b/abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1/
    dwi_preprocessed.nii.gz
    dwi_preprocessed.bval
    tensor/FA.nii.gz  MD.nii.gz  AD.nii.gz  RD.nii.gz

Output layout per subject:
  .../session_1/dti_1/qc/
    brain_mask.nii.gz
    b0_brain.nii.gz
    tensor_qc_report.json
    b0_mask_overlay.png
    fa_mid_slice.png
    md_mid_slice.png

Cohort outputs:
  data/processed_v2b/
    phase2r_3_brain_mask_tensor_qc_summary.csv
    phase2r_3_site_qc_summary.csv
    phase2r_3_site_stability_analysis.csv
  data/processed_v2b/site_qc/
    <site>_fa_representative.png
    <site>_md_representative.png
    <site>_mask_overlay_representative.png

Safety contract:
  - Reads only from data/processed_v2b/
  - guard_no_raw_write() called at every write entry point
  - Never writes to data/processed/ or data/processed_v2/ or data/raw/
"""

from __future__ import annotations

import csv
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

PIPELINE_VERSION = "2R.3"

# median_otsu parameters (same as v1 — reproducible)
_OTSU_RADIUS  = 3
_OTSU_NUMPASS = 2
_OTSU_DILATE  = 1

# QC thresholds
_FA_WARN_LOW  = 0.10
_FA_WARN_HIGH = 0.55

# Outlier detection: flag sites where cohort mean deviates by > N σ from grand mean
_OUTLIER_SIGMA = 2.0

SITES = ["BNI", "IP_1", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]
_SITE_FOLDER_MAP = {
    "BNI":    "bni",
    "IP_1":   "ip",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
}

SUBJECT_CSV_FIELDS = [
    "site", "dataset", "subject_id",
    "brain_voxel_count", "mask_coverage_ratio",
    "fa_mean", "fa_std", "fa_min", "fa_max",
    "md_mean", "md_std",
    "ad_mean", "ad_std",
    "rd_mean", "rd_std",
    "nan_count", "inf_count", "suspicious_fa_count",
    "status", "warning_message",
]

SITE_CSV_FIELDS = [
    "site", "subjects",
    "mean_mask_voxels", "std_mask_voxels",
    "mean_fa", "std_fa",
    "mean_md", "std_md",
    "mean_ad", "std_ad",
    "mean_rd", "std_rd",
    "outlier_flag",
]

STABILITY_CSV_FIELDS = SITE_CSV_FIELDS  # same structure, kept separate for clarity


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BrainMaskQCRecord:
    site:        str
    dataset:     str
    subject_id:  str
    brain_voxel_count:    int   = 0
    mask_coverage_ratio:  float = 0.0
    fa_mean:  Optional[float] = None
    fa_std:   Optional[float] = None
    fa_min:   Optional[float] = None
    fa_max:   Optional[float] = None
    md_mean:  Optional[float] = None
    md_std:   Optional[float] = None
    ad_mean:  Optional[float] = None
    ad_std:   Optional[float] = None
    rd_mean:  Optional[float] = None
    rd_std:   Optional[float] = None
    nan_count:            int  = 0
    inf_count:            int  = 0
    suspicious_fa_count:  int  = 0
    status:          str           = "pending"
    warning_message: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_csv_row(self) -> dict:
        return {k: getattr(self, k, None) for k in SUBJECT_CSV_FIELDS}


# ---------------------------------------------------------------------------
# Core subject pipeline
# ---------------------------------------------------------------------------

def process_subject(
    dti_dir:    Path,
    output_root: Path,
    raw_root:   Path,
    skip_if_exists: bool = True,
    save_png:   bool = True,
) -> BrainMaskQCRecord:
    """
    Generate brain mask and tensor QC for one subject.
    dti_dir: .../session_1/dti_1/  containing dwi_preprocessed.* and tensor/
    """
    subject_id  = dti_dir.parents[1].name
    dataset     = dti_dir.parents[2].name
    site_folder = dti_dir.parents[3].name
    site = next((k for k, v in _SITE_FOLDER_MAP.items() if v == site_folder), site_folder.upper())

    guard_no_raw_write(dti_dir, raw_root)

    rec = BrainMaskQCRecord(site=site, dataset=dataset, subject_id=subject_id)
    qc_dir = dti_dir / "qc"

    # Skip if already done
    if skip_if_exists and (qc_dir / "tensor_qc_report.json").exists():
        try:
            d = json.loads((qc_dir / "tensor_qc_report.json").read_text())
            if d.get("status") == "success":
                rec.status              = "success"
                rec.brain_voxel_count   = d.get("brain_voxel_count", 0)
                rec.mask_coverage_ratio = d.get("mask_coverage_ratio", 0.0)
                rec.fa_mean  = d.get("fa_mean");  rec.fa_std  = d.get("fa_std")
                rec.fa_min   = d.get("fa_min");   rec.fa_max  = d.get("fa_max")
                rec.md_mean  = d.get("md_mean");  rec.md_std  = d.get("md_std")
                rec.ad_mean  = d.get("ad_mean");  rec.ad_std  = d.get("ad_std")
                rec.rd_mean  = d.get("rd_mean");  rec.rd_std  = d.get("rd_std")
                rec.nan_count           = d.get("nan_count", 0)
                rec.inf_count           = d.get("inf_count", 0)
                rec.suspicious_fa_count = d.get("suspicious_fa_count", 0)
                rec.warning_message     = d.get("warning_message")
                logger.info("[%s/%s] already done — skipping", site, subject_id)
                return rec
        except Exception:
            pass

    qc_dir.mkdir(parents=True, exist_ok=True)

    dwi_path  = dti_dir / "dwi_preprocessed.nii.gz"
    bval_path = dti_dir / "dwi_preprocessed.bval"
    tensor_dir = dti_dir / "tensor"

    # Validate inputs
    missing = [p.name for p in [dwi_path, bval_path] if not p.exists()]
    if missing:
        return _failed(rec, f"Missing inputs: {missing}")
    if not tensor_dir.exists():
        return _failed(rec, "tensor/ directory not found — run Phase 2R.2b first")
    for m in ["FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz"]:
        if not (tensor_dir / m).exists():
            return _failed(rec, f"Missing tensor map: {m}")

    # Load B0
    try:
        b0_vol, img_ref = _load_b0(dwi_path, bval_path)
    except Exception as exc:
        return _failed(rec, f"load B0 failed: {exc}")

    # Brain mask
    try:
        mask, b0_brain = _run_median_otsu(b0_vol)
    except Exception as exc:
        return _failed(rec, f"median_otsu failed: {exc}")

    # Save mask and b0_brain
    try:
        _save_nifti(mask.astype(np.uint8),      img_ref, qc_dir / "brain_mask.nii.gz")
        _save_nifti(b0_brain.astype(np.float32), img_ref, qc_dir / "b0_brain.nii.gz")
    except Exception as exc:
        return _failed(rec, f"save mask/b0: {exc}")

    # Load tensor maps
    try:
        fa, md, ad, rd = _load_tensor_maps(tensor_dir)
    except Exception as exc:
        return _failed(rec, f"load tensor maps: {exc}")

    # QC stats
    _fill_qc_stats(rec, fa, md, ad, rd, mask)

    # PNGs
    if save_png:
        try:
            _save_subject_pngs(b0_vol, b0_brain, mask, fa, md, qc_dir)
        except Exception as exc:
            logger.warning("[%s/%s] PNG failed (non-fatal): %s", site, subject_id, exc)

    rec.status = "success"

    # Write JSON report
    report = {**rec.to_csv_row(), "pipeline_version": PIPELINE_VERSION,
              "input_dwi_path": str(dwi_path), "tensor_dir": str(tensor_dir),
              "timestamp": rec.timestamp}
    (qc_dir / "tensor_qc_report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "[%s/%s] mask=%d vox (%.1f%%)  FA=%.4f±%.4f",
        site, subject_id,
        rec.brain_voxel_count,
        rec.mask_coverage_ratio * 100,
        rec.fa_mean or 0, rec.fa_std or 0,
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_masking_batch(
    processed_v2b_root: Path,
    raw_root:           Path,
    sites:              list[str] = SITES,
    skip_if_exists:     bool = True,
    save_png:           bool = True,
) -> list[BrainMaskQCRecord]:
    """
    Run brain masking and tensor QC for all subjects across all sites.
    processed_v2b_root: e.g. data/processed_v2b/abide_ii
    """
    guard_no_raw_write(processed_v2b_root, raw_root)

    records: list[BrainMaskQCRecord] = []

    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = processed_v2b_root / folder
        if not site_root.exists():
            logger.warning("[%s] not found: %s", site, site_root)
            continue

        dti_dirs = sorted(site_root.rglob("dti_1"))
        # Include any dti_1 with a tensor dir (Phase 2R.2b output).
        # Missing DWI files are handled inside process_subject → status=failed.
        dti_dirs = [d for d in dti_dirs if d.is_dir() and (d / "tensor").exists()]
        logger.info("[%s] %d subjects", site, len(dti_dirs))

        for dti_dir in dti_dirs:
            rec = process_subject(
                dti_dir=dti_dir,
                output_root=processed_v2b_root,
                raw_root=raw_root,
                skip_if_exists=skip_if_exists,
                save_png=save_png,
            )
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def write_subject_summary(records: list[BrainMaskQCRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_csv_row() for r in records)
    logger.info("Subject QC summary → %s  (%d rows)", out_path, len(records))
    return out_path


def write_site_summary(
    records:  list[BrainMaskQCRecord],
    out_path: Path,
) -> tuple[Path, list[dict]]:
    """Compute per-site means/stds and write CSV. Returns (path, rows)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site_rows = _compute_site_stats(records)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SITE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(site_rows)
    logger.info("Site QC summary → %s", out_path)
    return out_path, site_rows


def write_stability_analysis(
    records:  list[BrainMaskQCRecord],
    out_path: Path,
) -> Path:
    """Compute cross-site outlier detection and write CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site_rows = _compute_site_stats(records, flag_outliers=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STABILITY_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(site_rows)
    logger.info("Stability analysis → %s", out_path)
    return out_path


def write_site_qc_pngs(
    records:        list[BrainMaskQCRecord],
    processed_v2b_root: Path,
    site_qc_dir:    Path,
) -> None:
    """
    For each site, select a representative subject (median FA) and save
    FA, MD, and mask overlay PNGs to site_qc_dir/<site>_*.png.
    """
    site_qc_dir.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        folder     = _SITE_FOLDER_MAP.get(site, site.lower())
        site_recs  = [r for r in records if r.site == site and r.status == "success"
                      and r.fa_mean is not None]
        if not site_recs:
            logger.warning("[%s] no successful subjects for site PNG", site)
            continue

        # Pick subject closest to cohort median FA
        fa_vals = [r.fa_mean for r in site_recs]
        median_fa = float(np.median(fa_vals))
        rep = min(site_recs, key=lambda r: abs(r.fa_mean - median_fa))

        dti_dir = (
            processed_v2b_root / folder / rep.dataset / rep.subject_id
            / "session_1" / "dti_1"
        )
        dwi_path   = dti_dir / "dwi_preprocessed.nii.gz"
        bval_path  = dti_dir / "dwi_preprocessed.bval"
        tensor_dir = dti_dir / "tensor"
        mask_path  = dti_dir / "qc" / "brain_mask.nii.gz"

        if not all(p.exists() for p in [dwi_path, bval_path, mask_path]):
            logger.warning("[%s] representative subject files missing", site)
            continue

        try:
            b0_vol, _ = _load_b0(dwi_path, bval_path)
            mask      = nib.load(str(mask_path)).get_fdata(dtype=np.float32).astype(bool)
            fa, md, _, _ = _load_tensor_maps(tensor_dir)
            b0_brain  = b0_vol * mask

            _save_site_pngs(site, rep.subject_id, b0_vol, b0_brain, mask,
                            fa, md, site_qc_dir)
        except Exception as exc:
            logger.warning("[%s] site PNG failed: %s", site, exc)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_b0(dwi_path: Path, bval_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img   = nib.load(str(dwi_path))
    bvals = np.loadtxt(str(bval_path))
    b0_idx = np.where(bvals < 100)[0]
    if len(b0_idx) == 0:
        raise ValueError("No B0 volumes found (all bvals ≥ 100)")
    data = img.get_fdata(dtype=np.float32)
    b0 = data[..., b0_idx].mean(axis=-1) if len(b0_idx) > 1 else data[..., b0_idx[0]]
    return b0, img


def _run_median_otsu(b0_vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, b0_brain = median_otsu(
            b0_vol,
            median_radius=_OTSU_RADIUS,
            numpass=_OTSU_NUMPASS,
            dilate=_OTSU_DILATE,
        )
    mask = b0_brain > 0
    if not mask.any():
        raise ValueError("median_otsu returned empty mask")
    return mask.astype(bool), b0_brain.astype(np.float32)


def _load_tensor_maps(tensor_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _load(name: str) -> np.ndarray:
        return nib.load(str(tensor_dir / name)).get_fdata(dtype=np.float32)
    return _load("FA.nii.gz"), _load("MD.nii.gz"), _load("AD.nii.gz"), _load("RD.nii.gz")


def _fill_qc_stats(
    rec:  BrainMaskQCRecord,
    fa:   np.ndarray,
    md:   np.ndarray,
    ad:   np.ndarray,
    rd:   np.ndarray,
    mask: np.ndarray,
) -> None:
    brain_voxels = int(mask.sum())
    total_voxels = mask.size
    all_maps  = np.stack([fa, md, ad, rd], axis=-1)
    nan_count = int(np.isnan(all_maps).sum())
    inf_count = int(np.isinf(all_maps).sum())

    fa_c = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
    md_c = np.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
    ad_c = np.nan_to_num(ad, nan=0.0, posinf=0.0, neginf=0.0)
    rd_c = np.nan_to_num(rd, nan=0.0, posinf=0.0, neginf=0.0)

    suspicious_fa = int(((fa_c[mask] < 0) | (fa_c[mask] > 1)).sum())

    def _s(arr: np.ndarray) -> tuple[float, float, float, float]:
        m = arr[mask]
        return float(m.mean()), float(m.std()), float(m.min()), float(m.max())

    fa_mean, fa_std, fa_min, fa_max = _s(fa_c)
    md_mean, md_std, _, _           = _s(md_c)
    ad_mean, ad_std, _, _           = _s(ad_c)
    rd_mean, rd_std, _, _           = _s(rd_c)

    warnings_list: list[str] = []
    if fa_mean < _FA_WARN_LOW:
        warnings_list.append(f"FA_mean={fa_mean:.3f} < {_FA_WARN_LOW} — poor mask or tissue coverage")
    if fa_mean > _FA_WARN_HIGH:
        warnings_list.append(f"FA_mean={fa_mean:.3f} > {_FA_WARN_HIGH} — mask may overrepresent WM")
    if nan_count > 0:
        warnings_list.append(f"{nan_count} NaN values")
    if inf_count > 0:
        warnings_list.append(f"{inf_count} Inf values")
    if suspicious_fa > 0:
        warnings_list.append(f"{suspicious_fa} FA voxels outside [0,1]")

    rec.brain_voxel_count   = brain_voxels
    rec.mask_coverage_ratio = round(brain_voxels / total_voxels, 6)
    rec.fa_mean = _r(fa_mean); rec.fa_std = _r(fa_std)
    rec.fa_min  = _r(max(0.0, fa_min)); rec.fa_max = _r(min(1.0, fa_max))
    rec.md_mean = _r(md_mean); rec.md_std = _r(md_std)
    rec.ad_mean = _r(ad_mean); rec.ad_std = _r(ad_std)
    rec.rd_mean = _r(rd_mean); rec.rd_std = _r(rd_std)
    rec.nan_count           = nan_count
    rec.inf_count           = inf_count
    rec.suspicious_fa_count = suspicious_fa
    rec.warning_message     = "; ".join(warnings_list) if warnings_list else None


def _compute_site_stats(
    records: list[BrainMaskQCRecord],
    flag_outliers: bool = False,
) -> list[dict]:
    """Compute per-site aggregate stats. Optionally flag cross-site outliers."""
    site_stats: list[dict] = []

    for site in SITES:
        recs = [r for r in records if r.site == site and r.status == "success"]
        n    = len(recs)

        def _agg(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
            if not vals:
                return None, None
            return _r(float(np.mean(vals))), _r(float(np.std(vals)))

        mask_vols = [r.brain_voxel_count for r in recs if r.brain_voxel_count]
        fa_vals   = [r.fa_mean for r in recs if r.fa_mean is not None]
        md_vals   = [r.md_mean for r in recs if r.md_mean is not None]
        ad_vals   = [r.ad_mean for r in recs if r.ad_mean is not None]
        rd_vals   = [r.rd_mean for r in recs if r.rd_mean is not None]

        mean_mask, std_mask = _agg([float(v) for v in mask_vols])
        mean_fa,   std_fa   = _agg(fa_vals)
        mean_md,   std_md   = _agg(md_vals)
        mean_ad,   std_ad   = _agg(ad_vals)
        mean_rd,   std_rd   = _agg(rd_vals)

        site_stats.append({
            "site":               site,
            "subjects":           n,
            "mean_mask_voxels":   mean_mask,
            "std_mask_voxels":    std_mask,
            "mean_fa":            mean_fa,
            "std_fa":             std_fa,
            "mean_md":            mean_md,
            "std_md":             std_md,
            "mean_ad":            mean_ad,
            "std_ad":             std_ad,
            "mean_rd":            mean_rd,
            "std_rd":             std_rd,
            "outlier_flag":       "",
        })

    if flag_outliers and len(site_stats) > 1:
        _detect_outliers(site_stats)

    return site_stats


def _detect_outliers(site_stats: list[dict]) -> None:
    """Flag sites whose mean FA deviates > _OUTLIER_SIGMA σ from grand mean."""
    for metric_key in ("mean_fa", "mean_md", "mean_mask_voxels"):
        vals = [s[metric_key] for s in site_stats if s[metric_key] is not None]
        if len(vals) < 2:
            continue
        grand_mean = float(np.mean(vals))
        grand_std  = float(np.std(vals))
        if grand_std == 0:
            continue
        for s in site_stats:
            v = s[metric_key]
            if v is None:
                continue
            z = abs(float(v) - grand_mean) / grand_std
            if z > _OUTLIER_SIGMA:
                flag = s["outlier_flag"]
                tag  = f"{metric_key}_outlier(z={z:.1f})"
                s["outlier_flag"] = f"{flag}; {tag}" if flag else tag


def _save_subject_pngs(
    b0_vol: np.ndarray, b0_brain: np.ndarray, mask: np.ndarray,
    fa: np.ndarray, md: np.ndarray, output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mid = b0_vol.shape[2] // 2

    # B0 + mask boundary
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(b0_vol[:, :, z_mid].T, cmap="gray", origin="lower",
              vmin=0, vmax=np.percentile(b0_vol, 99))
    ax.contour(mask[:, :, z_mid].T, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title("B0 + mask", fontsize=9); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "b0_mask_overlay.png"), dpi=120)
    plt.close(fig)

    # FA
    fa_m = np.where(mask, fa, np.nan)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fa_m[:, :, z_mid].T, cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("FA (masked)", fontsize=9); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "fa_mid_slice.png"), dpi=120)
    plt.close(fig)

    # MD
    md_m = np.where(mask, md * 1000, np.nan)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(md_m[:, :, z_mid].T, cmap="viridis", origin="lower", vmin=0, vmax=2.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("MD ×10³ mm²/s (masked)", fontsize=9); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_dir / "md_mid_slice.png"), dpi=120)
    plt.close(fig)


def _save_site_pngs(
    site: str, subject_id: str,
    b0_vol: np.ndarray, b0_brain: np.ndarray, mask: np.ndarray,
    fa: np.ndarray, md: np.ndarray, site_qc_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mid = b0_vol.shape[2] // 2
    subtitle = f"{site} — representative subject {subject_id}"

    # Mask overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(b0_vol[:, :, z_mid].T, cmap="gray", origin="lower",
              vmin=0, vmax=np.percentile(b0_vol, 99))
    ax.contour(mask[:, :, z_mid].T, levels=[0.5], colors="red", linewidths=1.0)
    ax.set_title(f"B0 + mask\n{subtitle}", fontsize=8); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(site_qc_dir / f"{site}_mask_overlay_representative.png"), dpi=150)
    plt.close(fig)

    # FA
    fa_m = np.where(mask, fa, np.nan)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(fa_m[:, :, z_mid].T, cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"FA\n{subtitle}", fontsize=8); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(site_qc_dir / f"{site}_fa_representative.png"), dpi=150)
    plt.close(fig)

    # MD
    md_m = np.where(mask, md * 1000, np.nan)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(md_m[:, :, z_mid].T, cmap="viridis", origin="lower", vmin=0, vmax=2.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"MD ×10³ mm²/s\n{subtitle}", fontsize=8); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(str(site_qc_dir / f"{site}_md_representative.png"), dpi=150)
    plt.close(fig)


def _save_nifti(data: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    nib.save(nib.Nifti1Image(data, ref.affine, ref.header), str(path))


def _failed(rec: BrainMaskQCRecord, error: str) -> BrainMaskQCRecord:
    rec.status = "failed"
    rec.warning_message = error
    logger.error("[%s/%s] %s", rec.site, rec.subject_id, error)
    return rec


def _r(v: float, decimals: int = 6) -> float:
    return round(v, decimals)
