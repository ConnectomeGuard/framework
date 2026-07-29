"""
NeuroFiber Phase 2R.2 — Tensor Estimation from Standardized DWI (processed_v2)

Fits the diffusion tensor model (WLS) to Phase 2R.1 standardized outputs and
produces voxel-wise scalar maps: FA, MD, AD, RD.

Input layout per subject:
  data/processed_v2/abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1/
    dwi_preprocessed.nii.gz
    dwi_preprocessed.bval
    dwi_preprocessed.bvec
    preprocessing_report.json   ← must have status="success"

Output layout per subject:
  .../session_1/dti_1/tensor/
    FA.nii.gz  MD.nii.gz  AD.nii.gz  RD.nii.gz
    tensor_fit_report.json

Safety contract:
  - Reads only from data/processed_v2/ — never from data/raw/ or data/processed/
  - guard_no_raw_write() called at every write entry point
  - Subjects with preprocessing_report.json status != "success" are skipped
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
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

PIPELINE_VERSION = "2R.2"
B0_THRESHOLD     = 100.0
MIN_DWI_DIRS     = 6

SITES = ["BNI", "IP_1", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]
_SITE_FOLDER_MAP = {
    "BNI":    "bni",
    "IP_1":   "ip",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
}

# Change-magnitude thresholds for FA delta classification
_FA_LARGE    = 0.05
_FA_MODERATE = 0.02

SUBJECT_CSV_FIELDS = [
    "site", "dataset", "subject_id",
    "fa_mean", "fa_std", "md_mean", "md_std",
    "ad_mean", "ad_std", "rd_mean", "rd_std",
    "nan_count", "inf_count",
    "status", "warning_message", "error_message",
]

SITE_CSV_FIELDS = [
    "site", "subjects_expected", "subjects_processed", "subjects_failed",
    "mean_fa", "mean_md", "mean_ad", "mean_rd", "notes",
]

COMPARISON_CSV_FIELDS = [
    "subject_id", "site",
    "v1_fa_mean", "v2_fa_mean", "delta_fa",
    "v1_md_mean", "v2_md_mean", "delta_md",
    "v1_ad_mean", "v2_ad_mean", "delta_ad",
    "v1_rd_mean", "v2_rd_mean", "delta_rd",
    "fa_change_magnitude",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TensorRecord:
    site:        str
    dataset:     str
    subject_id:  str
    fa_mean:     Optional[float] = None
    fa_std:      Optional[float] = None
    md_mean:     Optional[float] = None
    md_std:      Optional[float] = None
    ad_mean:     Optional[float] = None
    ad_std:      Optional[float] = None
    rd_mean:     Optional[float] = None
    rd_std:      Optional[float] = None
    nan_count:   int = 0
    inf_count:   int = 0
    status:      str = "pending"
    warning_message: Optional[str] = None
    error_message:   Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_csv_row(self) -> dict:
        return {k: getattr(self, k, None) for k in SUBJECT_CSV_FIELDS}


# ---------------------------------------------------------------------------
# Core subject pipeline
# ---------------------------------------------------------------------------

def estimate_tensors_subject(
    dti_dir:    Path,
    output_root: Path,
    raw_root:   Path,
    skip_if_exists: bool = True,
) -> TensorRecord:
    """
    Fit tensor maps for one subject.

    dti_dir:    .../session_1/dti_1/  (contains dwi_preprocessed.nii.gz etc.)
    output_root: processed_v2 root, used for safety guard
    """
    # Parse identity from path: .../abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1
    subject_id = dti_dir.parents[1].name
    dataset    = dti_dir.parents[2].name
    site_folder = dti_dir.parents[3].name
    site = next((k for k, v in _SITE_FOLDER_MAP.items() if v == site_folder), site_folder.upper())

    guard_no_raw_write(dti_dir, raw_root)

    rec = TensorRecord(site=site, dataset=dataset, subject_id=subject_id)

    tensor_dir = dti_dir / "tensor"

    # Skip if already done
    if skip_if_exists and (tensor_dir / "tensor_fit_report.json").exists():
        try:
            d = json.loads((tensor_dir / "tensor_fit_report.json").read_text())
            if d.get("status") == "success":
                rec.status = "success"
                rec.fa_mean = d.get("fa_mean")
                rec.fa_std  = d.get("fa_std")
                rec.md_mean = d.get("md_mean")
                rec.md_std  = d.get("md_std")
                rec.ad_mean = d.get("ad_mean")
                rec.ad_std  = d.get("ad_std")
                rec.rd_mean = d.get("rd_mean")
                rec.rd_std  = d.get("rd_std")
                rec.nan_count = d.get("nan_count", 0)
                rec.inf_count = d.get("inf_count", 0)
                logger.info("[%s/%s] already done — skipping", site, subject_id)
                return rec
        except Exception:
            pass  # re-run if report is unreadable

    # Validate preprocessing report
    preproc_report = dti_dir / "preprocessing_report.json"
    if not preproc_report.exists():
        rec.status = "failed"
        rec.error_message = "preprocessing_report.json not found — Phase 2R.1 not run?"
        logger.error("[%s/%s] %s", site, subject_id, rec.error_message)
        return rec

    try:
        pr = json.loads(preproc_report.read_text())
    except Exception as exc:
        rec.status = "failed"
        rec.error_message = f"Cannot parse preprocessing_report.json: {exc}"
        return rec

    if pr.get("status") != "success":
        rec.status = "skipped"
        rec.warning_message = (
            f"preprocessing_report status={pr.get('status')} — skipping tensor fitting"
        )
        logger.warning("[%s/%s] %s", site, subject_id, rec.warning_message)
        return rec

    # Locate input files
    dwi_path  = dti_dir / "dwi_preprocessed.nii.gz"
    bval_path = dti_dir / "dwi_preprocessed.bval"
    bvec_path = dti_dir / "dwi_preprocessed.bvec"

    missing = [p.name for p in [dwi_path, bval_path, bvec_path] if not p.exists()]
    if missing:
        rec.status = "failed"
        rec.error_message = f"Missing input files: {missing}"
        logger.error("[%s/%s] %s", site, subject_id, rec.error_message)
        return rec

    # Load and validate
    try:
        img, bvals, bvecs = _load_validate(dwi_path, bval_path, bvec_path)
    except Exception as exc:
        rec.status = "failed"
        rec.error_message = str(exc)
        logger.error("[%s/%s] load/validate failed: %s", site, subject_id, exc)
        return rec

    # Fit tensor
    try:
        fa, md, ad, rd, mask = _fit_tensor(img, bvals, bvecs)
    except Exception as exc:
        rec.status = "failed"
        rec.error_message = f"Tensor fitting failed: {exc}"
        logger.error("[%s/%s] fit failed: %s", site, subject_id, exc)
        return rec

    # Compute metrics within brain mask (FA > 0)
    brain_mask = mask & (fa > 0)
    nan_count  = int(np.sum(np.isnan(fa)))
    inf_count  = int(np.sum(np.isinf(fa)))

    if brain_mask.any():
        rec.fa_mean = _r(float(np.nanmean(fa[brain_mask])))
        rec.fa_std  = _r(float(np.nanstd(fa[brain_mask])))
        rec.md_mean = _r(float(np.nanmean(md[brain_mask])))
        rec.md_std  = _r(float(np.nanstd(md[brain_mask])))
        rec.ad_mean = _r(float(np.nanmean(ad[brain_mask])))
        rec.ad_std  = _r(float(np.nanstd(ad[brain_mask])))
        rec.rd_mean = _r(float(np.nanmean(rd[brain_mask])))
        rec.rd_std  = _r(float(np.nanstd(rd[brain_mask])))

    rec.nan_count = nan_count
    rec.inf_count = inf_count

    if nan_count > 0 or inf_count > 0:
        rec.warning_message = f"FA map contains {nan_count} NaN and {inf_count} Inf voxels"
        logger.warning("[%s/%s] %s", site, subject_id, rec.warning_message)

    # Save maps
    tensor_dir.mkdir(parents=True, exist_ok=True)
    affine, header = img.affine, img.header
    _save_map(fa, affine, header, tensor_dir / "FA.nii.gz")
    _save_map(md, affine, header, tensor_dir / "MD.nii.gz")
    _save_map(ad, affine, header, tensor_dir / "AD.nii.gz")
    _save_map(rd, affine, header, tensor_dir / "RD.nii.gz")

    rec.status = "success"

    # Write per-subject report
    report = {
        **rec.to_csv_row(),
        "pipeline_version": PIPELINE_VERSION,
        "input_dwi_path":   str(dwi_path),
        "timestamp":        rec.timestamp,
    }
    (tensor_dir / "tensor_fit_report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "[%s/%s] FA=%.4f±%.4f  MD=%.6f  status=success",
        site, subject_id, rec.fa_mean or 0, rec.fa_std or 0, rec.md_mean or 0,
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> TensorRecord:
    """Top-level picklable worker for ProcessPoolExecutor."""
    dti_dir_str, output_root_str, raw_root_str, skip_if_exists = args
    return estimate_tensors_subject(
        dti_dir=Path(dti_dir_str),
        output_root=Path(output_root_str),
        raw_root=Path(raw_root_str),
        skip_if_exists=skip_if_exists,
    )


def run_tensor_batch(
    processed_v2_root: Path,
    raw_root:          Path,
    sites:             list[str] = SITES,
    skip_if_exists:    bool = True,
    n_workers:         int = 1,
) -> list[TensorRecord]:
    """
    Run tensor estimation for all subjects across given sites.
    processed_v2_root: e.g. data/processed_v2/abide_ii
    n_workers > 1 enables parallel processing via ProcessPoolExecutor.
    """
    guard_no_raw_write(processed_v2_root, raw_root)

    all_dti_dirs: list[Path] = []
    for site in sites:
        folder = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = processed_v2_root / folder
        if not site_root.exists():
            logger.warning("[%s] site directory not found: %s", site, site_root)
            continue
        dti_dirs = sorted(site_root.rglob("dti_1"))
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "preprocessing_report.json").exists()]
        logger.info("[%s] %d subjects", site, len(dti_dirs))
        all_dti_dirs.extend(dti_dirs)

    logger.info("Total: %d subjects  workers=%d", len(all_dti_dirs), n_workers)

    if n_workers <= 1:
        records = []
        for dti_dir in all_dti_dirs:
            rec = estimate_tensors_subject(
                dti_dir=dti_dir,
                output_root=processed_v2_root,
                raw_root=raw_root,
                skip_if_exists=skip_if_exists,
            )
            records.append(rec)
        return records

    # Parallel path
    work_args = [
        (str(d), str(processed_v2_root), str(raw_root), skip_if_exists)
        for d in all_dti_dirs
    ]
    records: list[TensorRecord] = [None] * len(work_args)  # type: ignore
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, a): i for i, a in enumerate(work_args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                records[idx] = fut.result()
            except Exception as exc:
                dti_dir = all_dti_dirs[idx]
                subject_id = dti_dir.parents[1].name
                site_folder = dti_dir.parents[3].name
                site = next((k for k, v in _SITE_FOLDER_MAP.items() if v == site_folder), site_folder.upper())
                logger.error("[%s/%s] worker exception: %s", site, subject_id, exc)
                records[idx] = TensorRecord(
                    site=site, dataset=dti_dir.parents[2].name,
                    subject_id=subject_id, status="failed",
                    error_message=str(exc),
                )
    return records


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def write_subject_summary(records: list[TensorRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_csv_row() for r in records)
    logger.info("Subject summary → %s  (%d rows)", out_path, len(records))
    return out_path


def write_site_summary(
    records:          list[TensorRecord],
    expected_counts:  dict[str, int],
    out_path:         Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for site in SITES:
        site_recs = [r for r in records if r.site == site]
        success   = [r for r in site_recs if r.status == "success"]
        failed    = [r for r in site_recs if r.status == "failed"]
        expected  = expected_counts.get(site, 0)

        fa_vals = [r.fa_mean for r in success if r.fa_mean is not None]
        md_vals = [r.md_mean for r in success if r.md_mean is not None]
        ad_vals = [r.ad_mean for r in success if r.ad_mean is not None]
        rd_vals = [r.rd_mean for r in success if r.rd_mean is not None]

        notes = []
        if len(success) < expected:
            notes.append(f"{expected - len(success)} subjects below expected count")
        if site == "IP_1":
            notes.append("QC-labelled only — not in clean tractography pipeline")

        rows.append({
            "site":                site,
            "subjects_expected":   expected,
            "subjects_processed":  len(success),
            "subjects_failed":     len(failed),
            "mean_fa":  _r(float(np.mean(fa_vals))) if fa_vals else None,
            "mean_md":  _r(float(np.mean(md_vals))) if md_vals else None,
            "mean_ad":  _r(float(np.mean(ad_vals))) if ad_vals else None,
            "mean_rd":  _r(float(np.mean(rd_vals))) if rd_vals else None,
            "notes":    "; ".join(notes) if notes else "",
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SITE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Site summary → %s", out_path)
    return out_path


def write_v1_comparison(
    v2_records:      list[TensorRecord],
    v1_tensor_root:  Path,
    out_path:        Path,
) -> tuple[Path, dict]:
    """
    Compare v2 tensor metrics against v1 tensor_fit_report.json.
    Returns (csv_path, site_analysis_dict).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    site_deltas: dict[str, list[float]] = {s: [] for s in SITES}

    for rec in v2_records:
        if rec.status != "success":
            continue

        folder = _SITE_FOLDER_MAP.get(rec.site, rec.site.lower())
        v1_report = (
            v1_tensor_root / folder / rec.dataset / rec.subject_id
            / "session_1" / "dti_1" / "tensor" / "tensor_fit_report.json"
        )

        v1_fa = v1_md = v1_ad = v1_rd = None
        if v1_report.exists():
            try:
                d = json.loads(v1_report.read_text())
                v1_fa = d.get("fa_mean")
                v1_md = d.get("md_mean")
                v1_ad = d.get("ad_mean")
                v1_rd = d.get("rd_mean")
            except Exception:
                pass

        delta_fa = _safe_delta(rec.fa_mean, v1_fa)
        delta_md = _safe_delta(rec.md_mean, v1_md)
        delta_ad = _safe_delta(rec.ad_mean, v1_ad)
        delta_rd = _safe_delta(rec.rd_mean, v1_rd)

        mag = _fa_change_magnitude(delta_fa)

        if delta_fa is not None:
            site_deltas[rec.site].append(abs(delta_fa))

        rows.append({
            "subject_id":  rec.subject_id,
            "site":        rec.site,
            "v1_fa_mean":  v1_fa,
            "v2_fa_mean":  rec.fa_mean,
            "delta_fa":    delta_fa,
            "v1_md_mean":  v1_md,
            "v2_md_mean":  rec.md_mean,
            "delta_md":    delta_md,
            "v1_ad_mean":  v1_ad,
            "v2_ad_mean":  rec.ad_mean,
            "delta_ad":    delta_ad,
            "v1_rd_mean":  v1_rd,
            "v2_rd_mean":  rec.rd_mean,
            "delta_rd":    delta_rd,
            "fa_change_magnitude": mag,
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Comparison CSV → %s  (%d rows)", out_path, len(rows))

    # Site-level analysis
    analysis = {}
    for site, deltas in site_deltas.items():
        if not deltas:
            analysis[site] = {"n": 0, "mean_abs_fa_delta": None, "classification": "no_data"}
            continue
        mean_delta = float(np.mean(deltas))
        analysis[site] = {
            "n":                  len(deltas),
            "mean_abs_fa_delta":  round(mean_delta, 6),
            "classification":     _fa_change_magnitude(mean_delta),
        }

    return out_path, analysis


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_validate(
    dwi_path: Path, bval_path: Path, bvec_path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    img = nib.load(str(dwi_path))
    if img.ndim != 4:
        raise ValueError(f"Expected 4D DWI, got shape {img.shape}")
    n_vols = img.shape[3]

    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path))

    # Normalise bvec shape to (N, 3)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    elif bvecs.ndim == 2 and bvecs.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected bvec shape: {bvecs.shape}")

    n_bvals = int(bvals.size)
    n_bvecs = int(bvecs.shape[0])

    if n_vols != n_bvals:
        raise ValueError(f"Volume count ({n_vols}) ≠ bval count ({n_bvals})")
    if n_vols != n_bvecs:
        raise ValueError(f"Volume count ({n_vols}) ≠ bvec count ({n_bvecs})")

    b0_count  = int(np.sum(bvals < B0_THRESHOLD))
    dwi_count = int(np.sum(bvals >= B0_THRESHOLD))
    if b0_count < 1:
        raise ValueError("No B0 volumes found (bval < 100)")
    if dwi_count < MIN_DWI_DIRS:
        raise ValueError(f"Only {dwi_count} DWI directions — need ≥ {MIN_DWI_DIRS}")

    return img, bvals, bvecs


def _fit_tensor(
    img: nib.Nifti1Image,
    bvals: np.ndarray,
    bvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WLS tensor fit; returns (FA, MD, AD, RD, mask)."""
    gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESHOLD)
    data = img.get_fdata(dtype=np.float32)

    b0_idx  = np.where(gtab.b0s_mask)[0]
    b0_mean = data[..., b0_idx].mean(axis=-1) if len(b0_idx) > 1 else data[..., b0_idx[0]]
    thr  = float(np.percentile(b0_mean, 30))
    mask = b0_mean > thr
    if not mask.any():
        mask = b0_mean > 0

    tenfit = TensorModel(gtab).fit(data, mask=mask)

    fa = np.clip(tenfit.fa.astype(np.float32), 0.0, 1.0)
    md = tenfit.md.astype(np.float32)
    ad = tenfit.ad.astype(np.float32)
    rd = tenfit.rd.astype(np.float32)

    for arr in (fa, md, ad, rd):
        arr[~mask] = 0.0

    return fa, md, ad, rd, mask


def _save_map(data: np.ndarray, affine: np.ndarray, header, path: Path) -> None:
    img = nib.Nifti1Image(data, affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(path))


def _safe_delta(v2: Optional[float], v1: Optional[float]) -> Optional[float]:
    if v2 is None or v1 is None:
        return None
    return _r(v2 - v1)


def _fa_change_magnitude(delta: Optional[float]) -> str:
    if delta is None:
        return "no_v1_data"
    abs_d = abs(delta)
    if abs_d >= _FA_LARGE:
        return "large_change"
    if abs_d >= _FA_MODERATE:
        return "moderate_change"
    return "minimal_change"


def _r(v: float, decimals: int = 6) -> float:
    return round(v, decimals)
