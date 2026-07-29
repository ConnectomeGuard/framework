"""
Phase 3.2 — Multi-site Streamline Tractography MVP

Generates reproducible whole-brain streamlines from Phase 3.1 peak directions
using DIPY deterministic local tracking.

Per-subject outputs written to dti_1/tractography/:
  streamlines.trk          DIPY StatefulTractogram (RASMM space)
  tractography_report.json per-subject stats and QC flags
  backend_used.txt         "dipy_deterministic"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.peaks import load_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Site name normalisation  (display name → local folder name)
# ---------------------------------------------------------------------------

_DISPLAY_TO_FOLDER: dict[str, str] = {
    "BNI":      "bni",
    "ETH":      "eth",
    "EMC":      "emc",
    "GU":       "gu",
    "IP_1":     "ip",
    "IU_1":     "iu",
    "KKI_1":    "kki",
    "KUL_3":    "kul",
    "NYU_1":    "nyu1",
    "NYU_2":    "nyu2",
    "OHSU_1":   "ohsu",
    "ONRC_2":   "onrc",
    "SDSU_1":   "sdsu",
    "TCD_1":    "tcd",
    "UCD_1":    "ucd",
    "UCLA_1":   "ucla1",
    "UCLA_Long":"ucla_long",
    "UPSM_Long":"upsm_long",
    "USM_1":    "usm",
}

CLEAN_DTI_SITES_DISPLAY: list[str] = ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]


def site_to_folder(name: str) -> str:
    """Convert display name (BNI, NYU_1) or folder name to local folder name."""
    if name in _DISPLAY_TO_FOLDER:
        return _DISPLAY_TO_FOLDER[name]
    return name.lower()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and validate YAML tractography config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    required = {"included_sites", "fa_threshold", "seeds_per_subject", "step_size"}
    missing  = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StreamlineRecord:
    site: str
    dataset: str
    subject_id: str
    fa_mean: float
    fa_threshold: float
    seed_count: int
    streamline_count: int
    mean_streamline_length: Optional[float]
    median_streamline_length: Optional[float]
    min_streamline_length: Optional[float]
    max_streamline_length: Optional[float]
    output_file_size_mb: Optional[float]
    status: str            # "success" | "failed"
    warning_message: Optional[str] = field(default=None)
    error_message: Optional[str] = field(default=None)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        return {
            "site":                    self.site,
            "dataset":                 self.dataset,
            "subject_id":              self.subject_id,
            "fa_mean":                 round(self.fa_mean, 4),
            "fa_threshold":            self.fa_threshold,
            "seed_count":              self.seed_count,
            "streamline_count":        self.streamline_count,
            "mean_streamline_length":  _round_opt(self.mean_streamline_length, 2),
            "median_streamline_length":_round_opt(self.median_streamline_length, 2),
            "min_streamline_length":   _round_opt(self.min_streamline_length, 2),
            "max_streamline_length":   _round_opt(self.max_streamline_length, 2),
            "output_file_size_mb":     _round_opt(self.output_file_size_mb, 3),
            "status":                  self.status,
            "warning_message":         self.warning_message,
            "error_message":           self.error_message,
        }


# ---------------------------------------------------------------------------
# Core per-subject function
# ---------------------------------------------------------------------------

def generate_subject_streamlines(
    dti_dir: Path,
    output_dir: Path,
    raw_root: Path,
    fa_threshold: float = 0.15,
    seeds_per_subject: int = 5000,
    step_size: float = 0.5,
    max_angle: float = 30.0,
    max_cross: int = 1,
    random_seed: int = 42,
    qc_min_streamlines: int = 500,
    qc_min_mean_length: float = 20.0,
    qc_max_file_mb: float = 500.0,
) -> StreamlineRecord:
    """
    Generate deterministic streamlines for one subject.

    Parameters
    ----------
    dti_dir :
        Subject's dti_1/ directory (contains tensor/, qc/, fod/).
    output_dir :
        Destination for tractography/ outputs.
    raw_root :
        Raw data root — safety guard.
    fa_threshold :
        FA below which tracking stops.
    seeds_per_subject :
        Total number of seed points (placed randomly in WM mask).
    step_size :
        Propagation step size in mm.
    max_angle :
        Maximum turning angle (degrees) between consecutive steps.
    max_cross :
        Maximum number of crossing directions to follow per step.
    random_seed :
        NumPy random seed for reproducible seed placement.
    """
    subject_id = dti_dir.parent.parent.name
    dataset    = dti_dir.parent.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name

    guard_no_raw_write(output_dir, raw_root)

    # --- validate required inputs ---
    required = {
        "FA.nii.gz":       dti_dir / "tensor" / "FA.nii.gz",
        "brain_mask.nii.gz": dti_dir / "qc" / "brain_mask.nii.gz",
        "peaks.pam5":      dti_dir / "fod" / "peaks.pam5",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            f"Missing required inputs: {missing}",
        )

    # --- load data ---
    try:
        pam      = load_peaks(str(dti_dir / "fod" / "peaks.pam5"))
        fa_img   = nib.load(str(dti_dir / "tensor" / "FA.nii.gz"))
        mask_img = nib.load(str(dti_dir / "qc" / "brain_mask.nii.gz"))
    except Exception as exc:
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            f"Failed to load inputs: {exc}",
        )

    fa       = fa_img.get_fdata(dtype=np.float32)
    mask_arr = np.asarray(mask_img.dataobj).astype(bool)
    fa_mean  = float(fa[mask_arr].mean()) if mask_arr.any() else 0.0

    # WM-like seed mask: FA above threshold, inside brain mask
    wm_mask = (fa >= fa_threshold) & mask_arr

    if not wm_mask.any():
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            f"WM seed mask is empty at FA threshold={fa_threshold}",
            fa_mean=fa_mean,
        )

    try:
        # Reproducible seed placement
        np.random.seed(random_seed)
        seeds = random_seeds_from_mask(
            wm_mask, pam.affine,
            seeds_count=seeds_per_subject,
            seed_count_per_voxel=False,
        )
        seed_count = len(seeds)

        # Direction getter from spherical harmonic coefficients
        getter = DeterministicMaximumDirectionGetter.from_shcoeff(
            pam.shm_coeff,
            max_angle=max_angle,
            sphere=pam.sphere,
        )

        # Stopping criterion
        stopping = ThresholdStoppingCriterion(fa, fa_threshold)

        # Track
        streamline_gen = LocalTracking(
            getter, stopping, seeds, pam.affine,
            step_size=step_size,
            max_cross=max_cross,
            return_all=False,
        )
        streamlines = Streamlines(streamline_gen)

    except Exception as exc:
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            f"Tracking failed: {exc}",
            fa_mean=fa_mean,
        )

    streamline_count = len(streamlines)

    # --- compute length statistics ---
    if streamline_count > 0:
        lengths = _compute_lengths_mm(streamlines)
        mean_len   = float(np.mean(lengths))
        median_len = float(np.median(lengths))
        min_len    = float(np.min(lengths))
        max_len    = float(np.max(lengths))
    else:
        lengths = []
        mean_len = median_len = min_len = max_len = None

    # --- save streamlines ---
    output_dir.mkdir(parents=True, exist_ok=True)
    trk_path = output_dir / "streamlines.trk"
    try:
        sft = StatefulTractogram(streamlines, fa_img, Space.RASMM)
        save_trk(sft, str(trk_path))
        file_size_mb = trk_path.stat().st_size / (1024 ** 2)
    except Exception as exc:
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            f"Failed to save streamlines: {exc}",
            fa_mean=fa_mean, seed_count=seed_count,
            streamline_count=streamline_count,
        )

    # --- QC flags ---
    warnings: list[str] = []
    if streamline_count == 0:
        return _failed_record(
            site, dataset, subject_id, fa_threshold,
            "Zero streamlines generated",
            fa_mean=fa_mean, seed_count=seed_count,
        )
    if streamline_count < qc_min_streamlines:
        warnings.append(
            f"streamline_count={streamline_count} < {qc_min_streamlines}"
        )
    if mean_len is not None and mean_len < qc_min_mean_length:
        warnings.append(
            f"mean_length={mean_len:.1f}mm < {qc_min_mean_length}mm"
        )
    if file_size_mb > qc_max_file_mb:
        warnings.append(
            f"file_size={file_size_mb:.1f}MB > {qc_max_file_mb}MB"
        )
    warning_str = "; ".join(warnings) if warnings else None

    rec = StreamlineRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        fa_mean=fa_mean, fa_threshold=fa_threshold,
        seed_count=seed_count, streamline_count=streamline_count,
        mean_streamline_length=mean_len,
        median_streamline_length=median_len,
        min_streamline_length=min_len,
        max_streamline_length=max_len,
        output_file_size_mb=file_size_mb,
        status="success",
        warning_message=warning_str,
    )

    (output_dir / "tractography_report.json").write_text(
        json.dumps(rec.to_dict(), indent=2)
    )
    (output_dir / "backend_used.txt").write_text("dipy_deterministic\n")

    logger.info(
        "[%s/%s] %d seeds → %d streamlines  mean_len=%.1fmm  size=%.2fMB%s",
        site, subject_id, seed_count, streamline_count, mean_len or 0,
        file_size_mb,
        f"  WARN: {warning_str}" if warning_str else "",
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_streamline_batch(
    processed_root: Path,
    raw_root: Path,
    sites: list[str],
    fa_threshold: float = 0.15,
    seeds_per_subject: int = 5000,
    step_size: float = 0.5,
    max_angle: float = 30.0,
    max_cross: int = 1,
    random_seed: int = 42,
    qc_min_streamlines: int = 500,
    qc_min_mean_length: float = 20.0,
    qc_max_file_mb: float = 500.0,
) -> list[StreamlineRecord]:
    """
    Run streamline generation for all subjects across the given sites.

    Sites should be given as display names (BNI, NYU_1) or folder names (bni, nyu1).
    A failure in one subject is caught and recorded; the batch continues.
    """
    guard_no_raw_write(processed_root, raw_root)

    folder_names = [site_to_folder(s) for s in sites]
    records: list[StreamlineRecord] = []

    for folder_name in folder_names:
        site_dir = processed_root / folder_name
        if not site_dir.is_dir():
            logger.warning("[%s] site directory not found: %s", folder_name, site_dir)
            continue

        subject_dirs = []
        for dataset_dir in sorted(site_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for subject_dir in sorted(dataset_dir.iterdir()):
                dti_dir = subject_dir / "session_1" / "dti_1"
                if dti_dir.is_dir():
                    subject_dirs.append(dti_dir)

        logger.info("[%s] tracking %d subjects …", folder_name, len(subject_dirs))

        for dti_dir in subject_dirs:
            try:
                rec = generate_subject_streamlines(
                    dti_dir=dti_dir,
                    output_dir=dti_dir / "tractography",
                    raw_root=raw_root,
                    fa_threshold=fa_threshold,
                    seeds_per_subject=seeds_per_subject,
                    step_size=step_size,
                    max_angle=max_angle,
                    max_cross=max_cross,
                    random_seed=random_seed,
                    qc_min_streamlines=qc_min_streamlines,
                    qc_min_mean_length=qc_min_mean_length,
                    qc_max_file_mb=qc_max_file_mb,
                )
            except Exception as exc:
                # Catch unexpected errors to keep the batch going
                subject_id = dti_dir.parent.parent.name
                dataset    = dti_dir.parent.parent.parent.name
                logger.error(
                    "[%s/%s] unexpected error: %s", folder_name, subject_id, exc
                )
                rec = _failed_record(
                    folder_name, dataset, subject_id, fa_threshold,
                    f"Unexpected error: {exc}",
                )
            records.append(rec)

        n_ok   = sum(1 for r in records if r.site == folder_name and r.status == "success")
        n_fail = sum(1 for r in records if r.site == folder_name and r.status == "failed")
        logger.info("[%s] done — success=%d  failed=%d", folder_name, n_ok, n_fail)

    return records


# ---------------------------------------------------------------------------
# Summary CSV export
# ---------------------------------------------------------------------------

def save_summary_csvs(
    records: list[StreamlineRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write run and site-level summary CSVs. Returns (run_csv, site_csv)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_csv = output_dir / "phase3_2_streamline_generation_summary.csv"
    pd.DataFrame([r.to_summary_row() for r in records]).to_csv(run_csv, index=False)

    site_groups: dict[str, list[StreamlineRecord]] = {}
    for r in records:
        site_groups.setdefault(r.site, []).append(r)

    site_rows = []
    for site, recs in site_groups.items():
        success = [r for r in recs if r.status == "success"]
        failed  = [r for r in recs if r.status == "failed"]
        sl_counts = [r.streamline_count for r in success]
        sl_lens   = [r.mean_streamline_length for r in success
                     if r.mean_streamline_length is not None]
        fa_vals   = [r.fa_mean for r in success if r.fa_mean > 0]
        site_rows.append({
            "site":                    site,
            "subjects_considered":     len(recs),
            "subjects_processed":      len(success),
            "subjects_failed":         len(failed),
            "mean_streamline_count":   round(np.mean(sl_counts), 0) if sl_counts else None,
            "mean_streamline_length":  round(np.mean(sl_lens), 2)   if sl_lens   else None,
            "mean_fa":                 round(np.mean(fa_vals), 4)    if fa_vals   else None,
            "notes":                   "",
        })

    site_csv = output_dir / "phase3_2_streamline_site_summary.csv"
    pd.DataFrame(site_rows).to_csv(site_csv, index=False)

    logger.info("Run summary  → %s  (%d rows)", run_csv.name, len(records))
    logger.info("Site summary → %s  (%d sites)", site_csv.name, len(site_rows))
    return run_csv, site_csv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_lengths_mm(streamlines: Streamlines) -> list[float]:
    """Return per-streamline length in mm."""
    lengths = []
    for s in streamlines:
        if len(s) < 2:
            lengths.append(0.0)
        else:
            diffs = np.diff(s, axis=0)
            lengths.append(float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))))
    return lengths


def _round_opt(v: Optional[float], ndigits: int) -> Optional[float]:
    return round(v, ndigits) if v is not None else None


def _failed_record(
    site: str,
    dataset: str,
    subject_id: str,
    fa_threshold: float,
    error: str,
    fa_mean: float = 0.0,
    seed_count: int = 0,
    streamline_count: int = 0,
) -> StreamlineRecord:
    logger.error("[%s/%s] failed: %s", site, subject_id, error)
    return StreamlineRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        fa_mean=fa_mean, fa_threshold=fa_threshold,
        seed_count=seed_count, streamline_count=streamline_count,
        mean_streamline_length=None, median_streamline_length=None,
        min_streamline_length=None, max_streamline_length=None,
        output_file_size_mb=None,
        status="failed", error_message=error,
    )
