"""
Phase 3.5A — Deterministic Tractography with Interface Seeding

Uses the same DeterministicMaximumDirectionGetter + LocalTracking pipeline as
Phase 3.2, but seeds at the WM-GM interface rather than deep white matter.

Key differences from Phase 3.2 (streamline_generation.py):
  Seeds   : interface mask  fa_low ≤ FA < fa_high  (default [0.08, 0.20])
            vs WM mask      FA ≥ fa_threshold       (Phase 3.2 default 0.15)
  Count   : 10 000 seeds per subject  (vs 5 000)
  Output  : dti_1/tractography_det/  (not tractography/)

Interface seeding places one endpoint of each streamline near a cortical
parcel boundary, substantially improving the atlas-mapping ratio compared
with deep-WM seeding.
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

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.peaks import load_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask

from neurofiber.tractography.streamline_generation import site_to_folder, CLEAN_DTI_SITES_DISPLAY
from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

TRACTOGRAPHY_SUBDIR = "tractography_det"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetRecord:
    site:        str
    dataset:     str
    subject_id:  str

    fa_threshold:      float
    interface_fa_low:  float
    interface_fa_high: float

    seed_count:         int
    streamline_count:   int
    mean_length_mm:     Optional[float]
    median_length_mm:   Optional[float]
    min_length_mm:      Optional[float]
    max_length_mm:      Optional[float]
    output_file_size_mb: Optional[float]

    status:          str = "success"
    warning_message: Optional[str] = None
    error_message:   Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        d = self.to_dict()
        for k in ("mean_length_mm", "median_length_mm", "min_length_mm",
                  "max_length_mm", "output_file_size_mb"):
            if d[k] is not None:
                d[k] = round(d[k], 2)
        return d


# ---------------------------------------------------------------------------
# Core per-subject function
# ---------------------------------------------------------------------------

def generate_det_streamlines(
    dti_dir:          Path,
    output_dir:       Path,
    raw_root:         Path,
    fa_threshold:     float = 0.10,
    interface_fa_low: float = 0.08,
    interface_fa_high: float = 0.20,
    seeds_per_subject: int  = 10_000,
    step_size:        float = 0.5,
    max_angle:        float = 30.0,
    max_cross:        int   = 1,
    random_seed:      int   = 42,
    skip_if_exists:   bool  = False,
) -> DetRecord:
    """
    Generate interface-seeded deterministic streamlines for one subject.

    Seeds are placed in voxels where FA is in [interface_fa_low, interface_fa_high]
    — the white/grey matter transition zone. Tracking stops when FA drops below
    fa_threshold. Uses the same CSD peaks.pam5 as Phase 3.2.
    """
    subject_id = dti_dir.parent.parent.name
    dataset    = dti_dir.parent.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name

    guard_no_raw_write(output_dir, raw_root)

    def _fail(msg: str, seed_count: int = 0, sl_count: int = 0) -> DetRecord:
        logger.error("[%s/%s] det-tractography failed: %s", site, subject_id, msg)
        return DetRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            fa_threshold=fa_threshold,
            interface_fa_low=interface_fa_low, interface_fa_high=interface_fa_high,
            seed_count=seed_count, streamline_count=sl_count,
            mean_length_mm=None, median_length_mm=None,
            min_length_mm=None, max_length_mm=None,
            output_file_size_mb=None,
            status="failed", error_message=msg,
        )

    # Skip if already done
    report_path = output_dir / "tractography_report.json"
    if skip_if_exists and report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
            if existing.get("status") == "success":
                logger.debug("[%s/%s] det-tractography already done — skipping", site, subject_id)
                return _load_record_from_report(existing)
        except Exception:
            pass

    # Required inputs
    fa_path   = dti_dir / "tensor" / "FA.nii.gz"
    mask_path = dti_dir / "qc"     / "brain_mask.nii.gz"
    peaks_path = dti_dir / "fod"   / "peaks.pam5"

    for p in [fa_path, mask_path, peaks_path]:
        if not p.exists():
            return _fail(f"Missing input: {p.name}")

    try:
        pam      = load_peaks(str(peaks_path))
        fa_img   = nib.load(str(fa_path))
        mask_img = nib.load(str(mask_path))
    except Exception as exc:
        return _fail(f"Load error: {exc}")

    fa   = fa_img.get_fdata(dtype=np.float32)
    mask = np.asarray(mask_img.dataobj).astype(bool)

    # Interface seed mask: FA in [fa_low, fa_high] AND inside brain
    interface_mask = (fa >= interface_fa_low) & (fa < interface_fa_high) & mask

    if not interface_mask.any():
        return _fail(
            f"Interface seed mask empty (FA range [{interface_fa_low}, {interface_fa_high}))"
        )

    try:
        np.random.seed(random_seed)
        seeds = random_seeds_from_mask(
            interface_mask, pam.affine,
            seeds_count=seeds_per_subject,
            seed_count_per_voxel=False,
        )

        getter = DeterministicMaximumDirectionGetter.from_shcoeff(
            pam.shm_coeff,
            max_angle=max_angle,
            sphere=pam.sphere,
        )

        stopping = ThresholdStoppingCriterion(fa, fa_threshold)

        streamline_gen = LocalTracking(
            getter, stopping, seeds, pam.affine,
            step_size=step_size,
            max_cross=max_cross,
            return_all=False,
        )
        streamlines = Streamlines(streamline_gen)

    except Exception as exc:
        return _fail(f"Tracking error: {exc}", seed_count=len(seeds) if 'seeds' in dir() else 0)

    n_sl = len(streamlines)
    if n_sl == 0:
        return _fail("Zero streamlines generated", seed_count=len(seeds))

    lengths = _compute_lengths(streamlines)
    mean_l = float(np.mean(lengths))
    med_l  = float(np.median(lengths))
    min_l  = float(np.min(lengths))
    max_l  = float(np.max(lengths))

    output_dir.mkdir(parents=True, exist_ok=True)
    trk_path = output_dir / "streamlines.trk"
    try:
        sft = StatefulTractogram(streamlines, fa_img, Space.RASMM)
        save_trk(sft, str(trk_path))
        file_mb = trk_path.stat().st_size / 1024 ** 2
    except Exception as exc:
        return _fail(f"Save error: {exc}", seed_count=len(seeds), sl_count=n_sl)

    warnings = []
    if n_sl < 500:
        warnings.append(f"streamline_count={n_sl} < 500")
    if mean_l < 15.0:
        warnings.append(f"mean_length={mean_l:.1f}mm < 15mm")

    rec = DetRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        fa_threshold=fa_threshold,
        interface_fa_low=interface_fa_low, interface_fa_high=interface_fa_high,
        seed_count=len(seeds), streamline_count=n_sl,
        mean_length_mm=mean_l, median_length_mm=med_l,
        min_length_mm=min_l, max_length_mm=max_l,
        output_file_size_mb=file_mb,
        warning_message="; ".join(warnings) if warnings else None,
    )
    report_path.write_text(json.dumps(rec.to_dict(), indent=2))
    (output_dir / "backend_used.txt").write_text("dipy_deterministic_interface_seeded\n")

    logger.info(
        "[%s/%s] det %d seeds → %d streamlines  mean=%.1fmm  %.2fMB%s",
        site, subject_id, len(seeds), n_sl, mean_l, file_mb,
        f"  WARN: {rec.warning_message}" if rec.warning_message else "",
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_det_batch(
    processed_root:   Path,
    raw_root:         Path,
    sites:            list[str],
    fa_threshold:     float = 0.10,
    interface_fa_low: float = 0.08,
    interface_fa_high: float = 0.20,
    seeds_per_subject: int  = 10_000,
    step_size:        float = 0.5,
    max_angle:        float = 30.0,
    random_seed:      int   = 42,
    skip_if_exists:   bool  = True,
) -> list[DetRecord]:
    """Run interface-seeded deterministic tractography for all subjects."""
    guard_no_raw_write(processed_root, raw_root)
    records: list[DetRecord] = []

    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = processed_root / folder
        if not site_dir.exists():
            logger.warning("[%s] site directory not found: %s", folder, site_dir)
            continue

        dti_dirs = sorted(site_dir.glob("*/*/session_1/dti_1"))
        if not dti_dirs:
            logger.warning("[%s] no dti_1 dirs found", folder)
            continue

        logger.info("[%s] det-tractography %d subjects …", folder, len(dti_dirs))

        for dti_dir in dti_dirs:
            try:
                rec = generate_det_streamlines(
                    dti_dir=dti_dir,
                    output_dir=dti_dir / TRACTOGRAPHY_SUBDIR,
                    raw_root=raw_root,
                    fa_threshold=fa_threshold,
                    interface_fa_low=interface_fa_low,
                    interface_fa_high=interface_fa_high,
                    seeds_per_subject=seeds_per_subject,
                    step_size=step_size,
                    max_angle=max_angle,
                    random_seed=random_seed,
                    skip_if_exists=skip_if_exists,
                )
            except Exception as exc:
                sid = dti_dir.parent.parent.name
                rec = DetRecord(
                    site=folder, dataset=dti_dir.parent.parent.parent.name,
                    subject_id=sid,
                    fa_threshold=fa_threshold,
                    interface_fa_low=interface_fa_low, interface_fa_high=interface_fa_high,
                    seed_count=0, streamline_count=0,
                    mean_length_mm=None, median_length_mm=None,
                    min_length_mm=None, max_length_mm=None,
                    output_file_size_mb=None,
                    status="failed", error_message=str(exc),
                )
            records.append(rec)

        n_ok = sum(1 for r in records if r.site == folder and r.status == "success")
        n_fail = sum(1 for r in records if r.site == folder and r.status != "success")
        logger.info("[%s] done  success=%d  failed=%d", folder, n_ok, n_fail)

    return records


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_det_tractography_csvs(
    records:    list[DetRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write per-subject and per-site tractography summary CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_summary_row() for r in records])

    subj_csv = output_dir / "det_tractography_summary.csv"
    df.to_csv(subj_csv, index=False)

    ok = df[df["status"] == "success"]
    site_rows = []
    for site, grp in ok.groupby("site"):
        site_rows.append({
            "site":               site,
            "subjects_processed": len(grp),
            "subjects_failed":    int((df[df["site"] == site]["status"] != "success").sum()),
            "mean_streamline_count": round(grp["streamline_count"].mean(), 0),
            "mean_length_mm":    round(grp["mean_length_mm"].mean(), 2),
            "median_length_mm":  round(grp["median_length_mm"].mean(), 2),
        })
    site_csv = output_dir / "det_tractography_site_summary.csv"
    pd.DataFrame(site_rows).to_csv(site_csv, index=False)

    logger.info("Det tractography CSVs: %s  %s", subj_csv.name, site_csv.name)
    return subj_csv, site_csv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_lengths(streamlines) -> np.ndarray:
    lengths = np.zeros(len(streamlines))
    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl)
        if len(pts) >= 2:
            lengths[i] = float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))))
    return lengths


def _load_record_from_report(r: dict) -> DetRecord:
    return DetRecord(
        site=r.get("site", ""), dataset=r.get("dataset", ""),
        subject_id=r.get("subject_id", ""),
        fa_threshold=r.get("fa_threshold", 0.10),
        interface_fa_low=r.get("interface_fa_low", 0.08),
        interface_fa_high=r.get("interface_fa_high", 0.20),
        seed_count=r.get("seed_count", 0),
        streamline_count=r.get("streamline_count", 0),
        mean_length_mm=r.get("mean_length_mm"),
        median_length_mm=r.get("median_length_mm"),
        min_length_mm=r.get("min_length_mm"),
        max_length_mm=r.get("max_length_mm"),
        output_file_size_mb=r.get("output_file_size_mb"),
        status=r.get("status", "success"),
        warning_message=r.get("warning_message"),
        timestamp=r.get("timestamp", ""),
    )
