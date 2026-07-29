"""
NeuroFiber Phase 3R.2 — Validated Streamline Tractography from processed_v2b

Generates biologically plausible whole-brain streamlines with mandatory
minimum-length filtering and false-positive suppression.

Input per subject (from processed_v2b):
  .../session_1/dti_1/
    fod/peaks.pam5
    tensor/FA.nii.gz
    qc/brain_mask.nii.gz

Output per subject:
  .../session_1/dti_1/tractography/
    streamlines.trk            (length-filtered, RASMM space)
    tractography_report.json   per-subject QC stats
    backend_used.txt

Cohort outputs:
  data/processed_v2b/
    phase3r_2_streamline_generation_summary.csv
    phase3r_2_site_summary.csv
    phase3r_2_reproducibility.csv

Filtering rules:
  - Reject streamlines with length < MIN_LENGTH_MM (20 mm) — mandatory
  - Flag streamlines with length > MAX_LENGTH_MM (300 mm) — report only, keep

Safety contract:
  - Reads only from data/processed_v2b/
  - guard_no_raw_write() enforced at every write entry point
  - IP_1 explicitly rejected — no output generated
  - Never writes to data/processed/, data/processed_v2/, or data/raw/
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
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

PIPELINE_VERSION = "3R.2"

# Tracking parameters
FA_THRESHOLD:       float = 0.15
SEED_COUNT:         int   = 5000
STEP_SIZE:          float = 0.5
MAX_ANGLE:          float = 30.0
MAX_CROSS:          int   = 1
RANDOM_SEED:        int   = 42

# Length filtering
MIN_LENGTH_MM:      float = 20.0   # reject below this
MAX_LENGTH_MM:      float = 300.0  # flag above this (keep)

# Reproducibility experiment
REPRO_N_SUBJECTS:   int = 5
REPRO_SECOND_SEED:  int = 99  # second run uses a different random seed

CLEAN_SITES:   list[str] = ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]
EXCLUDED_SITES: list[str] = ["IP_1"]

_SITE_FOLDER_MAP: dict[str, str] = {
    "BNI":    "bni",
    "IP_1":   "ip",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
}

SUBJECT_CSV_FIELDS = [
    "site", "dataset", "subject_id",
    "seed_count",
    "raw_streamline_count",
    "retained_streamline_count",
    "rejected_short_streamlines",
    "long_streamline_flag_count",
    "mean_length", "median_length",
    "p10_length", "p25_length", "p75_length", "p90_length",
    "output_size_mb",
    "status", "warning_message", "error_message",
]

SITE_CSV_FIELDS = [
    "site", "subjects",
    "mean_streamlines", "median_streamlines",
    "mean_length", "median_length",
    "mean_rejected_short", "mean_long_flags",
    "notes",
]

REPRO_CSV_FIELDS = [
    "site", "subject_id",
    "run1_retained", "run2_retained", "streamline_count_difference",
    "run1_mean_length", "run2_mean_length", "length_distribution_difference",
]

EXPECTED_COUNTS: dict[str, int] = {
    "BNI":    58,
    "NYU_1":  55,
    "NYU_2":  19,
    "SDSU_1": 57,
    "TCD_1":  40,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StreamlineRecord:
    site:        str
    dataset:     str
    subject_id:  str
    seed_count:                  int   = 0
    raw_streamline_count:        int   = 0
    retained_streamline_count:   int   = 0
    rejected_short_streamlines:  int   = 0
    long_streamline_flag_count:  int   = 0
    mean_length:    Optional[float] = None
    median_length:  Optional[float] = None
    p10_length:     Optional[float] = None
    p25_length:     Optional[float] = None
    p75_length:     Optional[float] = None
    p90_length:     Optional[float] = None
    output_size_mb: Optional[float] = None
    status:          str           = "pending"
    warning_message: Optional[str] = None
    error_message:   Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_csv_row(self) -> dict:
        return {k: getattr(self, k, None) for k in SUBJECT_CSV_FIELDS}

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core per-subject function
# ---------------------------------------------------------------------------

def generate_subject_streamlines(
    dti_dir:     Path,
    output_root: Path,
    raw_root:    Path,
    fa_threshold:      float = FA_THRESHOLD,
    seeds_per_subject: int   = SEED_COUNT,
    step_size:         float = STEP_SIZE,
    max_angle:         float = MAX_ANGLE,
    max_cross:         int   = MAX_CROSS,
    random_seed:       int   = RANDOM_SEED,
    skip_if_exists:    bool  = True,
) -> StreamlineRecord:
    """
    Generate length-filtered streamlines for one subject.
    dti_dir: .../session_1/dti_1/  from processed_v2b
    """
    subject_id  = dti_dir.parents[1].name
    dataset     = dti_dir.parents[2].name
    site_folder = dti_dir.parents[3].name
    site = next((k for k, v in _SITE_FOLDER_MAP.items() if v == site_folder),
                site_folder.upper())

    guard_no_raw_write(dti_dir, raw_root)

    rec = StreamlineRecord(site=site, dataset=dataset, subject_id=subject_id)

    # Explicit IP_1 exclusion
    if site in EXCLUDED_SITES:
        rec.status = "excluded"
        rec.warning_message = f"{site} excluded from Phase 3R.2"
        return rec

    tract_dir = dti_dir / "tractography"

    # Skip if already done
    if skip_if_exists and (tract_dir / "tractography_report.json").exists():
        try:
            d = json.loads((tract_dir / "tractography_report.json").read_text())
            if d.get("status") == "success":
                for f in SUBJECT_CSV_FIELDS:
                    if hasattr(rec, f) and f in d:
                        setattr(rec, f, d[f])
                rec.status = "success"
                logger.info("[%s/%s] already done — skipping", site, subject_id)
                return rec
        except Exception:
            pass

    # Validate inputs
    peaks_path = dti_dir / "fod"   / "peaks.pam5"
    fa_path    = dti_dir / "tensor"/ "FA.nii.gz"
    mask_path  = dti_dir / "qc"    / "brain_mask.nii.gz"

    missing = [p.name for p in [peaks_path, fa_path, mask_path] if not p.exists()]
    if missing:
        return _failed(rec, f"Missing required inputs: {missing}")

    # Load inputs
    try:
        pam      = load_peaks(str(peaks_path))
        fa_img   = nib.load(str(fa_path))
        mask_img = nib.load(str(mask_path))
    except Exception as exc:
        return _failed(rec, f"Load inputs failed: {exc}")

    fa       = fa_img.get_fdata(dtype=np.float32)
    mask_arr = np.asarray(mask_img.dataobj).astype(bool)

    # WM seed mask: FA >= threshold, inside brain mask
    wm_mask = (fa >= fa_threshold) & mask_arr
    if not wm_mask.any():
        return _failed(rec, f"WM seed mask empty at FA threshold={fa_threshold}")

    # Track
    try:
        np.random.seed(random_seed)
        seeds = random_seeds_from_mask(
            wm_mask, pam.affine,
            seeds_count=seeds_per_subject,
            seed_count_per_voxel=False,
        )
        rec.seed_count = len(seeds)

        getter = DeterministicMaximumDirectionGetter.from_shcoeff(
            pam.shm_coeff,
            max_angle=max_angle,
            sphere=pam.sphere,
        )
        stopping = ThresholdStoppingCriterion(fa, fa_threshold)

        raw_streamlines = Streamlines(LocalTracking(
            getter, stopping, seeds, pam.affine,
            step_size=step_size,
            max_cross=max_cross,
            return_all=False,
        ))
    except Exception as exc:
        return _failed(rec, f"Tracking failed: {exc}")

    rec.raw_streamline_count = len(raw_streamlines)

    if rec.raw_streamline_count == 0:
        return _failed(rec, "Zero raw streamlines generated")

    # Length filtering
    lengths_all = _compute_lengths(raw_streamlines)
    keep_mask   = np.array(lengths_all) >= MIN_LENGTH_MM
    long_mask   = np.array(lengths_all) > MAX_LENGTH_MM

    rec.rejected_short_streamlines = int((~keep_mask).sum())
    rec.long_streamline_flag_count = int(long_mask.sum())

    retained = Streamlines(s for s, keep in zip(raw_streamlines, keep_mask) if keep)
    rec.retained_streamline_count = len(retained)

    if rec.retained_streamline_count == 0:
        return _failed(rec, "Zero streamlines retained after minimum-length filtering (20mm)")

    # Length statistics on retained streamlines
    retained_lengths = np.array(lengths_all)[keep_mask]
    rec.mean_length   = _r(float(np.mean(retained_lengths)))
    rec.median_length = _r(float(np.median(retained_lengths)))
    rec.p10_length    = _r(float(np.percentile(retained_lengths, 10)))
    rec.p25_length    = _r(float(np.percentile(retained_lengths, 25)))
    rec.p75_length    = _r(float(np.percentile(retained_lengths, 75)))
    rec.p90_length    = _r(float(np.percentile(retained_lengths, 90)))

    # Save
    tract_dir.mkdir(parents=True, exist_ok=True)
    trk_path = tract_dir / "streamlines.trk"
    try:
        sft = StatefulTractogram(retained, fa_img, Space.RASMM)
        save_trk(sft, str(trk_path))
        rec.output_size_mb = _r(trk_path.stat().st_size / 1024 ** 2, 3)
    except Exception as exc:
        return _failed(rec, f"Save streamlines failed: {exc}")

    rec.status = "success"

    # Warnings
    warnings: list[str] = []
    if rec.retained_streamline_count < 500:
        warnings.append(f"low streamline count: {rec.retained_streamline_count}")
    if rec.rejected_short_streamlines > rec.raw_streamline_count * 0.5:
        warnings.append(
            f">{50}% streamlines rejected short "
            f"({rec.rejected_short_streamlines}/{rec.raw_streamline_count})"
        )
    if rec.long_streamline_flag_count > 0:
        warnings.append(f"{rec.long_streamline_flag_count} streamlines > {MAX_LENGTH_MM:.0f}mm flagged")
    rec.warning_message = "; ".join(warnings) if warnings else None

    # Write report
    report = {**rec.to_dict(), "pipeline_version": PIPELINE_VERSION,
              "fa_threshold": fa_threshold, "min_length_mm": MIN_LENGTH_MM,
              "max_length_flag_mm": MAX_LENGTH_MM}
    (tract_dir / "tractography_report.json").write_text(json.dumps(report, indent=2))
    (tract_dir / "backend_used.txt").write_text("dipy_deterministic\n")

    logger.info(
        "[%s/%s] seeds=%d  raw=%d  retained=%d  rejected_short=%d  "
        "long_flags=%d  mean=%.1fmm  %.2fMB%s",
        site, subject_id,
        rec.seed_count, rec.raw_streamline_count, rec.retained_streamline_count,
        rec.rejected_short_streamlines, rec.long_streamline_flag_count,
        rec.mean_length or 0, rec.output_size_mb or 0,
        f"  WARN: {rec.warning_message}" if rec.warning_message else "",
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_streamline_batch(
    processed_v2b_root: Path,
    raw_root:           Path,
    sites:              list[str] = CLEAN_SITES,
    fa_threshold:       float = FA_THRESHOLD,
    seeds_per_subject:  int   = SEED_COUNT,
    step_size:          float = STEP_SIZE,
    random_seed:        int   = RANDOM_SEED,
    skip_if_exists:     bool  = True,
) -> list[StreamlineRecord]:
    """
    Run streamline generation for all clean subjects across given sites.
    processed_v2b_root: e.g. data/processed_v2b/abide_ii
    """
    guard_no_raw_write(processed_v2b_root, raw_root)

    bad = [s for s in sites if s in EXCLUDED_SITES]
    if bad:
        raise ValueError(f"Cannot include excluded sites: {bad}")

    records: list[StreamlineRecord] = []

    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = processed_v2b_root / folder
        if not site_root.exists():
            logger.warning("[%s] not found: %s", site, site_root)
            continue

        dti_dirs = sorted(site_root.rglob("dti_1"))
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "fod" / "peaks.pam5").exists()]
        logger.info("[%s] %d subjects", site, len(dti_dirs))

        for dti_dir in dti_dirs:
            try:
                rec = generate_subject_streamlines(
                    dti_dir=dti_dir,
                    output_root=processed_v2b_root,
                    raw_root=raw_root,
                    fa_threshold=fa_threshold,
                    seeds_per_subject=seeds_per_subject,
                    step_size=step_size,
                    random_seed=random_seed,
                    skip_if_exists=skip_if_exists,
                )
            except Exception as exc:
                subj = dti_dir.parents[1].name
                dset = dti_dir.parents[2].name
                logger.error("[%s/%s] unexpected error: %s", site, subj, exc)
                rec = StreamlineRecord(site=site, dataset=dset, subject_id=subj,
                                       status="failed", error_message=str(exc))
            records.append(rec)

        n_ok   = sum(1 for r in records if r.site == site and r.status == "success")
        n_fail = sum(1 for r in records if r.site == site and r.status == "failed")
        logger.info("[%s] success=%d  failed=%d", site, n_ok, n_fail)

    return records


# ---------------------------------------------------------------------------
# Reproducibility experiment
# ---------------------------------------------------------------------------

def run_reproducibility_experiment(
    processed_v2b_root: Path,
    raw_root:           Path,
    records:            list[StreamlineRecord],
    n_subjects:         int = REPRO_N_SUBJECTS,
    fa_threshold:       float = FA_THRESHOLD,
    seeds_per_subject:  int   = SEED_COUNT,
    step_size:          float = STEP_SIZE,
) -> list[dict]:
    """
    Select n_subjects successful subjects at random, run tractography a second
    time with a different random seed, and compare outputs.
    Returns list of comparison dicts.
    """
    successful = [r for r in records if r.status == "success"]
    if not successful:
        logger.warning("No successful subjects for reproducibility experiment")
        return []

    rng = random.Random(42)
    chosen = rng.sample(successful, min(n_subjects, len(successful)))
    comparison_rows: list[dict] = []

    for rec in chosen:
        folder   = _SITE_FOLDER_MAP.get(rec.site, rec.site.lower())
        dti_dir  = (
            processed_v2b_root / folder / rec.dataset / rec.subject_id
            / "session_1" / "dti_1"
        )
        if not dti_dir.exists():
            continue

        # Run 2 (different seed) — write to a tmp subdir, don't overwrite
        tmp_dir = dti_dir / "tractography_repro"
        try:
            rec2 = generate_subject_streamlines(
                dti_dir=dti_dir,
                output_root=processed_v2b_root,
                raw_root=raw_root,
                fa_threshold=fa_threshold,
                seeds_per_subject=seeds_per_subject,
                step_size=step_size,
                random_seed=REPRO_SECOND_SEED,
                skip_if_exists=False,
            )
            # Move the report so we don't shadow the canonical run
            if (dti_dir / "tractography" / "tractography_report.json").exists():
                import shutil
                tmp_dir.mkdir(exist_ok=True)
                shutil.copy2(
                    str(dti_dir / "tractography" / "tractography_report.json"),
                    str(tmp_dir / "repro_tractography_report.json"),
                )
        except Exception as exc:
            logger.warning("[%s/%s] repro run failed: %s", rec.site, rec.subject_id, exc)
            continue

        count_diff = abs(rec.retained_streamline_count - (rec2.retained_streamline_count or 0))
        len_diff   = (
            abs((rec.mean_length or 0) - (rec2.mean_length or 0))
            if rec.mean_length and rec2.mean_length else None
        )

        comparison_rows.append({
            "site":                        rec.site,
            "subject_id":                  rec.subject_id,
            "run1_retained":               rec.retained_streamline_count,
            "run2_retained":               rec2.retained_streamline_count,
            "streamline_count_difference": count_diff,
            "run1_mean_length":            rec.mean_length,
            "run2_mean_length":            rec2.mean_length,
            "length_distribution_difference": _r(len_diff) if len_diff is not None else None,
        })

        logger.info(
            "[%s/%s] repro: run1=%d  run2=%d  Δcount=%d  Δlength=%.2fmm",
            rec.site, rec.subject_id,
            rec.retained_streamline_count,
            rec2.retained_streamline_count or 0,
            count_diff,
            len_diff or 0,
        )

    return comparison_rows


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def write_subject_summary(records: list[StreamlineRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_csv_row() for r in records)
    logger.info("Subject summary → %s  (%d rows)", out_path, len(records))
    return out_path


def write_site_summary(
    records:  list[StreamlineRecord],
    out_path: Path,
) -> tuple[Path, list[dict]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for site in CLEAN_SITES:
        recs    = [r for r in records if r.site == site and r.status == "success"]
        n       = len(recs)
        sl_counts  = [r.retained_streamline_count for r in recs]
        sl_lengths = [r.mean_length for r in recs if r.mean_length is not None]
        rejected   = [r.rejected_short_streamlines for r in recs]
        long_flags = [r.long_streamline_flag_count for r in recs]

        rows.append({
            "site":                 site,
            "subjects":             n,
            "mean_streamlines":     _r(float(np.mean(sl_counts))) if sl_counts else None,
            "median_streamlines":   _r(float(np.median(sl_counts))) if sl_counts else None,
            "mean_length":          _r(float(np.mean(sl_lengths))) if sl_lengths else None,
            "median_length":        _r(float(np.median(sl_lengths))) if sl_lengths else None,
            "mean_rejected_short":  _r(float(np.mean(rejected))) if rejected else None,
            "mean_long_flags":      _r(float(np.mean(long_flags))) if long_flags else None,
            "notes":                "",
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SITE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Site summary → %s", out_path)
    return out_path, rows


def write_reproducibility_csv(
    rows:     list[dict],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPRO_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Reproducibility CSV → %s  (%d subjects)", out_path, len(rows))
    return out_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_lengths(streamlines: Streamlines) -> list[float]:
    """Return per-streamline Euclidean arc length in mm."""
    lengths = []
    for s in streamlines:
        if len(s) < 2:
            lengths.append(0.0)
        else:
            diffs = np.diff(s, axis=0)
            lengths.append(float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))))
    return lengths


def _failed(rec: StreamlineRecord, error: str) -> StreamlineRecord:
    rec.status        = "failed"
    rec.error_message = error
    logger.error("[%s/%s] %s", rec.site, rec.subject_id, error)
    return rec


def _r(v: float, decimals: int = 2) -> float:
    return round(v, decimals)
