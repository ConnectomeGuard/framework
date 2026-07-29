"""
Phase 3.3 — Streamline QC + Site-Normalized Tractography Metrics

Loads per-subject streamlines.trk, computes comprehensive tract metrics
(count, mean/median/std/IQR/percentile lengths), applies site-aware
z-score normalization, flags outliers, and generates QC plots for
cross-site comparison before Phase 3.4 connectome construction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLEAN_DTI_SITES_DISPLAY: list[str] = ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]

_DISPLAY_TO_FOLDER: dict[str, str] = {
    "BNI":       "bni",
    "ETH":       "eth",
    "EMC":       "emc",
    "GU":        "gu",
    "IP_1":      "ip",
    "IU_1":      "iu",
    "KKI_1":     "kki",
    "KUL_3":     "kul",
    "NYU_1":     "nyu1",
    "NYU_2":     "nyu2",
    "OHSU_1":    "ohsu",
    "ONRC_2":    "onrc",
    "SDSU_1":    "sdsu",
    "TCD_1":     "tcd",
    "UCD_1":     "ucd",
    "UCLA_1":    "ucla1",
    "UCLA_Long": "ucla_long",
    "UPSM_Long": "upsm_long",
    "USM_1":     "usm",
}

_OUTLIER_Z_THRESHOLD: float = 3.0

# QC warning thresholds (raw, pre-normalization)
_MIN_STREAMLINE_COUNT: int   = 500
_MIN_MEAN_LENGTH_MM:   float = 20.0


def site_to_folder(name: str) -> str:
    """Convert display name (BNI, NYU_1) or folder name to local folder name."""
    if name in _DISPLAY_TO_FOLDER:
        return _DISPLAY_TO_FOLDER[name]
    return name.lower()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StreamlineQCRecord:
    site:       str
    dataset:    str
    subject_id: str

    # Raw per-subject metrics
    streamline_count:   int
    mean_length_mm:     float
    median_length_mm:   float
    std_length_mm:      float
    p10_length_mm:      float
    p25_length_mm:      float
    p75_length_mm:      float
    p90_length_mm:      float
    min_length_mm:      float
    max_length_mm:      float
    iqr_length_mm:      float   # p75 - p25

    # Site-normalized z-scores (filled by compute_site_normalizations)
    z_streamline_count: Optional[float] = field(default=None)
    z_mean_length_mm:   Optional[float] = field(default=None)
    z_p90_length_mm:    Optional[float] = field(default=None)

    # QC flags
    qc_outlier: bool           = field(default=False)
    qc_reason:  Optional[str]  = field(default=None)

    status:        str           = field(default="success")
    error_message: Optional[str] = field(default=None)

    def to_summary_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_lengths_mm(streamlines) -> np.ndarray:
    """Compute arc-length (mm) for each streamline in a collection."""
    lengths = []
    for s in streamlines:
        if len(s) < 2:
            lengths.append(0.0)
        else:
            diffs = np.diff(s, axis=0)
            lengths.append(float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))))
    return np.array(lengths, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-subject QC loader
# ---------------------------------------------------------------------------

def load_subject_qc(dti_dir: Path) -> StreamlineQCRecord:
    """Load streamlines.trk for one subject and compute QC metrics."""
    from dipy.io.streamline import load_trk

    trk_path = dti_dir / "tractography" / "streamlines.trk"

    # Extract path hierarchy: .../site/dataset/subject_id/session_1/dti_1
    subject_id = dti_dir.parent.parent.name
    dataset    = dti_dir.parent.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name

    def _failed(msg: str) -> StreamlineQCRecord:
        return StreamlineQCRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            streamline_count=0,
            mean_length_mm=0.0, median_length_mm=0.0, std_length_mm=0.0,
            p10_length_mm=0.0, p25_length_mm=0.0,
            p75_length_mm=0.0, p90_length_mm=0.0,
            min_length_mm=0.0, max_length_mm=0.0, iqr_length_mm=0.0,
            status="failed", error_message=msg,
        )

    if not trk_path.exists():
        return _failed(f"streamlines.trk not found: {trk_path}")

    try:
        sft = load_trk(str(trk_path), "same", bbox_valid_check=False)
        streamlines = sft.streamlines
        n = len(streamlines)

        if n == 0:
            return _failed("Zero streamlines in .trk file")

        lengths = _compute_lengths_mm(streamlines)
        p25 = float(np.percentile(lengths, 25))
        p75 = float(np.percentile(lengths, 75))

        return StreamlineQCRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            streamline_count=n,
            mean_length_mm=   float(np.mean(lengths)),
            median_length_mm= float(np.median(lengths)),
            std_length_mm=    float(np.std(lengths)),
            p10_length_mm=    float(np.percentile(lengths, 10)),
            p25_length_mm=    p25,
            p75_length_mm=    p75,
            p90_length_mm=    float(np.percentile(lengths, 90)),
            min_length_mm=    float(np.min(lengths)),
            max_length_mm=    float(np.max(lengths)),
            iqr_length_mm=    p75 - p25,
        )

    except Exception as exc:  # noqa: BLE001
        return _failed(str(exc))


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_qc_batch(
    processed_root: Path,
    sites: list[str],
    raw_root: Optional[Path] = None,
) -> list[StreamlineQCRecord]:
    """
    Load streamlines.trk for all subjects across sites and compute QC metrics.
    Subject failures do not stop the batch.
    """
    records: list[StreamlineQCRecord] = []

    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = processed_root / folder

        if not site_dir.exists():
            logger.warning("[%s] site directory not found: %s", folder, site_dir)
            continue

        dti_dirs = sorted(site_dir.glob("*/*/session_1/dti_1"))
        if not dti_dirs:
            logger.warning("[%s] no dti_1 directories found under %s", folder, site_dir)
            continue

        logger.info("[%s] loading QC from %d subjects …", folder, len(dti_dirs))

        for dti_dir in dti_dirs:
            rec = load_subject_qc(dti_dir)
            records.append(rec)
            if rec.status == "success":
                logger.info(
                    "[%s/%s] %d streamlines  mean=%.1fmm  std=%.1fmm  p90=%.1fmm",
                    folder, rec.subject_id,
                    rec.streamline_count,
                    rec.mean_length_mm,
                    rec.std_length_mm,
                    rec.p90_length_mm,
                )
            else:
                logger.warning(
                    "[%s/%s] FAILED: %s", folder, rec.subject_id, rec.error_message
                )

    return records


# ---------------------------------------------------------------------------
# Site-aware z-score normalization
# ---------------------------------------------------------------------------

def compute_site_normalizations(
    records: list[StreamlineQCRecord],
    outlier_z_threshold: float = _OUTLIER_Z_THRESHOLD,
) -> list[StreamlineQCRecord]:
    """
    Add site-aware z-score columns to each record:
        z_site = (x - mean_site) / std_site

    Only successful records contribute to site statistics.
    Subjects with |z| > outlier_z_threshold on count or mean length
    are flagged as qc_outlier=True.
    """
    by_site: dict[str, list[StreamlineQCRecord]] = defaultdict(list)
    for r in records:
        if r.status == "success":
            by_site[r.site].append(r)

    for site_name, site_recs in by_site.items():
        counts = np.array([r.streamline_count for r in site_recs], dtype=float)
        means  = np.array([r.mean_length_mm   for r in site_recs], dtype=float)
        p90s   = np.array([r.p90_length_mm    for r in site_recs], dtype=float)

        mu_c, sig_c = float(np.mean(counts)), float(np.std(counts))
        mu_l, sig_l = float(np.mean(means)),  float(np.std(means))
        mu_p, sig_p = float(np.mean(p90s)),   float(np.std(p90s))

        for r in site_recs:
            r.z_streamline_count = (r.streamline_count - mu_c) / (sig_c + 1e-9)
            r.z_mean_length_mm   = (r.mean_length_mm   - mu_l) / (sig_l + 1e-9)
            r.z_p90_length_mm    = (r.p90_length_mm    - mu_p) / (sig_p + 1e-9)

            reasons = []
            if abs(r.z_streamline_count) > outlier_z_threshold:
                reasons.append(f"z_count={r.z_streamline_count:.2f}")
            if abs(r.z_mean_length_mm) > outlier_z_threshold:
                reasons.append(f"z_mean_len={r.z_mean_length_mm:.2f}")
            if reasons:
                r.qc_outlier = True
                r.qc_reason  = "; ".join(reasons)

    return records


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def generate_qc_plots(
    records: list[StreamlineQCRecord],
    output_dir: Path,
    raw_root: Optional[Path] = None,
) -> list[Path]:
    """
    Generate four QC plots:
      1. qc_site_comparison.png        — boxplot: count + mean_len by site
      2. qc_length_distribution.png    — KDE of mean length per site
      3. qc_zscore_scatter.png         — z_count vs z_mean_len per subject
      4. qc_length_percentiles.png     — p10/p50/p90 ribbon by site

    Returns list of written paths.
    """
    if raw_root is not None:
        guard_no_raw_write(output_dir, raw_root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    success = [r for r in records if r.status == "success"]
    sites   = sorted({r.site for r in success})

    if not sites:
        logger.warning("No successful records to plot.")
        return []

    cmap    = plt.cm.Set2(np.linspace(0, 1, len(sites)))
    colors  = {s: tuple(cmap[i]) for i, s in enumerate(sites)}
    plots: list[Path] = []

    # ------------------------------------------------------------------
    # Plot 1: Side-by-side boxplots — streamline count & mean length
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, ylabel, title in [
        (axes[0], "streamline_count", "Count",      "Streamline Count by Site"),
        (axes[1], "mean_length_mm",   "Length (mm)", "Mean Tract Length by Site"),
    ]:
        data = [[getattr(r, metric) for r in success if r.site == s] for s in sites]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="black", linewidth=2))
        for patch, s in zip(bp["boxes"], sites):
            patch.set_facecolor(colors[s])
            patch.set_alpha(0.7)
        for i, (s, vals) in enumerate(zip(sites, data)):
            ax.scatter([i + 1] * len(vals), vals,
                       color=colors[s], s=12, alpha=0.45, zorder=3)
        ax.set_xticks(range(1, len(sites) + 1))
        ax.set_xticklabels([s.upper() for s in sites])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Phase 3.3 — Tractography QC: Site Comparison", fontsize=13)
    fig.tight_layout()
    p = output_dir / "qc_site_comparison.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(p)

    # ------------------------------------------------------------------
    # Plot 2: Length distribution KDE per site
    # ------------------------------------------------------------------
    try:
        from scipy.stats import gaussian_kde

        fig, ax = plt.subplots(figsize=(10, 5))
        for s in sites:
            vals = [r.mean_length_mm for r in success if r.site == s]
            if len(vals) < 3:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            x   = np.linspace(min(vals) - 5, max(vals) + 5, 300)
            ax.plot(x, kde(x), color=colors[s], label=s.upper(), linewidth=2)
            ax.fill_between(x, kde(x), alpha=0.12, color=colors[s])
        ax.set_xlabel("Mean Streamline Length (mm)")
        ax.set_ylabel("Density")
        ax.set_title("Mean Streamline Length Distribution by Site")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.35)
        fig.tight_layout()
        p = output_dir / "qc_length_distribution.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(p)
    except ImportError:
        logger.warning("scipy not available — skipping KDE length distribution plot")

    # ------------------------------------------------------------------
    # Plot 3: Z-score scatter (z_count vs z_mean_len)
    # ------------------------------------------------------------------
    z_recs = [r for r in success if r.z_streamline_count is not None]
    if z_recs:
        fig, ax = plt.subplots(figsize=(8, 7))
        for s in sites:
            sr = [r for r in z_recs if r.site == s]
            ax.scatter(
                [r.z_streamline_count for r in sr],
                [r.z_mean_length_mm   for r in sr],
                color=colors[s], label=s.upper(), s=40, alpha=0.75,
            )
            for r in sr:
                if r.qc_outlier:
                    ax.annotate(r.subject_id,
                                xy=(r.z_streamline_count, r.z_mean_length_mm),
                                fontsize=7, color="red")
        for thresh in [-_OUTLIER_Z_THRESHOLD, _OUTLIER_Z_THRESHOLD]:
            ax.axvline(thresh, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axhline(thresh, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Z-score: Streamline Count (within site)")
        ax.set_ylabel("Z-score: Mean Length mm (within site)")
        ax.set_title("Per-Subject Site-Normalized Z-scores")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.35)
        fig.tight_layout()
        p = output_dir / "qc_zscore_scatter.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(p)

    # ------------------------------------------------------------------
    # Plot 4: p10 / median / p90 ribbon per site
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sites))
    for i, s in enumerate(sites):
        sr   = [r for r in success if r.site == s]
        p10s = [r.p10_length_mm    for r in sr]
        p50s = [r.median_length_mm for r in sr]
        p90s = [r.p90_length_mm    for r in sr]
        mu10, mu50, mu90 = np.mean(p10s), np.mean(p50s), np.mean(p90s)
        ax.bar(i, mu90 - mu10, bottom=mu10,
               color=colors[s], alpha=0.45, width=0.5, label=s.upper())
        ax.plot([i - 0.25, i + 0.25], [mu50, mu50],
                color=colors[s], linewidth=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sites])
    ax.set_ylabel("Streamline Length (mm)")
    ax.set_title("Tract Length Percentiles by Site  (bar = p10–p90, line = median)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    p = output_dir / "qc_length_percentiles.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(p)

    logger.info("QC plots written to %s  (%d files)", output_dir, len(plots))
    return plots


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------

def save_summary_csvs(
    records: list[StreamlineQCRecord],
    output_dir: Path,
    raw_root: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    """
    Write:
      phase3_3_subject_qc.csv      — one row per subject (all metrics + z-scores)
      phase3_3_site_summary_qc.csv — one row per site (aggregated statistics)
      phase3_3_outliers.csv        — flagged subjects only

    Returns (subject_csv, site_csv, outlier_csv).
    """
    if raw_root is not None:
        guard_no_raw_write(output_dir, raw_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.to_summary_row() for r in records])

    # Subject-level
    subj_csv = output_dir / "phase3_3_subject_qc.csv"
    df.to_csv(subj_csv, index=False)

    # Site-level summary (successful subjects only)
    ok = df[df["status"] == "success"]
    site_rows = []
    for site_name, grp in ok.groupby("site"):
        site_rows.append({
            "site":                   site_name,
            "n_subjects":             len(grp),
            "n_outliers":             int(grp["qc_outlier"].sum()),
            "mean_streamline_count":  round(grp["streamline_count"].mean(), 1),
            "std_streamline_count":   round(grp["streamline_count"].std(),  1),
            "mean_mean_length_mm":    round(grp["mean_length_mm"].mean(),   2),
            "std_mean_length_mm":     round(grp["mean_length_mm"].std(),    2),
            "mean_p10_length_mm":     round(grp["p10_length_mm"].mean(),    2),
            "mean_median_length_mm":  round(grp["median_length_mm"].mean(), 2),
            "mean_p90_length_mm":     round(grp["p90_length_mm"].mean(),    2),
            "mean_iqr_length_mm":     round(grp["iqr_length_mm"].mean(),    2),
            "mean_std_length_mm":     round(grp["std_length_mm"].mean(),    2),
        })
    site_df  = pd.DataFrame(site_rows)
    site_csv = output_dir / "phase3_3_site_summary_qc.csv"
    site_df.to_csv(site_csv, index=False)

    # Outliers only
    outlier_df  = df[df["qc_outlier"]]
    outlier_csv = output_dir / "phase3_3_outliers.csv"
    outlier_df.to_csv(outlier_csv, index=False)

    logger.info(
        "CSVs written — subject=%s (%d rows)  site=%s (%d rows)  outliers=%s (%d rows)",
        subj_csv.name,  len(df),
        site_csv.name,  len(site_df),
        outlier_csv.name, len(outlier_df),
    )

    return subj_csv, site_csv, outlier_csv
