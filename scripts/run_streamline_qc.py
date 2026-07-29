"""
Phase 3.3 runner — Streamline QC + Site-Normalized Tractography Metrics

Usage:
    python scripts/run_streamline_qc.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --output-root    data/processed \\
        --plot-dir       data/processed/qc_plots/phase3_3

    # Single site:
    python scripts/run_streamline_qc.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --sites BNI

Logs written to: logs/phase3_3_streamline_qc.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurofiber.tractography.streamline_qc import (
    CLEAN_DTI_SITES_DISPLAY,
    compute_site_normalizations,
    generate_qc_plots,
    run_qc_batch,
    save_summary_csvs,
)
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase3_3_streamline_qc.log"
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.3 — Streamline QC + site-normalized tractography metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-root", type=Path,
                   default=Path("data/processed/abide_ii"))
    p.add_argument("--raw-root",       type=Path,
                   default=Path("data/raw/abide_ii"))
    p.add_argument("--output-root",    type=Path,
                   default=Path("data/processed"))
    p.add_argument("--plot-dir",       type=Path,
                   default=Path("data/processed/qc_plots/phase3_3"))
    p.add_argument("--log-dir",        type=Path,
                   default=Path("logs"))
    p.add_argument("--sites", nargs="*", metavar="SITE",
                   default=CLEAN_DTI_SITES_DISPLAY,
                   help="Sites to process (default: all clean DTI sites).")
    p.add_argument("--outlier-z-threshold", type=float, default=3.0,
                   help="Z-score threshold for outlier flagging (default: 3.0).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_file_logger(args.log_dir)

    logger.info("=" * 62)
    logger.info("Phase 3.3 — Streamline QC + Site-Normalized Metrics")
    logger.info("  processed_root    : %s", args.processed_root)
    logger.info("  sites             : %s", args.sites)
    logger.info("  outlier_z         : %.1f", args.outlier_z_threshold)
    logger.info("=" * 62)

    # Step 1: Load per-subject metrics from .trk files
    records = run_qc_batch(
        processed_root=args.processed_root,
        sites=args.sites,
        raw_root=args.raw_root,
    )

    if not records:
        logger.error("No records loaded. Check --processed-root and --sites.")
        return 1

    # Step 2: Site-aware z-score normalization + outlier detection
    records = compute_site_normalizations(
        records, outlier_z_threshold=args.outlier_z_threshold
    )

    # Step 3: QC plots
    plots = generate_qc_plots(records, args.plot_dir, raw_root=args.raw_root)

    # Step 4: Save CSVs
    subj_csv, site_csv, outlier_csv = save_summary_csvs(
        records, args.output_root, raw_root=args.raw_root
    )

    # --- terminal summary ---
    n_total   = len(records)
    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")
    n_outlier = sum(1 for r in records if r.qc_outlier)

    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "out": 0, "sl": [], "len": [], "std_len": []}
    )
    for r in records:
        if r.status != "success":
            continue
        stats[r.site]["n"]       += 1
        stats[r.site]["out"]     += int(r.qc_outlier)
        stats[r.site]["sl"].append(r.streamline_count)
        stats[r.site]["len"].append(r.mean_length_mm)
        stats[r.site]["std_len"].append(r.std_length_mm)

    print("\n" + "=" * 70)
    print("  NEUROFIBER PHASE 3.3 — STREAMLINE QC SUMMARY")
    print("=" * 70)
    print(f"  Total subjects  : {n_total}")
    print(f"  Success         : {n_success}")
    print(f"  Failed          : {n_failed}")
    print(f"  QC outliers     : {n_outlier}")
    print("=" * 70)
    print(f"  {'Site':<10}  {'N':>4}  {'Outliers':>8}  "
          f"{'Mean SL':>9}  {'Mean Len':>10}  {'Len Std':>9}")
    print("  " + "-" * 60)

    for site in sorted(stats):
        s  = stats[site]
        sl = f"{np.mean(s['sl']):.0f}"      if s["sl"]      else "—"
        ln = f"{np.mean(s['len']):.1f}mm"   if s["len"]     else "—"
        sd = f"{np.mean(s['std_len']):.1f}mm" if s["std_len"] else "—"
        print(f"  {site:<10}  {s['n']:>4}  {s['out']:>8}  {sl:>9}  {ln:>10}  {sd:>9}")

    print("=" * 70)
    print(f"  Subject CSV  → {subj_csv}")
    print(f"  Site CSV     → {site_csv}")
    print(f"  Outliers CSV → {outlier_csv}")
    print(f"  Plots        → {args.plot_dir}/  ({len(plots)} files)")
    print("=" * 70 + "\n")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
