"""
Phase 3.1 runner — Multi-site FOD / Orientation Preparation

Usage:
    python scripts/run_fod_preparation.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --output-root    data/processed \\
        --sites bni nyu1 nyu2 sdsu tcd

    # Single site, custom FA range:
    python scripts/run_fod_preparation.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --sites          bni \\
        --fa-min 0.18 --fa-max 0.38

Logs are written to:
    logs/phase3_1_fod_preparation.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from neurofiber.tractography.fod_preparation import (
    CLEAN_DTI_SITES,
    FA_MAX_CLEAN,
    FA_MIN_CLEAN,
    run_fod_preparation_batch,
    save_summary_csvs,
)
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase3_1_fod_preparation.log"
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(fh)
    root.setLevel(logging.INFO)
    logger.info("Log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.1 — Multi-site FOD / orientation preparation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/abide_ii"),
        help="Root of processed ABIDE II data (default: data/processed/abide_ii).",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/abide_ii"),
        help="Root of raw data tree — safety guard (default: data/raw/abide_ii).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Root for summary CSVs (default: data/processed).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for run log file (default: logs/).",
    )
    p.add_argument(
        "--sites",
        nargs="*",
        metavar="SITE",
        default=None,
        help=(
            f"Site folder names to process (default: {CLEAN_DTI_SITES}). "
            "Example: --sites bni nyu1"
        ),
    )
    p.add_argument(
        "--fa-min",
        type=float,
        default=FA_MIN_CLEAN,
        help=f"Minimum acceptable FA mean (default: {FA_MIN_CLEAN}).",
    )
    p.add_argument(
        "--fa-max",
        type=float,
        default=FA_MAX_CLEAN,
        help=f"Maximum acceptable FA mean (default: {FA_MAX_CLEAN}).",
    )
    p.add_argument(
        "--use-mrtrix",
        action="store_true",
        default=False,
        help="Force MRtrix3 backend (requires mrconvert/dwi2response/dwi2fod on PATH).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_file_logger(args.log_dir)

    sites = args.sites or CLEAN_DTI_SITES
    use_mrtrix: bool | None = True if args.use_mrtrix else None

    logger.info("=" * 60)
    logger.info("Phase 3.1 — FOD Preparation")
    logger.info("  processed_root : %s", args.processed_root)
    logger.info("  raw_root       : %s", args.raw_root)
    logger.info("  sites          : %s", sites)
    logger.info("  FA range       : [%.2f, %.2f]", args.fa_min, args.fa_max)
    logger.info("  MRtrix3 forced : %s", args.use_mrtrix)
    logger.info("=" * 60)

    records = run_fod_preparation_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        sites=sites,
        fa_min=args.fa_min,
        fa_max=args.fa_max,
        use_mrtrix=use_mrtrix,
    )

    run_csv, site_csv = save_summary_csvs(records, args.output_root)

    # --- final summary ---
    n_total   = len(records)
    n_success = sum(1 for r in records if r.status == "success")
    n_skipped = sum(1 for r in records if r.status == "skipped")
    n_failed  = sum(1 for r in records if r.status == "failed")

    print("\n" + "=" * 62)
    print("  NEUROFIBER PHASE 3.1 — FOD PREPARATION SUMMARY")
    print("=" * 62)
    print(f"  Total subjects  : {n_total}")
    print(f"  Success         : {n_success}")
    print(f"  Skipped (FA)    : {n_skipped}")
    print(f"  Failed          : {n_failed}")
    print("=" * 62)
    print(f"  {'Site':<12}  {'Processed':>9}  {'Skipped':>7}  {'Failed':>6}  {'Mean FA':>8}")
    print("  " + "-" * 50)

    from collections import defaultdict
    site_stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "skip": 0, "fail": 0, "fa": []})
    for r in records:
        site_stats[r.site]["ok"   if r.status == "success" else
                           "skip" if r.status == "skipped" else "fail"] += 1
        if r.status == "success" and r.fa_mean > 0:
            site_stats[r.site]["fa"].append(r.fa_mean)

    for site in sorted(site_stats):
        s = site_stats[site]
        fa_str = f"{sum(s['fa'])/len(s['fa']):.4f}" if s["fa"] else "—"
        print(f"  {site:<12}  {s['ok']:>9}  {s['skip']:>7}  {s['fail']:>6}  {fa_str:>8}")

    print("=" * 62)
    print(f"  Run summary  → {run_csv}")
    print(f"  Site summary → {site_csv}")
    print("=" * 62 + "\n")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
