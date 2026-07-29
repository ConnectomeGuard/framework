"""
Phase 3.4 runner — Conventional Atlas-Based Connectome Construction

Usage:
    python scripts/run_connectome_construction.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --output-root    data/processed \\
        --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1

    # Resume interrupted run (skip already processed subjects):
    python scripts/run_connectome_construction.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --skip-existing

Logs written to: logs/phase3_4_connectome_construction.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurofiber.connectome.atlas_registration import fetch_atlas
from neurofiber.connectome.connectome_qc import run_connectome_batch, save_summary_csvs
from neurofiber.tractography.streamline_generation import CLEAN_DTI_SITES_DISPLAY
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase3_4_connectome_construction.log"
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.4 — Conventional atlas-based connectome construction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-root", type=Path,
                   default=Path("data/processed/abide_ii"))
    p.add_argument("--raw-root",       type=Path,
                   default=Path("data/raw/abide_ii"))
    p.add_argument("--output-root",    type=Path,
                   default=Path("data/processed"))
    p.add_argument("--log-dir",        type=Path,
                   default=Path("logs"))
    p.add_argument("--sites", nargs="*", metavar="SITE",
                   default=CLEAN_DTI_SITES_DISPLAY,
                   help="Sites to process (default: all clean DTI sites).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip subjects that already have connectome_report.json.")
    p.add_argument("--level-iters",   type=int, nargs=3, default=[200, 50, 10],
                   metavar=("L1", "L2", "L3"),
                   help="DIPY registration iteration counts per pyramid level.")
    p.add_argument("--factors",       type=int, nargs=3, default=[8, 4, 2],
                   metavar=("F1", "F2", "F3"),
                   help="DIPY registration downsampling factors per level.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_file_logger(args.log_dir)

    logger.info("=" * 66)
    logger.info("Phase 3.4 — Conventional Atlas-Based Connectome Construction")
    logger.info("  processed_root : %s", args.processed_root)
    logger.info("  raw_root       : %s", args.raw_root)
    logger.info("  sites          : %s", args.sites)
    logger.info("  skip_existing  : %s", args.skip_existing)
    logger.info("  level_iters    : %s", args.level_iters)
    logger.info("  factors        : %s", args.factors)
    logger.info("=" * 66)

    # Step 1: Fetch atlas (downloads if not cached)
    atlas_info = fetch_atlas()
    logger.info(
        "Atlas: %s  n_rois=%d",
        atlas_info.to_dict()["atlas_desc"], atlas_info.n_rois
    )

    # Step 2: Run batch
    records = run_connectome_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        sites=args.sites,
        atlas_info=atlas_info,
        skip_if_exists=args.skip_existing,
        registration_level_iters=args.level_iters,
        registration_factors=args.factors,
    )

    if not records:
        logger.error("No records produced. Check --processed-root and --sites.")
        return 1

    # Step 3: Save CSVs
    subj_csv, site_csv = save_summary_csvs(records, args.output_root)

    # --- Terminal summary ---
    n_total   = len(records)
    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")
    n_review  = sum(1 for r in records if r.review_required)

    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "fail": 0, "edges": [], "density": [], "fa": [], "ratio": []}
    )
    for r in records:
        stats[r.site]["n"] += 1
        if r.status != "success":
            stats[r.site]["fail"] += 1
        else:
            stats[r.site]["edges"].append(r.edge_count)
            stats[r.site]["density"].append(r.graph_density)
            if r.mean_edge_fa:    stats[r.site]["fa"].append(r.mean_edge_fa)
            stats[r.site]["ratio"].append(r.mapping_success_ratio)

    print("\n" + "=" * 78)
    print("  NEUROFIBER PHASE 3.4 — CONNECTOME CONSTRUCTION SUMMARY")
    print("=" * 78)
    print(f"  Total subjects  : {n_total}")
    print(f"  Success         : {n_success}")
    print(f"  Failed          : {n_failed}")
    print(f"  Review required : {n_review}")
    print("=" * 78)
    print(f"  {'Site':<10}  {'N':>4}  {'Fail':>5}  {'Edges':>7}  "
          f"{'Density':>9}  {'Mean FA':>9}  {'Map%':>7}")
    print("  " + "-" * 64)

    for site in sorted(stats):
        s  = stats[site]
        e  = f"{np.mean(s['edges']):.0f}"       if s["edges"]   else "—"
        d  = f"{np.mean(s['density']):.4f}"     if s["density"] else "—"
        fa = f"{np.mean(s['fa']):.3f}"          if s["fa"]      else "—"
        mp = f"{np.mean(s['ratio'])*100:.1f}%"  if s["ratio"]   else "—"
        print(f"  {site:<10}  {s['n']:>4}  {s['fail']:>5}  {e:>7}  {d:>9}  {fa:>9}  {mp:>7}")

    print("=" * 78)
    print(f"  Subject CSV  → {subj_csv}")
    print(f"  Site CSV     → {site_csv}")
    print("=" * 78 + "\n")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
