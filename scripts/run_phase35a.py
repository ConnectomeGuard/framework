"""
Phase 3.5A runner — Deterministic Tractography Baseline

Three-step pipeline:
  1. Interface-seeded deterministic tractography  → dti_1/tractography_det/
  2. Connectome construction (reuses Phase 3.4 atlas)  → session_1/connectome_det/
  3. Comparison analysis + report generation  → data/processed/phase35a/

Usage:
    python scripts/run_phase35a.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --output-root    data/processed/phase35a \\
        --baseline-csv   data/processed/phase3_4_connectome_summary.csv \\
        --baseline-site-csv data/processed/phase3_4_site_connectome_summary.csv

    # Resume interrupted run:
    python scripts/run_phase35a.py ... --skip-existing

Logs written to: logs/phase3_5a_deterministic.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from neurofiber.connectome.atlas_registration import fetch_atlas
from neurofiber.connectome.phase35a import (
    CONNECTOME_SUBDIR,
    build_qc_flags,
    compare_connectomes,
    compare_sites,
    compute_cohort_graph_properties,
    generate_phase35a_report,
    run_det_connectome_batch,
    save_summary_csvs,
)
from neurofiber.tractography.det_tractography import (
    run_det_batch,
    save_det_tractography_csvs,
)
from neurofiber.tractography.streamline_generation import CLEAN_DTI_SITES_DISPLAY
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase3_5a_deterministic.log"
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.5A — Deterministic tractography baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-root", type=Path, default=Path("data/processed/abide_ii"))
    p.add_argument("--raw-root",        type=Path, default=Path("data/raw/abide_ii"))
    p.add_argument("--output-root",     type=Path, default=Path("data/processed/phase35a"))
    p.add_argument("--baseline-csv",    type=Path,
                   default=Path("data/processed/phase3_4_connectome_summary.csv"))
    p.add_argument("--baseline-site-csv", type=Path,
                   default=Path("data/processed/phase3_4_site_connectome_summary.csv"))
    p.add_argument("--log-dir",         type=Path, default=Path("logs"))
    p.add_argument("--sites", nargs="*", metavar="SITE",
                   default=CLEAN_DTI_SITES_DISPLAY)
    p.add_argument("--skip-existing",   action="store_true")

    # Tractography parameters
    p.add_argument("--fa-threshold",      type=float, default=0.10)
    p.add_argument("--interface-fa-low",  type=float, default=0.08)
    p.add_argument("--interface-fa-high", type=float, default=0.20)
    p.add_argument("--seeds",             type=int,   default=10_000)
    p.add_argument("--step-size",         type=float, default=0.5)
    p.add_argument("--max-angle",         type=float, default=30.0)

    # Atlas registration (only used if Phase 3.4 atlas is absent)
    p.add_argument("--level-iters", type=int, nargs=3, default=[200, 50, 10],
                   metavar=("L1", "L2", "L3"))
    p.add_argument("--factors",     type=int, nargs=3, default=[8, 4, 2],
                   metavar=("F1", "F2", "F3"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_file_logger(args.log_dir)

    logger.info("=" * 66)
    logger.info("Phase 3.5A — Deterministic Tractography Baseline")
    logger.info("  processed_root   : %s", args.processed_root)
    logger.info("  output_root      : %s", args.output_root)
    logger.info("  sites            : %s", args.sites)
    logger.info("  skip_existing    : %s", args.skip_existing)
    logger.info("  interface FA     : [%.2f, %.2f)", args.interface_fa_low, args.interface_fa_high)
    logger.info("  seeds/subject    : %d", args.seeds)
    logger.info("=" * 66)

    args.output_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Interface-seeded deterministic tractography
    # -----------------------------------------------------------------------
    logger.info("STEP 1 — Deterministic tractography (interface seeding)")
    tract_records = run_det_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        sites=args.sites,
        fa_threshold=args.fa_threshold,
        interface_fa_low=args.interface_fa_low,
        interface_fa_high=args.interface_fa_high,
        seeds_per_subject=args.seeds,
        step_size=args.step_size,
        max_angle=args.max_angle,
        skip_if_exists=args.skip_existing,
    )

    tract_subj_csv, tract_site_csv = save_det_tractography_csvs(
        tract_records, args.output_root
    )

    n_tract_ok   = sum(1 for r in tract_records if r.status == "success")
    n_tract_fail = len(tract_records) - n_tract_ok
    logger.info(
        "Tractography: %d success / %d failed", n_tract_ok, n_tract_fail
    )

    # -----------------------------------------------------------------------
    # Step 2: Connectome construction (reuse Phase 3.4 atlas registrations)
    # -----------------------------------------------------------------------
    logger.info("STEP 2 — Connectome construction (reusing Phase 3.4 atlas)")
    atlas_info = fetch_atlas()

    conn_records = run_det_connectome_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        sites=args.sites,
        atlas_info=atlas_info,
        skip_if_exists=args.skip_existing,
        registration_level_iters=args.level_iters,
        registration_factors=args.factors,
    )

    if not conn_records:
        logger.error("No connectome records produced — check --processed-root.")
        return 1

    conn_subj_csv, conn_site_csv = save_summary_csvs(conn_records, args.output_root)

    # Rename to required deliverable names
    det_conn_subj_csv = args.output_root / "deterministic_connectome_summary.csv"
    det_conn_site_csv = args.output_root / "deterministic_site_summary.csv"
    conn_subj_csv.rename(det_conn_subj_csv)
    conn_site_csv.rename(det_conn_site_csv)

    n_conn_ok   = sum(1 for r in conn_records if r.status == "success")
    n_conn_fail = len(conn_records) - n_conn_ok
    logger.info(
        "Connectome: %d success / %d failed", n_conn_ok, n_conn_fail
    )

    # -----------------------------------------------------------------------
    # Step 3: Comparison analysis + report
    # -----------------------------------------------------------------------
    logger.info("STEP 3 — Comparison analysis and report generation")

    # Per-subject comparison
    comparison_df = pd.DataFrame()
    if args.baseline_csv.exists():
        try:
            comparison_df = compare_connectomes(args.baseline_csv, det_conn_subj_csv)
            cmp_csv = args.output_root / "deterministic_vs_baseline_comparison.csv"
            comparison_df.to_csv(cmp_csv, index=False)
            logger.info("Comparison CSV → %s", cmp_csv.name)
        except Exception as exc:
            logger.warning("Comparison failed: %s", exc)
    else:
        logger.warning("Baseline CSV not found (%s) — skipping comparison", args.baseline_csv)

    # QC flags
    det_df  = pd.read_csv(det_conn_subj_csv)
    base_df = pd.read_csv(args.baseline_csv) if args.baseline_csv.exists() else pd.DataFrame()
    qc_df   = build_qc_flags(det_df, base_df)
    qc_csv  = args.output_root / "deterministic_qc_flags.csv"
    qc_df.to_csv(qc_csv, index=False)
    logger.info("QC flags CSV → %s  (%d flagged)", qc_csv.name, len(qc_df))

    # Graph properties
    graph_props = compute_cohort_graph_properties(
        args.processed_root, args.sites, atlas_info.label_df,
        connectome_subdir=CONNECTOME_SUBDIR,
    )

    # Report
    generate_phase35a_report(
        det_site_csv=det_conn_site_csv,
        base_site_csv=args.baseline_site_csv if args.baseline_site_csv.exists()
                      else det_conn_site_csv,
        comparison_df=comparison_df,
        graph_props=graph_props,
        output_path=args.output_root / "PHASE_3_5A_REPORT.md",
        n_total=len(conn_records),
        n_success=n_conn_ok,
        n_failed=n_conn_fail,
        n_review=int(det_df.get("review_required", pd.Series(dtype=bool)).sum()),
    )

    # Terminal summary
    _print_summary(conn_records, args.output_root)

    return 0 if n_conn_fail == 0 else 1


def _print_summary(records, output_root: Path) -> None:
    import pandas as _pd
    from collections import defaultdict

    n_total   = len(records)
    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = n_total - n_success

    stats: dict = defaultdict(lambda: {"n": 0, "fail": 0, "edges": [], "density": [], "fa": [], "ratio": []})
    for r in records:
        stats[r.site]["n"] += 1
        if r.status != "success":
            stats[r.site]["fail"] += 1
        else:
            stats[r.site]["edges"].append(r.edge_count)
            stats[r.site]["density"].append(r.graph_density)
            if r.mean_edge_fa:
                stats[r.site]["fa"].append(r.mean_edge_fa)
            stats[r.site]["ratio"].append(r.mapping_success_ratio)

    print("\n" + "=" * 78)
    print("  NEUROFIBER PHASE 3.5A — DETERMINISTIC CONNECTOME SUMMARY")
    print("=" * 78)
    print(f"  Total subjects  : {n_total}")
    print(f"  Success         : {n_success}")
    print(f"  Failed          : {n_failed}")
    print("=" * 78)
    print(f"  {'Site':<10}  {'N':>4}  {'Fail':>5}  {'Edges':>7}  {'Density':>9}  {'Mean FA':>9}  {'Map%':>7}")
    print("  " + "-" * 60)
    for site in sorted(stats):
        s  = stats[site]
        e  = f"{np.mean(s['edges']):.0f}"      if s["edges"]   else "—"
        d  = f"{np.mean(s['density']):.4f}"    if s["density"] else "—"
        fa = f"{np.mean(s['fa']):.3f}"         if s["fa"]      else "—"
        mp = f"{np.mean(s['ratio'])*100:.1f}%" if s["ratio"]   else "—"
        print(f"  {site:<10}  {s['n']:>4}  {s['fail']:>5}  {e:>7}  {d:>9}  {fa:>9}  {mp:>7}")
    print("=" * 78)
    print(f"  Output dir → {output_root}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    sys.exit(main())
