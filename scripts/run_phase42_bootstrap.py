"""
Phase 4.2 — Bootstrap Uncertainty Estimation Runner

Usage:
    python run_phase42_bootstrap.py --config configs/phase42_bootstrap.yaml
    python run_phase42_bootstrap.py --config configs/phase42_bootstrap.yaml --resume
    python run_phase42_bootstrap.py --config configs/phase42_bootstrap.yaml --subjects 28920 29006

Outputs per subject:
    data/processed/abide_ii/{site}/{dataset}/{subj}/session_1/bootstrap/
        run_000/ ... run_019/  ← one adjacency_streamline_count.npy + run_meta.json per run
        bootstrap_mean.npy
        bootstrap_std.npy
        bootstrap_cov.npy
        bootstrap_presence.npy
        bootstrap_summary.json

Cohort summary:
    data/processed/phase42_bootstrap_summary.csv

Gate G-BOOT-SEED (cohort):
    Spearman rho(CoV, mean_weight) MUST be negative.
    Checked automatically after all subjects complete.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tractography.bootstrap import (
    run_bootstrap_batch,
    check_gate_boot_seed,
    save_bootstrap_summary_csv,
)

log = logging.getLogger("phase42_bootstrap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.2 — Bootstrap Uncertainty")
    p.add_argument("--config",    required=True, help="Path to phase42_bootstrap.yaml")
    p.add_argument("--resume",    action="store_true", help="Skip completed subjects")
    p.add_argument("--subjects",  nargs="+",     help="Run only these subject IDs")
    p.add_argument("--n_workers", type=int,      help="Override n_workers from config")
    p.add_argument("--dry_run",   action="store_true", help="List subjects, don't run")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def _filter_by_subject_ids(
    processed_root: Path,
    sites: list[str],
    subject_ids: list[str],
) -> None:
    from neurofiber.tractography.streamline_generation import site_to_folder
    dti_dirs = []
    for site_display in sites:
        folder = site_to_folder(site_display)
        for d in sorted((processed_root / folder).glob("*/*/session_1/dti_1")):
            if d.parent.parent.name in subject_ids:
                dti_dirs.append(d)
    return dti_dirs


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPO_ROOT / "data/processed/paper1/qc/phase42_bootstrap_run.log"
            ),
        ],
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_root = REPO_ROOT / cfg["paths"]["processed_root"]
    raw_root       = REPO_ROOT / cfg["paths"]["raw_root"]
    output_root    = REPO_ROOT / cfg["paths"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data/processed/paper1/qc").mkdir(parents=True, exist_ok=True)

    sites      = cfg["sites"]
    B          = cfg["bootstrap"]["B"]
    base_seed  = cfg["bootstrap"]["base_seed"]
    n_workers  = args.n_workers or cfg["bootstrap"]["n_workers"]

    log.info("=" * 60)
    log.info("Phase 4.2 — Bootstrap  B=%d  base_seed=%d  n_workers=%d", B, base_seed, n_workers)
    log.info("Processed root: %s", processed_root)
    log.info("Sites: %s", sites)
    log.info("=" * 60)

    # Dry run: just list subjects
    if args.dry_run:
        from neurofiber.tractography.streamline_generation import site_to_folder
        total = 0
        for site_display in sites:
            folder = site_to_folder(site_display)
            dti_dirs = sorted((processed_root / folder).glob("*/*/session_1/dti_1"))
            log.info("[%s] %d subjects", folder, len(dti_dirs))
            total += len(dti_dirs)
        log.info("Total: %d subjects — DRY RUN, not executing", total)
        return

    records = run_bootstrap_batch(
        processed_root=processed_root,
        raw_root=raw_root,
        sites=sites,
        B=B,
        base_seed=base_seed,
        seeds_per_subject=cfg["tractography"]["seeds_per_subject"],
        fa_threshold=cfg["tractography"]["fa_threshold"],
        interface_fa_low=cfg["tractography"]["interface_fa_low"],
        interface_fa_high=cfg["tractography"]["interface_fa_high"],
        step_size=cfg["tractography"]["step_size"],
        max_angle=cfg["tractography"]["max_angle"],
        max_cross=cfg["tractography"]["max_cross"],
        n_workers=n_workers,
        keep_trk=cfg["bootstrap"].get("keep_trk", False),
        skip_if_exists=args.resume,
    )

    # Filter to requested subjects if specified
    if args.subjects:
        records = [r for r in records if r.subject_id in args.subjects]

    # Save cohort summary CSV
    csv_path = save_bootstrap_summary_csv(records, output_root)

    # Print cohort summary
    n_ok    = sum(1 for r in records if r.status == "success")
    n_fail  = sum(1 for r in records if r.status == "failed")
    n_warn  = sum(1 for r in records if r.warning_message)
    log.info("=" * 60)
    log.info("Bootstrap complete: %d success / %d failed / %d flagged", n_ok, n_fail, n_warn)
    if n_warn:
        log.warning("Flagged subjects (review_required):")
        for r in records:
            if r.warning_message:
                log.warning("  [%s/%s] %s", r.site, r.subject_id, r.warning_message)

    # G-BOOT-SEED cohort gate
    log.info("Running G-BOOT-SEED cohort gate ...")
    try:
        cohort_rho = check_gate_boot_seed(processed_root)
        log.info("G-BOOT-SEED PASSED: cohort rho = %.3f (negative as required)", cohort_rho)
    except AssertionError as e:
        log.error("G-BOOT-SEED FAILED: %s", e)
        log.error("HARD STOP — bootstrap implementation broken. Debug before continuing.")
        sys.exit(1)

    # Site summary
    df = pd.read_csv(csv_path)
    if "site" in df.columns and "median_cov" in df.columns:
        site_summary = (
            df[df.status == "success"]
            .groupby("site")
            .agg(n=("subject_id", "count"),
                 median_cov=("median_cov", "mean"),
                 spearman_rho=("spearman_rho", "mean"))
            .round(4)
        )
        log.info("\n" + "=" * 60)
        log.info("SITE SUMMARY\n%s", site_summary.to_string())
        log.info("=" * 60)

    log.info("Summary CSV → %s", csv_path)
    log.info("Phase 4.2 complete.")


if __name__ == "__main__":
    main()
