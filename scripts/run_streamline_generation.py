"""
Phase 3.2 runner — Multi-site Streamline Tractography MVP

Usage:
    python scripts/run_streamline_generation.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --output-root    data/processed \\
        --config         configs/phase3_2_streamline_generation.yaml

    # Override seeds and FA threshold:
    python scripts/run_streamline_generation.py \\
        --processed-root data/processed/abide_ii \\
        --raw-root       data/raw/abide_ii \\
        --sites          BNI NYU_1 \\
        --seeds          2000 \\
        --fa-threshold   0.18

Logs written to: logs/phase3_2_streamline_generation.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from neurofiber.tractography.streamline_generation import (
    CLEAN_DTI_SITES_DISPLAY,
    load_config,
    run_streamline_batch,
    save_summary_csvs,
    site_to_folder,
)
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)


def _setup_file_logger(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase3_2_streamline_generation.log"
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.2 — Multi-site streamline tractography MVP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-root", type=Path, default=Path("data/processed/abide_ii"))
    p.add_argument("--raw-root",       type=Path, default=Path("data/raw/abide_ii"))
    p.add_argument("--output-root",    type=Path, default=Path("data/processed"))
    p.add_argument("--log-dir",        type=Path, default=Path("logs"))
    p.add_argument(
        "--config", type=Path,
        default=Path("configs/phase3_2_streamline_generation.yaml"),
        help="YAML config file (default: configs/phase3_2_streamline_generation.yaml).",
    )
    p.add_argument(
        "--sites", nargs="*", metavar="SITE",
        help="Override sites from config. E.g. --sites BNI NYU_1",
    )
    p.add_argument("--fa-threshold",   type=float, default=None)
    p.add_argument("--seeds",          type=int,   default=None,
                   dest="seeds_per_subject")
    p.add_argument("--step-size",      type=float, default=None)
    p.add_argument("--random-seed",    type=int,   default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_file_logger(args.log_dir)

    # Load config
    cfg: dict = {}
    if args.config.exists():
        cfg = load_config(args.config)
        logger.info("Config loaded from %s", args.config)
    else:
        logger.warning("Config not found at %s — using defaults", args.config)

    # Merge: CLI overrides config
    sites            = args.sites or cfg.get("included_sites", CLEAN_DTI_SITES_DISPLAY)
    fa_threshold     = args.fa_threshold     or cfg.get("fa_threshold", 0.15)
    seeds_per_subject= args.seeds_per_subject or cfg.get("seeds_per_subject", 5000)
    step_size        = args.step_size        or cfg.get("step_size", 0.5)
    max_angle        = float(cfg.get("max_angle", 30.0))
    max_cross        = int(cfg.get("max_cross", 1))
    random_seed      = args.random_seed      or cfg.get("random_seed", 42)
    qc               = cfg.get("qc", {})
    qc_min_sl        = int(qc.get("min_streamline_count", 500))
    qc_min_len       = float(qc.get("min_mean_length_mm", 20.0))
    qc_max_mb        = float(qc.get("max_file_size_mb", 500.0))

    logger.info("=" * 62)
    logger.info("Phase 3.2 — Streamline Tractography")
    logger.info("  processed_root   : %s", args.processed_root)
    logger.info("  sites            : %s", sites)
    logger.info("  fa_threshold     : %.2f", fa_threshold)
    logger.info("  seeds_per_subject: %d", seeds_per_subject)
    logger.info("  step_size        : %.2f mm", step_size)
    logger.info("  max_angle        : %.1f°", max_angle)
    logger.info("  random_seed      : %d", random_seed)
    logger.info("=" * 62)

    records = run_streamline_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        sites=sites,
        fa_threshold=fa_threshold,
        seeds_per_subject=seeds_per_subject,
        step_size=step_size,
        max_angle=max_angle,
        max_cross=max_cross,
        random_seed=random_seed,
        qc_min_streamlines=qc_min_sl,
        qc_min_mean_length=qc_min_len,
        qc_max_file_mb=qc_max_mb,
    )

    run_csv, site_csv = save_summary_csvs(records, args.output_root)

    # --- terminal summary ---
    n_total   = len(records)
    n_success = sum(1 for r in records if r.status == "success")
    n_warned  = sum(1 for r in records if r.status == "success" and r.warning_message)
    n_failed  = sum(1 for r in records if r.status == "failed")

    print("\n" + "=" * 66)
    print("  NEUROFIBER PHASE 3.2 — STREAMLINE TRACTOGRAPHY SUMMARY")
    print("=" * 66)
    print(f"  Total subjects  : {n_total}")
    print(f"  Success         : {n_success}  ({n_warned} with warnings)")
    print(f"  Failed          : {n_failed}")
    print("=" * 66)
    print(f"  {'Site':<10}  {'Done':>5}  {'Fail':>5}  {'Mean SL':>9}  {'Mean Len':>10}")
    print("  " + "-" * 52)

    site_stats: dict[str, dict] = defaultdict(
        lambda: {"ok": 0, "fail": 0, "sl": [], "len": []}
    )
    for r in records:
        key = "ok" if r.status == "success" else "fail"
        site_stats[r.site][key] += 1
        if r.status == "success":
            site_stats[r.site]["sl"].append(r.streamline_count)
            if r.mean_streamline_length:
                site_stats[r.site]["len"].append(r.mean_streamline_length)

    import numpy as np
    for site in sorted(site_stats):
        s  = site_stats[site]
        sl = f"{np.mean(s['sl']):.0f}" if s["sl"] else "—"
        ln = f"{np.mean(s['len']):.1f}mm" if s["len"] else "—"
        print(f"  {site:<10}  {s['ok']:>5}  {s['fail']:>5}  {sl:>9}  {ln:>10}")

    print("=" * 66)
    print(f"  Run summary  → {run_csv}")
    print(f"  Site summary → {site_csv}")
    print("=" * 66 + "\n")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
