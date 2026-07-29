"""
NeuroFiber Phase 3R.5 — Connectome Harmonization Strategy Selection Runner

Applies four harmonization strategies to edge-feature tables and evaluates
each for residual site effect and variance retention.

Usage
─────
Dry-run:
    python scripts/run_connectome_harmonization.py --dry-run

Full run:
    python scripts/run_connectome_harmonization.py

Outputs
───────
    data/processed_v2b/harmonized_connectomes/
      none/              count_edges.csv  fa_edges.csv  md_edges.csv  length_edges.csv
      site_zscore/       ...
      global_zscore/     ...
      residualized/      ...
      combat_optional/   ... (skipped if neuroCombat not installed)
      harmonization_strategy_comparison.csv

Notes
─────
    The `experimental_site_zscore` tables in connectome_features/ are NOT the same
    as the full site_zscore harmonization here — they are subject-level z-scores
    generated in Phase 3R.4. This phase uses population-level per-edge z-scoring.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.harmonization.connectome_harmonization import (
    COMPARISON_FIELDS,
    FEATURE_TYPES,
    STRATEGIES,
    run_harmonization_pipeline,
    write_comparison_csv,
)

log = logging.getLogger("harmonization_v2b")

FEAT_DIR  = REPO_ROOT / "data" / "processed_v2b" / "connectome_features"
OUT_ROOT  = REPO_ROOT / "data" / "processed_v2b" / "harmonized_connectomes"
RAW_ROOT  = REPO_ROOT / "data" / "raw"

_FORBIDDEN = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "processed_v2",
    FEAT_DIR,   # never overwrite original features
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3R.5 — Connectome harmonization strategy selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _safety_check() -> None:
    for forbidden in _FORBIDDEN:
        if not forbidden.exists():
            continue
        assert str(OUT_ROOT.resolve()) != str(forbidden.resolve()), (
            f"SAFETY: output must not point to {forbidden}"
        )


def _dry_run() -> None:
    log.info("=" * 64)
    log.info("DRY RUN — Phase 3R.5 Connectome Harmonization")
    log.info("Feature dir  : %s", FEAT_DIR)
    log.info("Output dir   : %s", OUT_ROOT)
    log.info("Strategies   : %s", STRATEGIES)
    log.info("Feature types: %s", FEATURE_TYPES)
    log.info("=" * 64)
    for ft in FEATURE_TYPES:
        p = FEAT_DIR / f"{ft}.csv"
        log.info("  [%s]  %s", ft, "found" if p.exists() else "MISSING")
    log.info("Total jobs: %d strategies × %d feature types = %d",
             len(STRATEGIES), len(FEATURE_TYPES), len(STRATEGIES) * len(FEATURE_TYPES))


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    if args.dry_run:
        _dry_run()
        return

    log.info("=" * 64)
    log.info("Phase 3R.5 — Connectome Harmonization Strategy Selection")
    log.info("Feature dir : %s", FEAT_DIR)
    log.info("Output dir  : %s", OUT_ROOT)
    log.info("Strategies  : %s", STRATEGIES)
    log.info("=" * 64)

    comparison_rows, recommended = run_harmonization_pipeline(
        feat_dir=FEAT_DIR,
        out_root=OUT_ROOT,
        raw_root=RAW_ROOT,
    )

    comp_csv = write_comparison_csv(
        comparison_rows,
        OUT_ROOT / "harmonization_strategy_comparison.csv",
    )

    log.info("=" * 64)
    log.info("Outputs written to: %s", OUT_ROOT.relative_to(REPO_ROOT))
    log.info("Strategy comparison: %s", comp_csv.relative_to(REPO_ROOT))

    # Summary table
    log.info("-" * 64)
    log.info("Strategy comparison (count_edges):")
    log.info("  %-20s  %-12s  %-12s  %-12s  %s",
             "strategy", "site_p", "var_ratio", "var_retained", "status")
    for row in comparison_rows:
        if row["feature_type"] == "count_edges":
            log.info("  %-20s  %-12.4f  %-12.6f  %-12.6f  %s",
                     row["strategy"],
                     float(row["site_effect_p_density"] or 0),
                     float(row["site_variance_ratio"] or 0),
                     float(row["variance_retained"] or 0),
                     row["status"])

    log.info("=" * 64)
    log.info("RECOMMENDATION: %s", recommended.upper())
    log.info("Use '%s' as the harmonized input for Phase 3R.6.", recommended)

    # Gate
    success_strategies = {r["strategy"] for r in comparison_rows if r["status"] == "success"}
    if not success_strategies:
        log.error("No strategy succeeded — cannot proceed to Phase 3R.6")
        sys.exit(1)

    log.info("Gate: PASSED — %d/%d strategies succeeded",
             len(success_strategies), len(STRATEGIES))
    log.info("Recommended strategy for Phase 3R.6: %s", recommended)
    log.info("WARNING: Do NOT perform ASD classification yet.")
    log.info("Next step: Phase 3R.6 — Reference feature preparation.")


if __name__ == "__main__":
    main()
