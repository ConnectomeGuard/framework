"""
NeuroFiber Phase 3R.4 — Connectome QC and Harmonization Preparation Runner

Computes graph metrics, edge feature tables, site-effect analysis,
and harmonization metadata for 229 clean subjects.

Usage
─────
Dry-run:
    python scripts/run_connectome_qc_harmonization.py --dry-run

Full run:
    python scripts/run_connectome_qc_harmonization.py

Outputs
───────
    data/processed_v2b/connectome_features/
        subject_graph_metrics.csv
        count_edges.csv  length_edges.csv  fa_edges.csv  md_edges.csv
        harmonization_metadata.csv
        experimental_site_zscore/
            count_edges_zscore.csv  ...  (EXPERIMENTAL — not for direct modeling)

    data/processed_v2b/
        phase3r_4_connectome_qc_summary.csv
        phase3r_4_site_effect_summary.csv
        phase3r_4_statistical_tests.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.connectome.connectome_qc_harmonization import (
    CLEAN_SITES,
    EXCLUDED_SITES,
    EXPECTED_COUNTS,
    _SITE_FOLDER_MAP,
    run_qc_harmonization,
    write_all_outputs,
)

log = logging.getLogger("connectome_qc_v2b")

PROCESSED_V2B_ROOT = REPO_ROOT / "data" / "processed_v2b" / "abide_ii"
RAW_ROOT           = REPO_ROOT / "data" / "raw"
OUT_ROOT           = REPO_ROOT / "data" / "processed_v2b"

_FORBIDDEN = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "processed_v2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3R.4 — Connectome QC and harmonization preparation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=CLEAN_SITES,
                   choices=CLEAN_SITES, metavar="SITE")
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
        assert str(PROCESSED_V2B_ROOT.resolve()) != str(forbidden.resolve()), (
            f"SAFETY: output root must not point to {forbidden}"
        )


def _dry_run(sites: list[str]) -> None:
    log.info("=" * 64)
    log.info("DRY RUN — Phase 3R.4 Connectome QC and Harmonization Prep")
    log.info("Excluded: %s", EXCLUDED_SITES)
    log.info("=" * 64)
    total = 0
    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = PROCESSED_V2B_ROOT / folder
        if not site_root.exists():
            log.warning("[%s] not found", site)
            continue
        conn_dirs = sorted(site_root.rglob("connectome"))
        conn_dirs = [d for d in conn_dirs if d.is_dir() and
                     (d / "count_matrix.npy").exists()]
        log.info("[%s]  %d connectomes  expected=%d",
                 site, len(conn_dirs), EXPECTED_COUNTS.get(site, "?"))
        total += len(conn_dirs)
    log.info("=" * 64)
    log.info("Total: %d subjects", total)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    sites = [s for s in args.sites if s not in EXCLUDED_SITES]

    if args.dry_run:
        _dry_run(sites)
        return

    log.info("=" * 64)
    log.info("Phase 3R.4 — Connectome QC and Harmonization Preparation")
    log.info("Sites    : %s", sites)
    log.info("Excluded : %s", EXCLUDED_SITES)
    log.info("=" * 64)

    results = run_qc_harmonization(
        processed_v2b_root=PROCESSED_V2B_ROOT,
        raw_root=RAW_ROOT,
        sites=sites,
    )

    paths = write_all_outputs(results, OUT_ROOT, RAW_ROOT)

    metrics   = results["metrics"]
    n_success = sum(1 for m in metrics if m.status == "success")
    n_failed  = sum(1 for m in metrics if m.status == "failed")

    log.info("=" * 64)
    log.info("Done: %d success / %d failed", n_success, n_failed)
    for name, p in paths.items():
        log.info("  %-30s → %s", name, p.relative_to(REPO_ROOT))

    # Site effect summary
    log.info("-" * 64)
    log.info("Site effects (density — Kruskal-Wallis):")
    for row in results["stat_tests"]:
        if row["metric"] == "density":
            log.info("  density: H=%.3f  p=%.6f  → %s",
                     row["statistic"], row["p_value"], row["interpretation"])

    log.info("-" * 64)
    log.info("Site density z-scores:")
    density_effects = [r for r in results["site_effects"] if r["metric"] == "density"]
    for row in density_effects:
        flag = "  *** OUTLIER" if row.get("outlier_flag") else ""
        log.info("  [%s]  mean=%.4f  z=%.2f%s",
                 row["site"], row["mean"], row["site_z"], flag)

    # QC flag summary
    flagged = [m for m in metrics if m.qc_flag and m.status == "success"]
    if flagged:
        log.warning("-" * 64)
        log.warning("%d subjects flagged:", len(flagged))
        for m in flagged[:10]:
            log.warning("  [%s/%s] %s", m.site, m.subject_id, m.qc_flag)
        if len(flagged) > 10:
            log.warning("  ... and %d more", len(flagged) - 10)
    else:
        log.info("No subjects with QC flags")

    log.info("=" * 64)
    log.info("GATE CHECK — Phase 3R.4")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        n    = sum(1 for m in metrics if m.site == site and m.status == "success")
        ok   = n == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, n, expected)
        if not ok:
            all_pass = False

    if all_pass:
        log.info("Gate: PASSED — feature tables ready for Phase 3R.5 (harmonization)")
        log.info("NOTE: experimental z-score tables in connectome_features/experimental_site_zscore/")
        log.info("      These are NOT ready for modeling — review site effects first.")
    else:
        log.warning("Gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
