"""
NeuroFiber Phase 3R.2 — Validated Streamline Tractography Runner

Generates length-filtered deterministic streamlines for 229 clean DTI subjects
from the canonical processed_v2b pipeline, with a reproducibility experiment.

Usage
─────
Dry-run (count subjects, no tracking):
    python scripts/run_streamline_generation_v2b.py --dry-run

Full run:
    python scripts/run_streamline_generation_v2b.py

Resume (skip already-tracked subjects):
    python scripts/run_streamline_generation_v2b.py --resume

Restrict sites:
    python scripts/run_streamline_generation_v2b.py --sites BNI NYU_1

Skip reproducibility experiment:
    python scripts/run_streamline_generation_v2b.py --no-repro

Outputs
───────
    data/processed_v2b/abide_ii/<site>/.../<subj>/session_1/dti_1/tractography/
        streamlines.trk
        tractography_report.json
        backend_used.txt

    data/processed_v2b/
        phase3r_2_streamline_generation_summary.csv
        phase3r_2_site_summary.csv
        phase3r_2_reproducibility.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tractography.streamline_generation_v2b import (
    CLEAN_SITES,
    EXCLUDED_SITES,
    EXPECTED_COUNTS,
    _SITE_FOLDER_MAP,
    run_reproducibility_experiment,
    run_streamline_batch,
    write_reproducibility_csv,
    write_site_summary,
    write_subject_summary,
)

log = logging.getLogger("streamline_v2b")

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
        description="Phase 3R.2 — Streamline tractography from processed_v2b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=CLEAN_SITES,
                   choices=CLEAN_SITES, metavar="SITE")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects with existing successful tractography_report.json.")
    p.add_argument("--no-repro", dest="run_repro", action="store_false",
                   help="Skip reproducibility experiment.")
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
    log.info("DRY RUN — Phase 3R.2 Streamline Tractography from processed_v2b")
    log.info("Excluded: %s", EXCLUDED_SITES)
    log.info("=" * 64)
    total = 0
    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = PROCESSED_V2B_ROOT / folder
        if not site_root.exists():
            log.warning("[%s] not found", site)
            continue
        dti_dirs = sorted(site_root.rglob("dti_1"))
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "fod" / "peaks.pam5").exists()]
        done = sum(1 for d in dti_dirs
                   if (d / "tractography" / "tractography_report.json").exists())
        log.info("[%s]  %d subjects  (%d done)  expected=%d",
                 site, len(dti_dirs), done, EXPECTED_COUNTS.get(site, "?"))
        total += len(dti_dirs)
    log.info("=" * 64)
    log.info("Total clean subjects: %d  (expected 229)", total)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    sites = [s for s in args.sites if s not in EXCLUDED_SITES]

    if args.dry_run:
        _dry_run(sites)
        return

    log.info("=" * 64)
    log.info("Phase 3R.2 — Streamline Tractography from processed_v2b")
    log.info("Sites     : %s", sites)
    log.info("Excluded  : %s", EXCLUDED_SITES)
    log.info("Resume    : %s", args.resume)
    log.info("=" * 64)

    records = run_streamline_batch(
        processed_v2b_root=PROCESSED_V2B_ROOT,
        raw_root=RAW_ROOT,
        sites=sites,
        skip_if_exists=args.resume,
    )

    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")
    n_skipped = sum(1 for r in records if r.status == "skipped")

    subj_csv = write_subject_summary(
        records, OUT_ROOT / "phase3r_2_streamline_generation_summary.csv"
    )
    site_csv, site_rows = write_site_summary(
        records, OUT_ROOT / "phase3r_2_site_summary.csv"
    )

    # Reproducibility experiment
    repro_csv = None
    if args.run_repro:
        log.info("Running reproducibility experiment (5 subjects, 2 seeds)...")
        repro_rows = run_reproducibility_experiment(
            processed_v2b_root=PROCESSED_V2B_ROOT,
            raw_root=RAW_ROOT,
            records=records,
        )
        repro_csv = write_reproducibility_csv(
            repro_rows, OUT_ROOT / "phase3r_2_reproducibility.csv"
        )

    log.info("=" * 64)
    log.info("Done: %d success / %d failed / %d skipped", n_success, n_failed, n_skipped)
    log.info("Subject CSV → %s", subj_csv.relative_to(REPO_ROOT))
    log.info("Site CSV    → %s", site_csv.relative_to(REPO_ROOT))
    if repro_csv:
        log.info("Repro CSV   → %s", repro_csv.relative_to(REPO_ROOT))

    # Site summary
    log.info("-" * 64)
    log.info("Site summary:")
    for row in site_rows:
        log.info(
            "  [%s]  n=%s  mean_streamlines=%.0f  mean_length=%.1fmm  "
            "mean_rejected_short=%.0f",
            row["site"], row["subjects"],
            row["mean_streamlines"] or 0,
            row["mean_length"] or 0,
            row["mean_rejected_short"] or 0,
        )

    # Gate check
    log.info("=" * 64)
    log.info("GATE CHECK — Phase 3R.2 (expected 229/229)")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        n = sum(1 for r in records if r.site == site and r.status == "success")
        ok = n == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, n, expected)
        if not ok:
            all_pass = False

    # Check no zero-streamline subjects
    zero_sl = [r for r in records if r.status == "success" and r.retained_streamline_count == 0]
    if zero_sl:
        log.error("  ✗ %d subjects have zero retained streamlines!", len(zero_sl))
        all_pass = False
    else:
        log.info("  ✓ No subjects with zero retained streamlines")

    # Confirm IP_1 exclusion
    ip1_ok = sum(1 for r in records if r.site == "IP_1" and r.status == "success")
    if ip1_ok > 0:
        log.error("  ✗ IP_1 has %d unexpected success records", ip1_ok)
        all_pass = False
    else:
        log.info("  ✓ IP_1 correctly excluded")

    if all_pass:
        log.info("Gate: PASSED — proceed to Phase 3R.3 (connectome construction)")
    else:
        log.warning("Gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
