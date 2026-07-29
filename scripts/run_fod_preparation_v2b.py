"""
NeuroFiber Phase 3R.1 — FOD / Peak Direction Preparation Runner

Estimates DIPY CsaOdfModel peak directions for 229 clean DTI subjects
across 5 sites from the canonical processed_v2b pipeline.

IP_1 is explicitly excluded.

Usage
─────
Dry-run (show subject counts, no FOD estimation):
    python scripts/run_fod_preparation_v2b.py --dry-run

Full run:
    python scripts/run_fod_preparation_v2b.py

Resume (skip already-completed subjects):
    python scripts/run_fod_preparation_v2b.py --resume

Restrict sites:
    python scripts/run_fod_preparation_v2b.py --sites BNI NYU_1

Outputs
───────
    data/processed_v2b/abide_ii/<site>/.../<subj>/session_1/dti_1/fod/
        peaks.pam5
        fod_prep_report.json
        backend_used.txt

    data/processed_v2b/
        phase3r_1_fod_preparation_summary.csv
        phase3r_1_fod_site_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tractography.fod_preparation_v2b import (
    CLEAN_SITES_DISPLAY,
    EXCLUDED_SITES_DISPLAY,
    EXPECTED_COUNTS,
    _SITE_FOLDER_MAP,
    run_fod_preparation_batch,
    write_site_summary,
    write_subject_summary,
)

log = logging.getLogger("fod_v2b")

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
        description="Phase 3R.1 — FOD preparation from processed_v2b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=CLEAN_SITES_DISPLAY,
                   choices=CLEAN_SITES_DISPLAY, metavar="SITE",
                   help="Clean sites to process (IP_1 always excluded).")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects with an existing successful fod_prep_report.json.")
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
    log.info("DRY RUN — Phase 3R.1 FOD Preparation from processed_v2b")
    log.info("Excluded sites: %s", EXCLUDED_SITES_DISPLAY)
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
                    (d / "dwi_preprocessed.nii.gz").exists()]
        done = sum(1 for d in dti_dirs if (d / "fod" / "fod_prep_report.json").exists())
        log.info("[%s]  %d subjects  (%d already done)  expected=%d",
                 site, len(dti_dirs), done, EXPECTED_COUNTS.get(site, "?"))
        total += len(dti_dirs)
    log.info("=" * 64)
    log.info("Total clean subjects: %d", total)
    log.info("Expected: %d", sum(EXPECTED_COUNTS.values()))


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    # Enforce IP_1 exclusion even if somehow passed
    sites = [s for s in args.sites if s not in EXCLUDED_SITES_DISPLAY]
    excluded_requested = [s for s in args.sites if s in EXCLUDED_SITES_DISPLAY]
    if excluded_requested:
        log.warning("Requested excluded sites rejected: %s", excluded_requested)

    if args.dry_run:
        _dry_run(sites)
        return

    log.info("=" * 64)
    log.info("Phase 3R.1 — FOD Preparation from processed_v2b")
    log.info("Clean sites   : %s", sites)
    log.info("Excluded sites: %s (no output will be generated)", EXCLUDED_SITES_DISPLAY)
    log.info("Resume        : %s", args.resume)
    log.info("Input root    : %s", PROCESSED_V2B_ROOT)
    log.info("=" * 64)

    records = run_fod_preparation_batch(
        processed_v2b_root=PROCESSED_V2B_ROOT,
        raw_root=RAW_ROOT,
        sites=sites,
        skip_if_exists=args.resume,
    )

    n_success  = sum(1 for r in records if r.status == "success")
    n_failed   = sum(1 for r in records if r.status == "failed")
    n_skipped  = sum(1 for r in records if r.status == "skipped")
    n_excluded = sum(1 for r in records if r.status == "excluded")

    # Write summaries
    subj_csv = write_subject_summary(
        records, OUT_ROOT / "phase3r_1_fod_preparation_summary.csv"
    )
    site_csv, site_rows = write_site_summary(
        records, EXPECTED_COUNTS, OUT_ROOT / "phase3r_1_fod_site_summary.csv"
    )

    log.info("=" * 64)
    log.info("Done: %d success / %d failed / %d skipped / %d excluded",
             n_success, n_failed, n_skipped, n_excluded)
    log.info("Subject CSV → %s", subj_csv.relative_to(REPO_ROOT))
    log.info("Site CSV    → %s", site_csv.relative_to(REPO_ROOT))

    # Gate check
    log.info("=" * 64)
    log.info("GATE CHECK — Phase 3R.1 (expected 229/229 clean subjects)")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        site_success = sum(1 for r in records if r.site == site and r.status == "success")
        ok   = site_success == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, site_success, expected)
        if not ok:
            all_pass = False

    # Confirm IP_1 produced no output
    ip1_success = sum(1 for r in records if r.site == "IP_1" and r.status == "success")
    if ip1_success > 0:
        log.error("  ✗ [IP_1]  %d unexpected success records — IP_1 should be excluded!",
                  ip1_success)
        all_pass = False
    else:
        log.info("  ✓ [IP_1]  correctly excluded — 0 output records")

    if all_pass:
        log.info("Gate: PASSED — proceed to Phase 3R.2 (tractography from processed_v2b)")
    else:
        log.warning("Gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
