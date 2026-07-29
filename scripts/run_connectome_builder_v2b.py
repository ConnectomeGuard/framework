"""
NeuroFiber Phase 3R.3 — Connectome Builder Runner

Builds Schaefer-100 structural connectivity matrices from Phase 3R.2
streamlines for 229 clean DTI subjects, reusing v1 atlas registrations.

Usage
─────
Dry-run (count subjects, no building):
    python scripts/run_connectome_builder_v2b.py --dry-run

Full run:
    python scripts/run_connectome_builder_v2b.py

Resume (skip already-built subjects):
    python scripts/run_connectome_builder_v2b.py --resume

Outputs
───────
    data/processed_v2b/abide_ii/<site>/.../<subj>/session_1/dti_1/connectome/
        count_matrix.npy  mean_length_matrix.npy
        mean_fa_matrix.npy  mean_md_matrix.npy
        mean_ad_matrix.npy  mean_rd_matrix.npy
        connectome_report.json

    data/processed_v2b/
        phase3r_3_connectome_summary.csv
        phase3r_3_connectome_site_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.connectome.connectome_builder_v2b import (
    ATLAS_NAME,
    ATLAS_N_ROIS,
    CLEAN_SITES,
    EXCLUDED_SITES,
    EXPECTED_COUNTS,
    _SITE_FOLDER_MAP,
    run_connectome_batch,
    write_site_summary,
    write_subject_summary,
)

log = logging.getLogger("connectome_v2b")

PROCESSED_V2B_ROOT = REPO_ROOT / "data" / "processed_v2b" / "abide_ii"
V1_PROCESSED_ROOT  = REPO_ROOT / "data" / "processed" / "abide_ii"
RAW_ROOT           = REPO_ROOT / "data" / "raw"
OUT_ROOT           = REPO_ROOT / "data" / "processed_v2b"

_FORBIDDEN = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "processed_v2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3R.3 — Connectome construction from processed_v2b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=CLEAN_SITES,
                   choices=CLEAN_SITES, metavar="SITE")
    p.add_argument("--resume", action="store_true")
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
    log.info("DRY RUN — Phase 3R.3 Connectome Construction")
    log.info("Atlas: %s (%d ROIs)", ATLAS_NAME, ATLAS_N_ROIS)
    log.info("Atlas source: %s", V1_PROCESSED_ROOT)
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
                    (d / "tractography" / "streamlines.trk").exists()]
        done = sum(1 for d in dti_dirs
                   if (d / "connectome" / "connectome_report.json").exists())
        # Atlas availability check
        folder_v1 = V1_PROCESSED_ROOT / folder
        atlas_count = len(list(folder_v1.glob("*/*/session_1/connectome/atlas/atlas_subject_space.nii.gz")))
        log.info("[%s]  %d subjects  (%d done)  atlas_available=%d  expected=%d",
                 site, len(dti_dirs), done, atlas_count, EXPECTED_COUNTS.get(site, "?"))
        total += len(dti_dirs)
    log.info("=" * 64)
    log.info("Total: %d clean subjects", total)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    sites = [s for s in args.sites if s not in EXCLUDED_SITES]

    if args.dry_run:
        _dry_run(sites)
        return

    log.info("=" * 64)
    log.info("Phase 3R.3 — Connectome Construction from processed_v2b")
    log.info("Atlas      : %s (%d ROIs)", ATLAS_NAME, ATLAS_N_ROIS)
    log.info("Atlas root : %s", V1_PROCESSED_ROOT)
    log.info("Sites      : %s", sites)
    log.info("Excluded   : %s", EXCLUDED_SITES)
    log.info("Resume     : %s", args.resume)
    log.info("=" * 64)

    records = run_connectome_batch(
        processed_v2b_root=PROCESSED_V2B_ROOT,
        v1_processed_root=V1_PROCESSED_ROOT,
        raw_root=RAW_ROOT,
        sites=sites,
        skip_if_exists=args.resume,
    )

    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")

    subj_csv = write_subject_summary(
        records, OUT_ROOT / "phase3r_3_connectome_summary.csv"
    )
    site_csv, site_rows = write_site_summary(
        records, OUT_ROOT / "phase3r_3_connectome_site_summary.csv"
    )

    log.info("=" * 64)
    log.info("Done: %d success / %d failed", n_success, n_failed)
    log.info("Subject CSV → %s", subj_csv.relative_to(REPO_ROOT))
    log.info("Site CSV    → %s", site_csv.relative_to(REPO_ROOT))

    log.info("-" * 64)
    log.info("Site summary:")
    for row in site_rows:
        log.info(
            "  [%s]  n=%s  mean_edges=%.0f  mean_density=%.4f  mean_used=%.0f%s",
            row["site"], row["subjects"],
            row["mean_nonzero_edges"] or 0,
            row["mean_density"] or 0,
            row["mean_streamlines_used"] or 0,
            f"  NOTE: {row['notes']}" if row.get("notes") else "",
        )

    # Gate check
    log.info("=" * 64)
    log.info("GATE CHECK — Phase 3R.3 (expected 229/229)")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        n    = sum(1 for r in records if r.site == site and r.status == "success")
        ok   = n == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, n, expected)
        if not ok:
            all_pass = False

    zero_edge = [r for r in records if r.status == "success" and r.nonzero_edges == 0]
    if zero_edge:
        log.error("  ✗ %d subjects have zero nonzero edges!", len(zero_edge))
        all_pass = False
    else:
        log.info("  ✓ No subjects with zero-edge connectomes")

    if all_pass:
        log.info("Gate: PASSED — proceed to Phase 3R.4 (connectome QC and harmonization)")
    else:
        log.warning("Gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
