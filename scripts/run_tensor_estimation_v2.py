"""
NeuroFiber Phase 2R.2 — Tensor Estimation Runner

Fits FA / MD / AD / RD tensor maps from Phase 2R.1 standardized DWI outputs,
writes per-subject and site summary CSVs, and generates a v1 vs v2 comparison.

Usage
─────
Dry-run (shows subjects; no tensor fitting):
    python scripts/run_tensor_estimation_v2.py --dry-run

Full run:
    python scripts/run_tensor_estimation_v2.py

Resume (skip already-completed subjects):
    python scripts/run_tensor_estimation_v2.py --resume

Restrict to specific sites:
    python scripts/run_tensor_estimation_v2.py --sites BNI NYU_1

Outputs
───────
    data/processed_v2/abide_ii/<site>/.../<subj>/session_1/dti_1/tensor/
        FA.nii.gz  MD.nii.gz  AD.nii.gz  RD.nii.gz
        tensor_fit_report.json
    data/processed_v2/phase2r_2_tensor_summary.csv
    data/processed_v2/phase2r_2_tensor_site_summary.csv
    data/processed_v2/phase2r_2_tensor_v1_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.tensor.tensor_estimation_v2 import (
    SITES,
    _SITE_FOLDER_MAP,
    run_tensor_batch,
    write_subject_summary,
    write_site_summary,
    write_v1_comparison,
)

log = logging.getLogger("tensor_v2")

PROCESSED_V2_ROOT = REPO_ROOT / "data" / "processed_v2" / "abide_ii"
PROCESSED_V1_ROOT = REPO_ROOT / "data" / "processed" / "abide_ii"
RAW_ROOT          = REPO_ROOT / "data" / "raw"

EXPECTED_COUNTS = {
    "BNI":    58,
    "IP_1":   48,
    "NYU_1":  55,
    "NYU_2":  19,
    "SDSU_1": 57,
    "TCD_1":  40,
}

OUT_ROOT = REPO_ROOT / "data" / "processed_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2R.2 — Tensor estimation from processed_v2 DWI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=SITES, choices=SITES,
                   metavar="SITE", help="Sites to process (default: all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects that already have a successful tensor_fit_report.json.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="List subjects without fitting tensors.")
    p.add_argument("--n-workers", dest="n_workers", type=int, default=4,
                   help="Parallel workers for tensor fitting (default: 4).")
    p.add_argument("--input-root", dest="input_root", default=None,
                   help="Override processed_v2 root (e.g. data/processed_v2b/abide_ii).")
    p.add_argument("--output-root", dest="output_root", default=None,
                   help="Override output root for CSVs (default: parent of input_root).")
    p.add_argument("--csv-prefix", dest="csv_prefix", default="phase2r_2",
                   help="Prefix for output CSV filenames (default: phase2r_2).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _dry_run(sites: list[str]) -> None:
    log.info("=" * 64)
    log.info("DRY RUN — Phase 2R.2 Tensor Estimation")
    log.info("Input root: %s", PROCESSED_V2_ROOT)
    log.info("=" * 64)
    total = 0
    for site in sites:
        folder = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = PROCESSED_V2_ROOT / folder
        if not site_root.exists():
            log.warning("[%s] not found: %s", site, site_root)
            continue
        dti_dirs = sorted(site_root.rglob("dti_1"))
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "dwi_preprocessed.nii.gz").exists()]
        done = sum(
            1 for d in dti_dirs
            if (d / "tensor" / "tensor_fit_report.json").exists()
        )
        log.info("[%s]  %d subjects  (%d already done)", site, len(dti_dirs), done)
        total += len(dti_dirs)
    log.info("=" * 64)
    log.info("Total: %d subjects", total)
    log.info("Re-run without --dry-run to process.")


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    # Resolve roots — allow override via --input-root / --output-root
    input_root = Path(args.input_root) if args.input_root else PROCESSED_V2_ROOT
    out_root   = Path(args.output_root) if args.output_root else input_root.parent
    prefix     = args.csv_prefix

    # Safety: never write into processed_v1
    old_processed = REPO_ROOT / "data" / "processed"
    assert str(input_root.resolve()) != str(old_processed.resolve()), (
        "SAFETY: input root must not point to data/processed (v1 pipeline)."
    )

    if args.dry_run:
        _dry_run(args.sites)
        return

    log.info("=" * 64)
    log.info("Tensor Estimation from %s", input_root)
    log.info("Sites     : %s", args.sites)
    log.info("Resume    : %s", args.resume)
    log.info("Workers   : %d", args.n_workers)
    log.info("Output    : %s", out_root)
    log.info("=" * 64)

    records = run_tensor_batch(
        processed_v2_root=input_root,
        raw_root=RAW_ROOT,
        sites=args.sites,
        skip_if_exists=args.resume,
        n_workers=args.n_workers,
    )

    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")
    n_skipped = sum(1 for r in records if r.status == "skipped")

    # Write subject summary
    subj_csv = write_subject_summary(
        records, out_root / f"{prefix}_tensor_summary.csv"
    )

    # Write site summary
    site_csv = write_site_summary(
        records, EXPECTED_COUNTS, out_root / f"{prefix}_tensor_site_summary.csv"
    )

    # Write v1 comparison
    comp_csv, site_analysis = write_v1_comparison(
        records, PROCESSED_V1_ROOT, out_root / f"{prefix}_tensor_v1_comparison.csv"
    )

    log.info("=" * 64)
    log.info("Done: %d success / %d failed / %d skipped", n_success, n_failed, n_skipped)
    def _rel(p: Path) -> Path:
        try:
            return p.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return p
    log.info("Subject CSV  → %s", _rel(subj_csv))
    log.info("Site CSV     → %s", _rel(site_csv))
    log.info("Comparison   → %s", _rel(comp_csv))

    log.info("-" * 64)
    log.info("Site FA change analysis (v2 − v1):")
    for site, info in site_analysis.items():
        if info["n"] == 0:
            log.info("  [%s]  no matched v1 data", site)
        else:
            log.info(
                "  [%s]  n=%d  mean_abs_FA_delta=%.5f  → %s",
                site, info["n"], info["mean_abs_fa_delta"], info["classification"],
            )

    # Gate summary
    log.info("=" * 64)
    log.info("GATE CHECK — Phase 2R.2")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        site_recs = [r for r in records if r.site == site and r.status == "success"]
        ok = len(site_recs) == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, len(site_recs), expected)
        if not ok:
            all_pass = False

    if all_pass:
        log.info("Gate: PASSED — proceed to Phase 2R.3 (brain masking + tensor QC)")
    else:
        log.warning("Gate: FAILED — investigate failed subjects before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
