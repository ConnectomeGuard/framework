"""
NeuroFiber Phase 2R.3 — Brain Masking and Tensor QC Runner

Generates DIPY median_otsu brain masks, per-subject tensor QC metrics,
site-level stability analysis, and representative QC PNGs for all 277
valid DTI subjects in the processed_v2b canonical pipeline.

Usage
─────
Dry-run (count subjects, no masking):
    python scripts/run_brain_masking_v2b.py --dry-run

Full run:
    python scripts/run_brain_masking_v2b.py

Resume (skip already-masked subjects):
    python scripts/run_brain_masking_v2b.py --resume

Restrict sites:
    python scripts/run_brain_masking_v2b.py --sites BNI NYU_1

Outputs
───────
    data/processed_v2b/abide_ii/<site>/.../<subj>/session_1/dti_1/qc/
        brain_mask.nii.gz  b0_brain.nii.gz
        tensor_qc_report.json
        b0_mask_overlay.png  fa_mid_slice.png  md_mid_slice.png

    data/processed_v2b/
        phase2r_3_brain_mask_tensor_qc_summary.csv
        phase2r_3_site_qc_summary.csv
        phase2r_3_site_stability_analysis.csv

    data/processed_v2b/site_qc/
        <SITE>_fa_representative.png
        <SITE>_md_representative.png
        <SITE>_mask_overlay_representative.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.preprocessing.brain_masking_v2b import (
    SITES,
    _SITE_FOLDER_MAP,
    run_masking_batch,
    write_site_summary,
    write_stability_analysis,
    write_subject_summary,
    write_site_qc_pngs,
)

log = logging.getLogger("brain_mask_v2b")

PROCESSED_V2B_ROOT = REPO_ROOT / "data" / "processed_v2b" / "abide_ii"
RAW_ROOT           = REPO_ROOT / "data" / "raw"
OUT_ROOT           = REPO_ROOT / "data" / "processed_v2b"
SITE_QC_DIR        = OUT_ROOT / "site_qc"

# Forbidden write targets
_FORBIDDEN = [
    REPO_ROOT / "data" / "raw",
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "processed_v2",
]

EXPECTED_COUNTS = {
    "BNI":    58,
    "IP_1":   48,
    "NYU_1":  55,
    "NYU_2":  19,
    "SDSU_1": 57,
    "TCD_1":  40,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2R.3 — Brain masking and tensor QC from processed_v2b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sites", nargs="+", default=SITES, choices=SITES, metavar="SITE")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects with an existing successful tensor_qc_report.json.")
    p.add_argument("--no-png", dest="save_png", action="store_false",
                   help="Skip PNG generation (faster).")
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
    log.info("DRY RUN — Phase 2R.3 Brain Masking and Tensor QC")
    log.info("Input root: %s", PROCESSED_V2B_ROOT)
    log.info("=" * 64)
    total = 0
    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = PROCESSED_V2B_ROOT / folder
        if not site_root.exists():
            log.warning("[%s] not found: %s", site, site_root)
            continue
        dti_dirs = sorted(site_root.rglob("dti_1"))
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "dwi_preprocessed.nii.gz").exists()]
        done = sum(1 for d in dti_dirs
                   if (d / "qc" / "tensor_qc_report.json").exists())
        log.info("[%s]  %d subjects  (%d already done)", site, len(dti_dirs), done)
        total += len(dti_dirs)
    log.info("=" * 64)
    log.info("Total: %d subjects", total)


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    _safety_check()

    if args.dry_run:
        _dry_run(args.sites)
        return

    log.info("=" * 64)
    log.info("Phase 2R.3 — Brain Masking and Tensor QC from processed_v2b")
    log.info("Sites      : %s", args.sites)
    log.info("Resume     : %s", args.resume)
    log.info("PNG        : %s", args.save_png)
    log.info("Input root : %s", PROCESSED_V2B_ROOT)
    log.info("=" * 64)

    records = run_masking_batch(
        processed_v2b_root=PROCESSED_V2B_ROOT,
        raw_root=RAW_ROOT,
        sites=args.sites,
        skip_if_exists=args.resume,
        save_png=args.save_png,
    )

    n_success = sum(1 for r in records if r.status == "success")
    n_failed  = sum(1 for r in records if r.status == "failed")

    # Write outputs
    subj_csv = write_subject_summary(
        records, OUT_ROOT / "phase2r_3_brain_mask_tensor_qc_summary.csv"
    )
    site_csv, site_rows = write_site_summary(
        records, OUT_ROOT / "phase2r_3_site_qc_summary.csv"
    )
    stab_csv = write_stability_analysis(
        records, OUT_ROOT / "phase2r_3_site_stability_analysis.csv"
    )

    # Site-level PNGs
    if args.save_png:
        write_site_qc_pngs(records, PROCESSED_V2B_ROOT, SITE_QC_DIR)

    # Summary
    log.info("=" * 64)
    log.info("Done: %d success / %d failed", n_success, n_failed)
    log.info("Subject CSV   → %s", subj_csv.relative_to(REPO_ROOT))
    log.info("Site CSV      → %s", site_csv.relative_to(REPO_ROOT))
    log.info("Stability CSV → %s", stab_csv.relative_to(REPO_ROOT))
    if args.save_png:
        log.info("Site QC PNGs  → %s", SITE_QC_DIR.relative_to(REPO_ROOT))

    # Gate check
    log.info("=" * 64)
    log.info("GATE CHECK — Phase 2R.3")
    all_pass = True
    for site, expected in EXPECTED_COUNTS.items():
        site_success = sum(1 for r in records if r.site == site and r.status == "success")
        ok = site_success == expected
        mark = "✓" if ok else "✗"
        log.info("  %s [%s]  success=%d / expected=%d", mark, site, site_success, expected)
        if not ok:
            all_pass = False

    log.info("-" * 64)
    log.info("Site-level QC summary:")
    for row in site_rows:
        flag = f"  ⚠ {row['outlier_flag']}" if row.get("outlier_flag") else ""
        log.info(
            "  [%s]  n=%d  FA=%.4f±%.4f  MD=%.6f  mask_voxels=%.0f%s",
            row["site"], row["subjects"],
            row["mean_fa"] or 0, row["std_fa"] or 0,
            row["mean_md"] or 0, row["mean_mask_voxels"] or 0,
            flag,
        )

    if all_pass:
        log.info("Gate: PASSED — site-level QC complete")
        log.info("Review outlier_flag column before proceeding to Phase 3R.1 (FOD generation).")
    else:
        log.warning("Gate: FAILED — subject count mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
