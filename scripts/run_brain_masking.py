#!/usr/bin/env python3
"""
Phase 2.3 — BNI DTI Brain Masking and Tensor QC

Applies median_otsu brain extraction to the B0 image and produces QC metrics
for tensor maps (FA, MD, AD, RD) within the anatomical brain mask.

Usage:
    python scripts/run_brain_masking.py \
        --processed-root  data/processed/abide_ii/bni/ABIDEII-BNI_1 \
        --raw-root        data/raw \
        --summary         data/processed/phase2_3_brain_mask_tensor_qc_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from neurofiber.preprocessing.brain_masking import run_masking_batch
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_SEP = "=" * 62


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2.3: Brain masking and tensor QC for BNI subjects."
    )
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--raw-root",       type=Path, default=None)
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("data/processed/phase2_3_brain_mask_tensor_qc_summary.csv"),
    )
    p.add_argument("--session",   default="session_1")
    p.add_argument("--no-png",    action="store_true",
                   help="Skip PNG QC image generation.")
    return p.parse_args()


def _print_summary(records) -> None:
    total    = len(records)
    success  = sum(1 for r in records if r.status == "success")
    failed   = sum(1 for r in records if r.status == "failed")
    warnings = sum(1 for r in records if r.warning_message and r.status == "success")

    fa_vals   = [r.fa_mean for r in records if r.fa_mean is not None]
    md_vals   = [r.md_mean for r in records if r.md_mean is not None]
    cov_vals  = [r.mask_coverage_ratio for r in records if r.mask_coverage_ratio > 0]

    print(f"\n{_SEP}")
    print("  PHASE 2.3 — BRAIN MASK + TENSOR QC SUMMARY")
    print(_SEP)
    print(f"  Total subjects      : {total}")
    print(f"  Success             : {success}")
    print(f"  Failed              : {failed}")
    print(f"  With warnings       : {warnings}")
    if fa_vals:
        print(f"  FA  mean ± std      : {np.mean(fa_vals):.4f} ± {np.std(fa_vals):.4f}")
        print(f"  FA  range           : {min(fa_vals):.4f} – {max(fa_vals):.4f}")
        print(f"  MD  mean            : {np.mean(md_vals):.6f} mm²/s")
        print(f"  Mask coverage mean  : {np.mean(cov_vals)*100:.1f}%")
    print(_SEP)
    if failed:
        print("\n  Failed subjects:")
        for r in records:
            if r.status == "failed":
                print(f"    {r.subject_id:<12}  {r.warning_message}")
    if warnings:
        print("\n  Subjects with QC warnings:")
        for r in records:
            if r.warning_message and r.status == "success":
                print(f"    {r.subject_id:<12}  {r.warning_message}")
    print()


def main() -> int:
    args = parse_args()
    logger.info("Phase 2.3 — DTI brain masking and tensor QC")
    logger.info("  processed-root : %s", args.processed_root)

    records = run_masking_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        session=args.session,
        save_png=not args.no_png,
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_summary_row() for r in records])
    df.to_csv(args.summary, index=False)
    logger.info("Summary CSV → %s  (%d rows)", args.summary, len(df))

    _print_summary(records)
    return 1 if any(r.status == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
