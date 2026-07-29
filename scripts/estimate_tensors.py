#!/usr/bin/env python3
"""
Phase 2.2 — BNI DTI Tensor Estimation

Fits diffusion tensor maps (FA, MD, AD, RD) for all Phase 2.1 corrected
BNI subjects.

Usage:
    python scripts/estimate_tensors.py \
        --processed-root  data/processed/abide_ii/bni/ABIDEII-BNI_1 \
        --raw-root        data/raw \
        --summary         data/processed/phase2_tensor_estimation_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from neurofiber.preprocessing.tensor_estimation import run_tensor_batch
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_SEP = "=" * 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2.2: Fit diffusion tensor maps for BNI subjects."
    )
    p.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Phase 2.1 output root (contains per-subject subdirs).",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Raw data root — used to verify no outputs land in raw/ (recommended).",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("data/processed/phase2_tensor_estimation_summary.csv"),
        help="Output summary CSV path.",
    )
    p.add_argument("--session", default="session_1")
    return p.parse_args()


def _print_summary(records) -> None:
    total    = len(records)
    success  = sum(1 for r in records if r.status == "success")
    failed   = sum(1 for r in records if r.status == "failed")
    skipped  = sum(1 for r in records if r.status == "skipped")

    fa_vals  = [r.fa_mean for r in records if r.fa_mean is not None]
    md_vals  = [r.md_mean for r in records if r.md_mean is not None]

    print(f"\n{_SEP}")
    print("  PHASE 2.2 — TENSOR ESTIMATION SUMMARY")
    print(_SEP)
    print(f"  Total subjects    : {total}")
    print(f"  Success           : {success}")
    print(f"  Failed            : {failed}")
    print(f"  Skipped           : {skipped}")
    if fa_vals:
        import numpy as np
        print(f"  FA  mean ± std    : {np.mean(fa_vals):.4f} ± {np.std(fa_vals):.4f}")
        print(f"  FA  min  / max    : {min(fa_vals):.4f} / {max(fa_vals):.4f}")
        print(f"  MD  mean          : {np.mean(md_vals):.6f}")
    print(_SEP)
    if failed:
        print("\n  Failed subjects:")
        for r in records:
            if r.status == "failed":
                print(f"    {r.subject_id:<12}  {r.error_message}")
    print()


def main() -> int:
    args = parse_args()

    logger.info("Phase 2.2 — DTI tensor estimation")
    logger.info("  processed-root : %s", args.processed_root)
    if args.raw_root:
        logger.info("  raw-root       : %s", args.raw_root)

    records = run_tensor_batch(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        session=args.session,
    )

    # save summary CSV
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_summary_row() for r in records])
    df.to_csv(args.summary, index=False)
    logger.info("Summary CSV → %s  (%d rows)", args.summary, len(df))

    _print_summary(records)

    return 1 if any(r.status == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
