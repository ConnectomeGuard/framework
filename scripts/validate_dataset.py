#!/usr/bin/env python3
"""
Scan raw ABIDE II directory structure, validate modalities and DTI consistency,
and produce a dataset_index.csv.

Usage:
    python scripts/validate_dataset.py \
        --data-root data/raw/abide_ii/bni/ABIDEII-BNI_1 \
        --site BNI \
        --dataset ABIDE_II \
        --output data/dataset_index.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from neurofiber.registry.scanner import scan_subjects
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_SEP = "=" * 56


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate ABIDE II dataset structure and produce dataset_index.csv."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing subject folders (e.g. ABIDEII-BNI_1/).",
    )
    p.add_argument("--site",    default="BNI",      help="Site label (default: BNI).")
    p.add_argument("--dataset", default="ABIDE_II", help="Dataset label (default: ABIDE_II).")
    p.add_argument("--session", default="session_1", help="Session directory name (default: session_1).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_index.csv"),
        help="Output CSV path (default: data/dataset_index.csv).",
    )
    return p.parse_args()


def _print_summary(records) -> None:
    total        = len(records)
    valid        = sum(1 for r in records if r.validation_status == "valid")
    missing      = sum(1 for r in records if r.validation_status == "missing_files")
    mismatch     = sum(1 for r in records if r.validation_status == "dti_mismatch")
    failed       = sum(1 for r in records if r.validation_status == "failed")

    print(f"\n{_SEP}")
    print("  DATASET VALIDATION SUMMARY")
    print(_SEP)
    print(f"  Total subjects         : {total}")
    print(f"  Valid (all files + DTI): {valid}")
    print(f"  Missing modalities     : {missing}")
    print(f"  DTI inconsistent       : {mismatch}")
    print(f"  Failed to load         : {failed}")
    print(_SEP)

    print("\n  Modality presence:")
    for mod in ("anat", "dti", "bval", "bvec", "rest"):
        count = sum(1 for r in records if getattr(r, f"{mod}_exists"))
        bar   = "#" * count + "-" * (total - count)
        print(f"    {mod:4s}  [{bar}]  {count}/{total}")

    if mismatch or failed:
        print("\n  Subjects with issues:")
        for r in records:
            if r.validation_status not in ("valid", "missing_files"):
                print(f"    {r.subject_id:<30}  {r.validation_status}")
                print(f"      volumes={r.dti_volume_count}  bvals={r.bval_count}  bvecs={r.bvec_count}")

    if missing:
        print("\n  Subjects with missing files:")
        for r in records:
            if r.validation_status == "missing_files":
                absent = [m for m in ("anat", "dti", "bval", "bvec", "rest")
                          if not getattr(r, f"{m}_exists")]
                print(f"    {r.subject_id:<30}  missing: {', '.join(absent)}")

    print()


def main() -> int:
    args = parse_args()

    logger.info("Scanning %s …", args.data_root)
    records = scan_subjects(
        args.data_root,
        site=args.site,
        dataset=args.dataset,
        session=args.session,
    )

    if not records:
        logger.error("No subject directories found under %s", args.data_root)
        return 1

    df = pd.DataFrame([r.to_dict() for r in records])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved dataset_index.csv → %s  (%d records)", args.output, len(records))

    _print_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
