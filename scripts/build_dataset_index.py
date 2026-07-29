#!/usr/bin/env python3
"""Merge one or more source CSVs into a validated, unified dataset index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as  python scripts/build_dataset_index.py  from ai_pipeline/
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofiber.registry import load_dataset_index, save_dataset_index, validate_records
from neurofiber.utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified NeuroFiber dataset index.")
    p.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="One or more source CSV files to merge.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_index.csv"),
        help="Output path for the unified index (default: data/dataset_index.csv).",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation (not recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_records = []
    for src in args.sources:
        logger.info("Loading %s …", src)
        records = load_dataset_index(src)
        logger.info("  → %d records", len(records))
        all_records.extend(records)

    logger.info("Total records loaded: %d", len(all_records))

    if not args.no_validate:
        logger.info("Validating …")
        errors = validate_records(all_records, raise_on_error=False)
        if errors:
            for e in errors:
                logger.error("  %s", e)
            logger.error("%d validation error(s). Fix the source data and re-run.", len(errors))
            sys.exit(1)
        logger.info("Validation passed.")

    save_dataset_index(all_records, args.output)
    logger.info("Saved index → %s  (%d records)", args.output, len(all_records))


if __name__ == "__main__":
    main()
