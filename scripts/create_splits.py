#!/usr/bin/env python3
"""Create train / validation / test splits from the dataset index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofiber.registry import load_dataset_index, random_split, site_heldout_split, save_splits
from neurofiber.utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create dataset splits for NeuroFiber.")
    p.add_argument(
        "--index",
        type=Path,
        default=Path("data/dataset_index.csv"),
        help="Path to the dataset index CSV (default: data/dataset_index.csv).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to write split CSVs (default: data/splits).",
    )
    p.add_argument(
        "--strategy",
        choices=["random", "site_heldout"],
        default="random",
        help="Split strategy (default: random).",
    )
    p.add_argument(
        "--heldout-site",
        type=str,
        default=None,
        help="Site name to hold out (required when --strategy=site_heldout).",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading index from %s …", args.index)
    records = load_dataset_index(args.index)
    logger.info("  → %d records", len(records))

    if args.strategy == "random":
        logger.info(
            "Applying random split (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d) …",
            args.train_ratio * 100,
            args.val_ratio * 100,
            (1 - args.train_ratio - args.val_ratio) * 100,
            args.seed,
        )
        split_records = random_split(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.strategy == "site_heldout":
        if not args.heldout_site:
            logger.error("--heldout-site is required when using site_heldout strategy.")
            sys.exit(1)
        logger.info("Applying site-held-out split (held-out site: %s, seed=%d) …", args.heldout_site, args.seed)
        split_records = site_heldout_split(
            records,
            heldout_site=args.heldout_site,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    else:
        logger.error("Unknown strategy: %s", args.strategy)
        sys.exit(1)

    written = save_splits(split_records, args.output_dir)

    for split_name, path in sorted(written.items()):
        count = sum(1 for r in split_records if r.split == split_name)
        logger.info("  %-22s → %s  (%d records)", split_name, path, count)

    logger.info("Done.")


if __name__ == "__main__":
    main()
