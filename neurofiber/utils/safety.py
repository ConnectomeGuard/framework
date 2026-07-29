"""
Raw-data write guard — shared across all Phase 2+ preprocessing modules.

The invariant: processed outputs must never land inside data/raw/.
"""

from __future__ import annotations

from pathlib import Path


def guard_no_raw_write(output_dir: Path, raw_root: Path) -> None:
    """
    Raise ValueError immediately if *output_dir* is inside *raw_root*.

    Called at the start of every function that writes files, so any
    misconfiguration fails loudly before a single byte is written.
    """
    try:
        Path(output_dir).resolve().relative_to(Path(raw_root).resolve())
        raise ValueError(
            f"SAFETY: output_dir '{output_dir}' is inside the raw data tree "
            f"'{raw_root}'. Raw data must never be modified. "
            f"Use a path under data/processed/ instead."
        )
    except ValueError as exc:
        if "SAFETY" in str(exc):
            raise
        # relative_to() raised because output_dir is NOT under raw_root — safe.
