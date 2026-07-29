from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


@dataclass
class DTIResult:
    volume_count: Optional[int] = None
    bval_count: Optional[int] = None
    bvec_count: Optional[int] = None
    errors: list[str] = field(default_factory=list)
    status: str = "valid"


def validate_dti(dti_path: Path, bval_path: Path, bvec_path: Path) -> DTIResult:
    result = DTIResult()

    result.volume_count = _load_volume_count(dti_path, result)
    if result.status == "failed":
        return result

    result.bval_count = _load_bval_count(bval_path, result)
    if result.status == "failed":
        return result

    result.bvec_count = _load_bvec_count(bvec_path, result)
    if result.status == "failed":
        return result

    _check_consistency(result)
    return result


def _load_volume_count(dti_path: Path, result: DTIResult) -> Optional[int]:
    try:
        img = nib.load(str(dti_path))
        shape = img.shape
        return int(shape[3]) if len(shape) == 4 else 1
    except Exception as exc:
        result.errors.append(f"Cannot load DTI NIfTI: {exc}")
        result.status = "failed"
        return None


def _load_bval_count(bval_path: Path, result: DTIResult) -> Optional[int]:
    try:
        bvals = np.loadtxt(str(bval_path))
        return int(bvals.size)
    except Exception as exc:
        result.errors.append(f"Cannot load bval: {exc}")
        result.status = "failed"
        return None


def _load_bvec_count(bvec_path: Path, result: DTIResult) -> Optional[int]:
    try:
        bvecs = np.loadtxt(str(bvec_path))
        if bvecs.ndim == 1:
            # Single volume: 3 components stored as a flat row
            return 1
        # FSL convention: (3 rows × N cols).  Transposed: (N rows × 3 cols).
        if bvecs.shape[0] == 3:
            return int(bvecs.shape[1])
        return int(bvecs.shape[0])
    except Exception as exc:
        result.errors.append(f"Cannot load bvec: {exc}")
        result.status = "failed"
        return None


def _check_consistency(result: DTIResult) -> None:
    vols = result.volume_count
    bvals = result.bval_count
    bvecs = result.bvec_count

    if vols != bvals:
        result.errors.append(
            f"Volume count ({vols}) ≠ bval count ({bvals})"
        )
    if vols != bvecs:
        result.errors.append(
            f"Volume count ({vols}) ≠ bvec count ({bvecs})"
        )
    if result.errors:
        result.status = "dti_mismatch"
