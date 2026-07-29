"""
Phase 3.4 — Atlas Registration

Registers a standard atlas (Schaefer 2018 100-parcel 7-networks) from
MNI152NLin2009cAsym space into each subject's DTI (FA) voxel space using
a two-step DIPY affine pipeline:

  Step 1: Subject T1w → MNI T1w brain (affine 12-DOF)
           Apply inverse → atlas in T1w space
  Step 2: Subject FA → Subject T1w (rigid 6-DOF)
           Apply inverse → atlas in FA/DTI space

Outputs per subject:
  connectome/atlas/
    atlas_subject_space.nii.gz   atlas parcellation in subject DTI space
    atlas_registration_report.json
    atlas_labels.csv             label index → name mapping
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Atlas and template constants
# ---------------------------------------------------------------------------

_ATLAS_TEMPLATE    = "MNI152NLin2009cAsym"
_ATLAS_NAME        = "Schaefer2018"
_ATLAS_DESC        = "100Parcels7Networks"
_ATLAS_RESOLUTION  = 2
_ATLAS_N_ROIS      = 100
_MNI_T1_BRAIN_PATH: Optional[Path] = None  # set at first call to fetch_atlas()

# Schaefer 2018 100-parcel 7-networks label names (LH 1-50, RH 51-100)
_SCHAEFER_100_NETWORKS = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
]
# Approximate label names — canonical names require the full TSV from Schaefer
# The exact names vary per parcel; this is a human-readable index.
def _make_schaefer_labels(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        hemi = "L" if i <= n // 2 else "R"
        rows.append({"index": i, "name": f"Schaefer100_7Net_{hemi}_{i:03d}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AtlasInfo:
    atlas_path:   Path
    n_rois:       int
    label_df:     pd.DataFrame
    mni_t1_path:  Path
    atlas_affine: np.ndarray  # atlas native affine (MNI space)

    def to_dict(self) -> dict:
        return {
            "atlas_name":  _ATLAS_NAME,
            "atlas_desc":  _ATLAS_DESC,
            "atlas_res":   _ATLAS_RESOLUTION,
            "n_rois":      self.n_rois,
            "atlas_path":  str(self.atlas_path),
            "mni_t1_path": str(self.mni_t1_path),
        }


@dataclass
class RegisteredAtlas:
    atlas_subject_space_path: Path    # atlas_subject_space.nii.gz
    affine:       np.ndarray          # == FA affine (atlas is in FA space)
    n_rois:       int
    label_df:     pd.DataFrame
    registration_report: dict


# ---------------------------------------------------------------------------
# Atlas fetch
# ---------------------------------------------------------------------------

def fetch_atlas(cache_dir: Optional[Path] = None) -> AtlasInfo:
    """
    Fetch Schaefer 2018 100-parcel atlas and MNI T1w brain template via
    templateflow. Results are cached locally by templateflow.
    """
    import templateflow.api as tflow

    logger.info("Fetching %s atlas (%s) from templateflow …", _ATLAS_NAME, _ATLAS_DESC)
    atlas_files = tflow.get(
        _ATLAS_TEMPLATE,
        atlas=_ATLAS_NAME,
        desc=_ATLAS_DESC,
        extension=".nii.gz",
        resolution=_ATLAS_RESOLUTION,
    )
    if not atlas_files:
        raise RuntimeError(
            f"Could not fetch {_ATLAS_NAME} {_ATLAS_DESC} from templateflow. "
            "Check internet connection."
        )
    atlas_path = Path(atlas_files) if isinstance(atlas_files, (str, Path)) else Path(atlas_files[0])

    logger.info("Fetching MNI152NLin2009cAsym brain T1w template …")
    t1w_files = tflow.get(
        _ATLAS_TEMPLATE,
        resolution=_ATLAS_RESOLUTION,
        suffix="T1w",
        desc="brain",
        extension=".nii.gz",
    )
    if not t1w_files:
        raise RuntimeError("Could not fetch MNI T1w brain template from templateflow.")
    mni_t1_path = Path(t1w_files[0] if isinstance(t1w_files, list) else t1w_files)

    atlas_img   = nib.load(str(atlas_path))
    label_df    = _make_schaefer_labels(_ATLAS_N_ROIS)

    logger.info(
        "Atlas: %s  n_rois=%d  path=%s", _ATLAS_NAME, _ATLAS_N_ROIS, atlas_path.name
    )
    return AtlasInfo(
        atlas_path=atlas_path,
        n_rois=_ATLAS_N_ROIS,
        label_df=label_df,
        mni_t1_path=mni_t1_path,
        atlas_affine=atlas_img.affine,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _dipy_affine_register(
    static_data:       np.ndarray,
    static_affine:     np.ndarray,
    moving_data:       np.ndarray,
    moving_affine:     np.ndarray,
    level_iters:       list[int],
    sigmas:            list[float],
    factors:           list[int],
    transform_type:    str = "affine",
) -> object:
    """Run DIPY affine/rigid registration and return AffineMap."""
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import AffineTransform3D, RigidTransform3D

    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
    )
    transform = AffineTransform3D() if transform_type == "affine" else RigidTransform3D()
    affine_map = affreg.optimize(
        static_data.astype(np.float32),
        moving_data.astype(np.float32),
        transform, None,
        static_affine, moving_affine,
    )
    return affine_map


def register_atlas_to_subject(
    t1w_path:   Path,
    fa_path:    Path,
    atlas_info: AtlasInfo,
    output_dir: Path,
    raw_root:   Optional[Path] = None,
    level_iters: list[int]    = None,
    sigmas:      list[float]  = None,
    factors:     list[int]    = None,
) -> RegisteredAtlas:
    """
    Two-step affine atlas registration into subject DTI space:

      1. Subject T1w → MNI T1w brain (12-DOF affine)
         Apply inverse → atlas in T1w space (nearest-neighbour)
      2. Subject FA → T1w (6-DOF rigid)
         Apply inverse → atlas in FA space (nearest-neighbour)

    Outputs atlas_subject_space.nii.gz alongside a JSON report.
    """
    if raw_root is not None:
        guard_no_raw_write(output_dir, raw_root)

    if level_iters is None:
        level_iters = [200, 50, 10]
    if sigmas is None:
        sigmas = [3.0, 1.0, 0.0]
    if factors is None:
        factors = [8, 4, 2]

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)

    t1w_img  = nib.load(str(t1w_path))
    fa_img   = nib.load(str(fa_path))
    mni_img  = nib.load(str(atlas_info.mni_t1_path))
    atl_img  = nib.load(str(atlas_info.atlas_path))

    t1w_data  = np.asarray(t1w_img.dataobj, dtype=np.float32)
    fa_data   = np.asarray(fa_img.dataobj,  dtype=np.float32)
    mni_data  = np.asarray(mni_img.dataobj, dtype=np.float32)
    atl_data  = np.asarray(atl_img.dataobj, dtype=np.float32)

    # --- Step 1: T1w → MNI (affine 12-DOF) --------------------------------
    logger.debug("  Step 1: T1w → MNI affine …")
    reg1 = _dipy_affine_register(
        mni_data,  mni_img.affine,
        t1w_data,  t1w_img.affine,
        level_iters, sigmas, factors,
        transform_type="affine",
    )

    # Apply inverse (MNI → T1w) to atlas
    atlas_in_t1w = reg1.transform_inverse(
        atl_data,
        interpolation="nearest",
        image_grid2world=atl_img.affine,
        sampling_grid_shape=t1w_data.shape,
        sampling_grid2world=t1w_img.affine,
    )

    # --- Step 2: FA → T1w (rigid 6-DOF) ------------------------------------
    logger.debug("  Step 2: FA → T1w rigid …")
    reg2 = _dipy_affine_register(
        t1w_data,  t1w_img.affine,
        fa_data,   fa_img.affine,
        level_iters, sigmas, factors,
        transform_type="rigid",
    )

    # Apply inverse (T1w → FA) to atlas-in-T1w → atlas in FA space
    atlas_in_fa = reg2.transform_inverse(
        atlas_in_t1w,
        interpolation="nearest",
        image_grid2world=t1w_img.affine,
        sampling_grid_shape=fa_data.shape,
        sampling_grid2world=fa_img.affine,
    )

    atlas_in_fa = np.round(atlas_in_fa).astype(np.int16)

    # --- Save outputs -------------------------------------------------------
    atlas_nii_path = output_dir / "atlas_subject_space.nii.gz"
    nib.save(nib.Nifti1Image(atlas_in_fa, fa_img.affine), str(atlas_nii_path))

    labels_csv_path = output_dir / "atlas_labels.csv"
    atlas_info.label_df.to_csv(labels_csv_path, index=False)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    n_atlas_voxels = int(np.sum(atlas_in_fa > 0))
    n_rois_present = int(len(np.unique(atlas_in_fa[atlas_in_fa > 0])))

    report = {
        "atlas_name":        _ATLAS_NAME,
        "atlas_desc":        _ATLAS_DESC,
        "atlas_n_rois":      atlas_info.n_rois,
        "rois_present":      n_rois_present,
        "atlas_voxels":      n_atlas_voxels,
        "registration_method": "dipy_affine_t1w_mediated",
        "level_iters":       level_iters,
        "sigmas":            sigmas,
        "factors":           factors,
        "elapsed_sec":       round(elapsed, 1),
        "timestamp":         t0.isoformat(),
        "warning":           "Affine-only registration. Nonlinear (ANTs) recommended for production.",
    }
    report_path = output_dir / "atlas_registration_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(
        "  Atlas registered — %d/%d ROIs present in FA space  (%.1fs)",
        n_rois_present, atlas_info.n_rois, elapsed,
    )
    return RegisteredAtlas(
        atlas_subject_space_path=atlas_nii_path,
        affine=fa_img.affine,
        n_rois=atlas_info.n_rois,
        label_df=atlas_info.label_df,
        registration_report=report,
    )


def load_registered_atlas(atlas_dir: Path, label_df: pd.DataFrame) -> RegisteredAtlas:
    """Load a previously registered atlas from disk (skip re-registration)."""
    atlas_path  = atlas_dir / "atlas_subject_space.nii.gz"
    report_path = atlas_dir / "atlas_registration_report.json"
    if not atlas_path.exists():
        raise FileNotFoundError(f"atlas_subject_space.nii.gz not found in {atlas_dir}")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    img = nib.load(str(atlas_path))
    return RegisteredAtlas(
        atlas_subject_space_path=atlas_path,
        affine=img.affine,
        n_rois=int(report.get("atlas_n_rois", label_df["index"].max())),
        label_df=label_df,
        registration_report=report,
    )
