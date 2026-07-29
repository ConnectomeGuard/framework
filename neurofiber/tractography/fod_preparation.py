"""
Phase 3.1 — Multi-site FOD / Orientation Preparation

Prepares clean DTI subjects for tractography by estimating fiber orientation
information using DIPY CsaOdfModel (constant solid angle ODF).

Only subjects from sites with clean FA statistics are processed:
  BNI, NYU_1, NYU_2, SDSU_1, TCD_1  (IP_1 excluded — b=1000 FA inflation)

Quality filter: 0.15 <= fa_mean <= 0.40 per subject.

Per-subject outputs written to dti_1/fod/:
  peaks.pam5               DIPY peak directions (PeaksAndMetrics)
  fod_prep_report.json     per-subject stats and QC
  backend_used.txt         "dipy_csa" or "mrtrix3"
  optional (MRtrix3 only):
    dwi.mif
    response.txt
    fod.mif
    mrtrix_commands.log
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.io.peaks import save_peaks
from dipy.reconst.shm import CsaOdfModel

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# Acceptable whole-brain FA range for clean tractography preparation
FA_MIN_CLEAN: float = 0.15
FA_MAX_CLEAN: float = 0.40

# sh_order=6 needs ≥28 gradient directions; BNI has 32 DWI dirs — the minimum site.
_SH_ORDER: int = 6

# b-value threshold to separate B0 volumes from DWI volumes
_B0_THRESHOLD: int = 50

# Default sites for Phase 3.1 (IP_1 excluded — FA inflation)
CLEAN_DTI_SITES: list[str] = ["bni", "nyu1", "nyu2", "sdsu", "tcd"]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _mrtrix3_available() -> bool:
    """Return True if MRtrix3 binaries are on PATH."""
    return (
        shutil.which("mrconvert") is not None
        and shutil.which("dwi2response") is not None
        and shutil.which("dwi2fod") is not None
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FODPrepRecord:
    site: str
    dataset: str
    subject_id: str
    fa_mean: float
    backend: str
    input_volume_count: int
    bval_count: int
    bvec_count: int
    b0_count: int
    dwi_count: int
    output_created: bool
    status: str           # "success" | "skipped" | "failed"
    warning_message: Optional[str] = field(default=None)
    error_message: Optional[str] = field(default=None)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        return {
            "site":                 self.site,
            "dataset":              self.dataset,
            "subject_id":           self.subject_id,
            "fa_mean":              round(self.fa_mean, 4),
            "backend":              self.backend,
            "input_volume_count":   self.input_volume_count,
            "bval_count":           self.bval_count,
            "bvec_count":           self.bvec_count,
            "b0_count":             self.b0_count,
            "dwi_count":            self.dwi_count,
            "output_created":       self.output_created,
            "status":               self.status,
            "warning_message":      self.warning_message,
            "error_message":        self.error_message,
        }


# ---------------------------------------------------------------------------
# Core per-subject function
# ---------------------------------------------------------------------------

def prepare_subject_fod(
    dti_dir: Path,
    output_dir: Path,
    raw_root: Path,
    fa_min: float = FA_MIN_CLEAN,
    fa_max: float = FA_MAX_CLEAN,
    use_mrtrix: Optional[bool] = None,
) -> FODPrepRecord:
    """
    Prepare FOD / orientation outputs for one subject.

    Parameters
    ----------
    dti_dir :
        Subject's dti_1/ directory containing dti_corrected.nii.gz,
        dti.bval, dti.bvec, tensor/, and qc/.
    output_dir :
        Destination directory for fod/ outputs (typically dti_dir / "fod").
    raw_root :
        Raw data root — safety guard prevents writing here.
    fa_min, fa_max :
        Inclusive FA range; subjects outside are skipped.
    use_mrtrix :
        None  → auto-detect MRtrix3 on PATH.
        True  → require MRtrix3 (fail if absent).
        False → always use DIPY.
    """
    # --- path decomposition ---
    subject_id = dti_dir.parent.parent.name           # e.g. "29048"
    dataset    = dti_dir.parent.parent.parent.name    # e.g. "ABIDEII-BNI_1"
    site       = dti_dir.parent.parent.parent.parent.name  # e.g. "bni"

    guard_no_raw_write(output_dir, raw_root)

    # --- validate required inputs ---
    required = {
        "dti_corrected.nii.gz":   dti_dir / "dti_corrected.nii.gz",
        "dti.bval":               dti_dir / "dti.bval",
        "dti.bvec":               dti_dir / "dti.bvec",
        "brain_mask.nii.gz":      dti_dir / "qc" / "brain_mask.nii.gz",
        "tensor_qc_report.json":  dti_dir / "qc" / "tensor_qc_report.json",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        return _failed_record(
            site, dataset, subject_id,
            f"Missing required inputs: {missing}",
        )

    # --- read FA mean from Phase 2.3 QC report ---
    qc_report = json.loads((dti_dir / "qc" / "tensor_qc_report.json").read_text())
    fa_mean   = float(qc_report.get("fa_mean", 0.0))

    if not (fa_min <= fa_mean <= fa_max):
        logger.warning(
            "[%s/%s] skipped — FA=%.4f outside clean range [%.2f, %.2f]",
            site, subject_id, fa_mean, fa_min, fa_max,
        )
        return FODPrepRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            fa_mean=fa_mean, backend="none",
            input_volume_count=0, bval_count=0, bvec_count=0,
            b0_count=0, dwi_count=0,
            output_created=False, status="skipped",
            warning_message=(
                f"FA={fa_mean:.4f} outside clean range [{fa_min:.2f}, {fa_max:.2f}]"
            ),
        )

    # --- load gradient info ---
    bvals     = np.loadtxt(str(dti_dir / "dti.bval"))
    bvecs_raw = np.loadtxt(str(dti_dir / "dti.bvec"))

    bval_count = int(bvals.size)
    if bvecs_raw.ndim == 1:
        bvec_count = 1
    elif bvecs_raw.shape[0] == 3:
        bvec_count = int(bvecs_raw.shape[1])   # FSL: 3 × N
    else:
        bvec_count = int(bvecs_raw.shape[0])   # N × 3

    b0_count  = int(np.sum(bvals <= _B0_THRESHOLD))
    dwi_count = int(np.sum(bvals >  _B0_THRESHOLD))

    try:
        gtab = gradient_table(bvals, bvecs_raw)
    except Exception as exc:
        return _failed_record(
            site, dataset, subject_id, f"gradient_table error: {exc}",
            fa_mean=fa_mean, bval_count=bval_count, bvec_count=bvec_count,
            b0_count=b0_count, dwi_count=dwi_count,
        )

    # --- load image data ---
    img      = nib.load(str(dti_dir / "dti_corrected.nii.gz"))
    mask_img = nib.load(str(dti_dir / "qc" / "brain_mask.nii.gz"))
    n_vols   = img.shape[3]

    # --- backend selection ---
    if use_mrtrix is None:
        use_mrtrix = _mrtrix3_available()
    backend = "mrtrix3" if use_mrtrix else "dipy_csa"

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_mrtrix:
        status, warning, error = _run_mrtrix_backend(dti_dir, output_dir, subject_id)
    else:
        status, warning, error = _run_dipy_backend(img, gtab, mask_img, output_dir)

    output_created = status == "success"

    rec = FODPrepRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        fa_mean=fa_mean, backend=backend,
        input_volume_count=n_vols,
        bval_count=bval_count, bvec_count=bvec_count,
        b0_count=b0_count, dwi_count=dwi_count,
        output_created=output_created, status=status,
        warning_message=warning, error_message=error,
    )

    (output_dir / "fod_prep_report.json").write_text(
        json.dumps(rec.to_dict(), indent=2)
    )
    (output_dir / "backend_used.txt").write_text(backend + "\n")

    if status == "success":
        logger.info(
            "[%s/%s] %s complete — peaks saved  FA=%.4f  DWI=%d",
            site, subject_id, backend, fa_mean, dwi_count,
        )
    else:
        logger.error("[%s/%s] failed: %s", site, subject_id, error)

    return rec


# ---------------------------------------------------------------------------
# DIPY backend
# ---------------------------------------------------------------------------

def _run_dipy_backend(
    img: nib.Nifti1Image,
    gtab,
    mask_img: nib.Nifti1Image,
    output_dir: Path,
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Fit CsaOdfModel and extract peak directions.
    Returns (status, warning, error).
    """
    try:
        data     = img.get_fdata(dtype=np.float32)
        mask_arr = np.asarray(mask_img.dataobj).astype(bool)

        model  = CsaOdfModel(gtab, sh_order=_SH_ORDER, smooth=0.006)
        sphere = get_sphere("repulsion724")

        peaks = peaks_from_model(
            model=model,
            data=data,
            sphere=sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=mask_arr,
            return_sh=True,
            npeaks=5,
            normalize_peaks=True,
            parallel=False,
        )

        save_peaks(str(output_dir / "peaks.pam5"), peaks, img.affine)
        return "success", None, None

    except Exception as exc:
        return "failed", None, str(exc)


# ---------------------------------------------------------------------------
# MRtrix3 backend (optional — used only when MRtrix3 is on PATH)
# ---------------------------------------------------------------------------

def _run_mrtrix_backend(
    dti_dir: Path,
    output_dir: Path,
    subject_id: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Run MRtrix3 pipeline: mrconvert → dwi2response → dwi2fod.
    Returns (status, warning, error).
    """
    import subprocess

    log_lines: list[str] = []

    def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
        log_lines.append("$ " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            log_lines.append(result.stdout.strip())
        if result.stderr:
            log_lines.append(result.stderr.strip())
        return result

    mif_path      = output_dir / "dwi.mif"
    response_path = output_dir / "response.txt"
    fod_path      = output_dir / "fod.mif"
    mask_path     = dti_dir / "qc" / "brain_mask.nii.gz"

    try:
        r = _run_cmd([
            "mrconvert",
            str(dti_dir / "dti_corrected.nii.gz"), str(mif_path),
            "-fslgrad", str(dti_dir / "dti.bvec"), str(dti_dir / "dti.bval"),
            "-force",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"mrconvert failed (rc={r.returncode})")

        r = _run_cmd([
            "dwi2response", "tournier",
            str(mif_path), str(response_path),
            "-mask", str(mask_path), "-force",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"dwi2response failed (rc={r.returncode})")

        r = _run_cmd([
            "dwi2fod", "csd",
            str(mif_path), str(response_path), str(fod_path),
            "-mask", str(mask_path), "-force",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"dwi2fod failed (rc={r.returncode})")

        (output_dir / "mrtrix_commands.log").write_text("\n".join(log_lines))
        return "success", None, None

    except Exception as exc:
        (output_dir / "mrtrix_commands.log").write_text("\n".join(log_lines))
        return "failed", None, str(exc)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_fod_preparation_batch(
    processed_root: Path,
    raw_root: Path,
    sites: Optional[list[str]] = None,
    fa_min: float = FA_MIN_CLEAN,
    fa_max: float = FA_MAX_CLEAN,
    use_mrtrix: Optional[bool] = None,
) -> list[FODPrepRecord]:
    """
    Run FOD preparation across all subjects for the given sites.

    Parameters
    ----------
    processed_root :
        Root of processed ABIDE II data (data/processed/abide_ii/).
    raw_root :
        Raw data root — safety guard.
    sites :
        Site folder names (e.g. ["bni", "nyu1"]). Defaults to CLEAN_DTI_SITES.
    """
    guard_no_raw_write(processed_root, raw_root)
    sites = sites if sites is not None else CLEAN_DTI_SITES

    records: list[FODPrepRecord] = []

    for site_name in sites:
        site_dir = processed_root / site_name
        if not site_dir.is_dir():
            logger.warning("[%s] site directory not found: %s", site_name, site_dir)
            continue

        subject_dirs = []
        for dataset_dir in sorted(site_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for subject_dir in sorted(dataset_dir.iterdir()):
                dti_dir = subject_dir / "session_1" / "dti_1"
                if dti_dir.is_dir():
                    subject_dirs.append(dti_dir)

        logger.info("[%s] processing %d subjects …", site_name, len(subject_dirs))

        for dti_dir in subject_dirs:
            rec = prepare_subject_fod(
                dti_dir=dti_dir,
                output_dir=dti_dir / "fod",
                raw_root=raw_root,
                fa_min=fa_min,
                fa_max=fa_max,
                use_mrtrix=use_mrtrix,
            )
            records.append(rec)

        n_ok   = sum(1 for r in records if r.site == site_name and r.status == "success")
        n_skip = sum(1 for r in records if r.site == site_name and r.status == "skipped")
        n_fail = sum(1 for r in records if r.site == site_name and r.status == "failed")
        logger.info(
            "[%s] done — success=%d  skipped=%d  failed=%d",
            site_name, n_ok, n_skip, n_fail,
        )

    return records


# ---------------------------------------------------------------------------
# Summary CSV export
# ---------------------------------------------------------------------------

def save_summary_csvs(
    records: list[FODPrepRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Write phase3_1_fod_preparation_summary.csv and phase3_1_fod_site_summary.csv.

    Returns the two output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject summary
    run_csv = output_dir / "phase3_1_fod_preparation_summary.csv"
    pd.DataFrame([r.to_summary_row() for r in records]).to_csv(run_csv, index=False)

    # Site-level summary
    site_groups: dict[str, list[FODPrepRecord]] = {}
    for r in records:
        site_groups.setdefault(r.site, []).append(r)

    site_rows = []
    for site, recs in site_groups.items():
        success = [r for r in recs if r.status == "success"]
        skipped = [r for r in recs if r.status == "skipped"]
        failed  = [r for r in recs if r.status == "failed"]
        fa_vals  = [r.fa_mean for r in success if r.fa_mean > 0]
        dwi_vals = [r.dwi_count for r in success if r.dwi_count > 0]
        backends = sorted({r.backend for r in success}) or ["none"]
        site_rows.append({
            "site":                       site,
            "total_subjects_considered":  len(recs),
            "subjects_processed":         len(success),
            "subjects_skipped":           len(skipped),
            "subjects_failed":            len(failed),
            "mean_fa":                    round(sum(fa_vals) / len(fa_vals), 4) if fa_vals else None,
            "mean_dwi_count":             round(sum(dwi_vals) / len(dwi_vals), 1) if dwi_vals else None,
            "backend_used":               ",".join(backends),
            "common_b_values":            "",
            "notes":                      "",
        })

    site_csv = output_dir / "phase3_1_fod_site_summary.csv"
    pd.DataFrame(site_rows).to_csv(site_csv, index=False)

    logger.info("Run summary  → %s  (%d rows)", run_csv.name, len(records))
    logger.info("Site summary → %s  (%d sites)", site_csv.name, len(site_rows))
    return run_csv, site_csv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _failed_record(
    site: str,
    dataset: str,
    subject_id: str,
    error: str,
    fa_mean: float = 0.0,
    n_vols: int = 0,
    bval_count: int = 0,
    bvec_count: int = 0,
    b0_count: int = 0,
    dwi_count: int = 0,
) -> FODPrepRecord:
    logger.error("[%s/%s] failed: %s", site, subject_id, error)
    return FODPrepRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        fa_mean=fa_mean, backend="none",
        input_volume_count=n_vols,
        bval_count=bval_count, bvec_count=bvec_count,
        b0_count=b0_count, dwi_count=dwi_count,
        output_created=False, status="failed",
        error_message=error,
    )
