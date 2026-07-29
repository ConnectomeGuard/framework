"""
Phase 2R.1 — Standard DWI Preprocessing Foundation

Pipeline order (per subject):
  Step 0  raw input validation
  Step 1  site-specific volume correction
  Step 2  MP-PCA denoising (MRtrix3 dwidenoise | DIPY patch2self)
  Step 3  Gibbs ringing correction (MRtrix3 mrdegibbs | skip_with_warning)
  Step 4  eddy-current / motion correction (MRtrix3 dwifslpreproc | skip_with_warning)
  Step 5  bias-field correction (MRtrix3 dwibiascorrect | skip_with_warning)
  Step 6  export final NIfTI + bval/bvec + report

Invariants:
  - Never writes to data/raw (guard_no_raw_write enforced)
  - Every skipped step is explicitly recorded in JSON and CSV
  - Outputs land in data/processed_v2 only
  - Old data/processed is never touched
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

PIPELINE_VERSION = "2R.1"
PIPELINE_VERSION_NO_DENOISE = "2R.1b"
OUTPUT_SUBDIR = "processed_v2"

# Reason string embedded in every report when patch2self is blocked
_PATCH2SELF_DISABLED_REASON = "directional_signal_variance_collapse_detected"

# Site display names → raw folder names
_SITE_FOLDER_MAP = {
    "BNI":    "bni",
    "IP_1":   "ip",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
    # aliases
    "IP":     "ip",
    "SDSU":   "sdsu",
    "TCD":    "tcd",
    "NYU1":   "nyu1",
    "NYU2":   "nyu2",
}


def site_to_folder(site_display: str) -> str:
    return _SITE_FOLDER_MAP.get(site_display, site_display.lower().replace("_", ""))


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

@dataclass
class BackendAvailability:
    mrtrix_version:    Optional[str]   = None
    fsl_version:       Optional[str]   = None
    ants_available:    bool            = False

    @property
    def has_mrtrix(self) -> bool:
        return self.mrtrix_version is not None

    @property
    def has_fsl(self) -> bool:
        return self.fsl_version is not None

    def denoise_backend(self, preference: str, fallback: str = "dipy_patch2self") -> str:
        if preference == "none":
            return "none"
        if preference == "mrtrix_dwidenoise" and self.has_mrtrix:
            return "mrtrix_dwidenoise"
        return fallback

    def gibbs_backend(self, preference: str, fallback: str) -> str:
        if preference == "mrtrix_mrdegibbs" and self.has_mrtrix:
            return "mrtrix_mrdegibbs"
        return fallback  # "skip_with_warning"

    def eddy_backend(self, preference: str, fallback: str) -> str:
        if preference == "mrtrix_dwifslpreproc" and self.has_mrtrix and self.has_fsl:
            return "mrtrix_dwifslpreproc"
        return fallback

    def bias_backend(self, preference: str, fallback: str) -> str:
        if preference == "mrtrix_dwibiascorrect":
            if self.has_mrtrix and (self.ants_available or self.has_fsl):
                return "mrtrix_dwibiascorrect"
        return fallback


def detect_backends() -> BackendAvailability:
    av = BackendAvailability()

    try:
        r = subprocess.run(["mrinfo", "--version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            av.mrtrix_version = r.stdout.strip().splitlines()[0]
    except Exception:
        pass

    try:
        r = subprocess.run(["flirt", "-version"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            av.fsl_version = r.stdout.strip().splitlines()[0]
    except Exception:
        pass

    try:
        r = subprocess.run(["antsRegistration", "--version"],
                           capture_output=True, text=True, timeout=10)
        av.ants_available = r.returncode == 0
    except Exception:
        pass

    logger.info(
        "Backends: mrtrix=%s  fsl=%s  ants=%s",
        av.mrtrix_version or "NOT FOUND",
        av.fsl_version    or "NOT FOUND",
        "yes" if av.ants_available else "no",
    )
    return av


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubjectPreprocessingReport:
    site:       str
    dataset:    str
    subject_id: str

    pipeline_version: str = PIPELINE_VERSION

    # Volume counts
    original_volume_count:  int = 0
    corrected_volume_count: int = 0
    bval_count:             int = 0
    bvec_count:             int = 0
    b0_count:               int = 0
    dwi_count:              int = 0
    unique_bvals:           list = field(default_factory=list)

    # Backend choices
    denoise_backend:              str = "not_run"
    gibbs_backend:                str = "not_run"
    eddy_backend:                 str = "not_run"
    bias_backend:                 str = "not_run"
    patch2self_disabled_reason:   Optional[str] = None

    # Step outcomes
    steps_completed: list = field(default_factory=list)
    steps_skipped:   list = field(default_factory=list)
    warning_count:   int  = 0
    warnings:        list = field(default_factory=list)

    # Final
    status:          str           = "success"
    error_message:   Optional[str] = None
    elapsed_sec:     Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        self.warning_count += 1
        logger.warning("[%s/%s] %s", self.site, self.subject_id, msg)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        d = self.to_dict()
        d["unique_bvals"]    = json.dumps(d["unique_bvals"])
        d["steps_completed"] = "|".join(d["steps_completed"])
        d["steps_skipped"]   = "|".join(d["steps_skipped"])
        d["warnings"]        = "; ".join(d["warnings"]) if d["warnings"] else ""
        if d.get("elapsed_sec") is not None:
            d["elapsed_sec"] = round(d["elapsed_sec"], 1)
        return d


# ---------------------------------------------------------------------------
# Step 0 — Raw validation
# ---------------------------------------------------------------------------

def _step0_validate(
    dti_path:  Path,
    bval_path: Path,
    bvec_path: Path,
) -> dict:
    """Validate raw DWI volume/bval/bvec counts and report mismatch pattern."""
    for p in [dti_path, bval_path, bvec_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    img   = nib.load(str(dti_path))
    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path))

    if bvals.ndim == 0:
        bvals = bvals.reshape(1)
    if bvecs.ndim == 1:
        bvecs = bvecs.reshape(-1, 1)
    # bvecs can be (3, N) or (N, 3) — normalise to (N, 3)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    n_vols  = img.shape[3] if img.ndim == 4 else 1
    n_bvals = len(bvals)
    n_bvecs = bvecs.shape[0]

    b0_mask    = bvals < 50
    b0_count   = int(b0_mask.sum())
    dwi_count  = int((~b0_mask).sum())
    unique_b   = sorted(set(np.round(bvals, -2).astype(int).tolist()))

    mismatch = "none"
    if n_vols != n_bvals:
        mismatch = f"volumes={n_vols}_bvals={n_bvals}"
    if n_bvals != n_bvecs:
        mismatch = f"bvals={n_bvals}_bvecs={n_bvecs}"

    return {
        "original_volume_count": n_vols,
        "bval_count":            n_bvals,
        "bvec_count":            n_bvecs,
        "b0_count":              b0_count,
        "dwi_count":             dwi_count,
        "unique_bvals":          unique_b,
        "mismatch_pattern":      mismatch,
        "img":                   img,
        "bvals":                 bvals,
        "bvecs":                 bvecs,
    }


# ---------------------------------------------------------------------------
# Step 1 — Volume correction
# ---------------------------------------------------------------------------

def _step1_volume_correction(
    img:    "nib.Nifti1Image",
    bvals:  np.ndarray,
    bvecs:  np.ndarray,
    signal_ratio_threshold: float = 0.2,
) -> dict:
    """
    Apply site-specific volume correction rule.

    Rule: if volume_count == bval_count + 1 and bval_count == bvec_count,
    compute ratio = mean(last_volume) / mean(volume_0).
    If ratio < threshold: strip trailing volume.
    Else: mark manual_review.
    """
    n_vols  = img.shape[3]
    n_bvals = len(bvals)
    n_bvecs = bvecs.shape[0]

    if n_vols == n_bvals and n_bvals == n_bvecs:
        # Already consistent — no correction needed
        return {
            "action":                  "no_correction_needed",
            "corrected_volume_count":  n_vols,
            "bvals_corrected":         bvals,
            "bvecs_corrected":         bvecs,
            "img_data":                img.get_fdata(dtype=np.float32),
            "affine":                  img.affine,
            "header":                  img.header,
        }

    if n_vols == n_bvals + 1 and n_bvals == n_bvecs:
        data = img.get_fdata(dtype=np.float32)
        vol0_mean  = float(data[..., 0].mean())
        last_mean  = float(data[..., -1].mean())
        ratio      = last_mean / (vol0_mean + 1e-9)

        if ratio < signal_ratio_threshold:
            return {
                "action":                 "trailing_volume_stripped",
                "signal_ratio":           round(ratio, 4),
                "corrected_volume_count": n_vols - 1,
                "bvals_corrected":        bvals,
                "bvecs_corrected":        bvecs,
                "img_data":               data[..., :-1],
                "affine":                 img.affine,
                "header":                 img.header,
            }
        else:
            raise ValueError(
                f"Volume mismatch (n_vols={n_vols}, n_bvals={n_bvals}) but "
                f"signal_ratio={ratio:.3f} >= {signal_ratio_threshold} — "
                "requires manual review; cannot auto-correct."
            )

    raise ValueError(
        f"Unhandled volume/bval/bvec mismatch: "
        f"n_vols={n_vols}, n_bvals={n_bvals}, n_bvecs={n_bvecs}"
    )


# ---------------------------------------------------------------------------
# Step 2 — Denoising
# ---------------------------------------------------------------------------

def _step2_denoise_patch2self(
    data:   np.ndarray,
    bvals:  np.ndarray,
    affine: np.ndarray,
    header,
    work_dir: Path,
) -> dict:
    """DIPY Patch2Self denoising (fallback when MRtrix3 unavailable)."""
    from dipy.denoise.patch2self import patch2self

    t0 = time.time()
    denoised = patch2self(data, bvals, model="ols", shift_intensity=True,
                          clip_negative_vals=False, b0_threshold=50)
    elapsed = time.time() - t0

    noise_estimate = data.astype(float) - denoised.astype(float)
    noise_std = float(noise_estimate.std())

    return {
        "backend":    "dipy_patch2self",
        "elapsed_sec": round(elapsed, 1),
        "noise_std_estimate": round(noise_std, 4),
        "data":       denoised.astype(np.float32),
        "affine":     affine,
        "header":     header,
    }


def _step2_denoise_mrtrix(
    input_nii: Path,
    bval_path: Path,
    bvec_path: Path,
    work_dir:  Path,
) -> dict:
    """MRtrix3 dwidenoise denoising (preferred when available)."""
    mif_in      = work_dir / "dwi_in.mif"
    mif_denoised = work_dir / "dwi_denoised.mif"
    noise_mif   = work_dir / "noise.mif"
    noise_nii   = work_dir / "noise_map.nii.gz"

    _run(["mrconvert", str(input_nii), str(mif_in),
          "-fslgrad", str(bvec_path), str(bval_path), "-force"])
    t0 = time.time()
    _run(["dwidenoise", str(mif_in), str(mif_denoised),
          "-noise", str(noise_mif), "-force"])
    elapsed = time.time() - t0
    _run(["mrconvert", str(noise_mif), str(noise_nii), "-force"])

    # Read back denoised data
    out_nii = work_dir / "dwi_denoised.nii.gz"
    _run(["mrconvert", str(mif_denoised), str(out_nii),
          "-export_grad_fsl",
          str(work_dir / "dwi_denoised.bvec"),
          str(work_dir / "dwi_denoised.bval"),
          "-force"])

    img = nib.load(str(out_nii))
    return {
        "backend":    "mrtrix_dwidenoise",
        "elapsed_sec": round(elapsed, 1),
        "noise_map_path": str(noise_nii),
        "mif_denoised": str(mif_denoised),
        "bval_out": str(work_dir / "dwi_denoised.bval"),
        "bvec_out": str(work_dir / "dwi_denoised.bvec"),
        "data":   img.get_fdata(dtype=np.float32),
        "affine": img.affine,
        "header": img.header,
    }


# ---------------------------------------------------------------------------
# Step 3 — Gibbs ringing correction
# ---------------------------------------------------------------------------

def _step3_gibbs_mrtrix(mif_denoised: str, work_dir: Path) -> dict:
    """MRtrix3 mrdegibbs."""
    mif_out = work_dir / "dwi_degibbs.mif"
    t0 = time.time()
    _run(["mrdegibbs", mif_denoised, str(mif_out), "-force"])
    return {
        "backend":     "mrtrix_mrdegibbs",
        "elapsed_sec": round(time.time() - t0, 1),
        "mif_out":     str(mif_out),
    }


# ---------------------------------------------------------------------------
# Step 4 — Eddy-current / motion correction
# ---------------------------------------------------------------------------

def _step4_eddy_mrtrix(
    mif_in:   str,
    work_dir: Path,
    pe_dir:   Optional[str],
) -> dict:
    """MRtrix3 dwifslpreproc eddy-current + motion correction."""
    if pe_dir is None:
        raise ValueError(
            "phase-encoding direction unknown — cannot run dwifslpreproc safely. "
            "Set pe_dir in config or mark as limited_eddy."
        )
    mif_out = work_dir / "dwi_eddy.mif"
    t0 = time.time()
    _run([
        "dwifslpreproc", mif_in, str(mif_out),
        "-rpe_none", "-pe_dir", pe_dir, "-force",
    ])
    return {
        "backend":     "mrtrix_dwifslpreproc",
        "pe_dir":      pe_dir,
        "elapsed_sec": round(time.time() - t0, 1),
        "mif_out":     str(mif_out),
    }


# ---------------------------------------------------------------------------
# Step 5 — Bias field correction
# ---------------------------------------------------------------------------

def _step5_bias_mrtrix(
    mif_in:  str,
    work_dir: Path,
    backend_tool: str = "ants",
) -> dict:
    """MRtrix3 dwibiascorrect ants|fsl."""
    mif_out = work_dir / "dwi_bias.mif"
    t0 = time.time()
    _run(["dwibiascorrect", backend_tool, mif_in, str(mif_out), "-force"])
    return {
        "backend":     f"mrtrix_dwibiascorrect_{backend_tool}",
        "elapsed_sec": round(time.time() - t0, 1),
        "mif_out":     str(mif_out),
    }


# ---------------------------------------------------------------------------
# Step 6 — Export final outputs
# ---------------------------------------------------------------------------

def _step6_export_from_array(
    data:    np.ndarray,
    affine:  np.ndarray,
    header,
    bvals:   np.ndarray,
    bvecs:   np.ndarray,
    out_dir: Path,
) -> dict:
    """Save final NIfTI + bval/bvec from in-memory arrays (DIPY path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nii_out  = out_dir / "dwi_preprocessed.nii.gz"
    bval_out = out_dir / "dwi_preprocessed.bval"
    bvec_out = out_dir / "dwi_preprocessed.bvec"

    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, str(nii_out))
    np.savetxt(str(bval_out), bvals[np.newaxis, :], fmt="%g")
    # bvecs: save as (3, N) — FSL convention
    if bvecs.shape[1] == 3:
        bvecs_out = bvecs.T
    else:
        bvecs_out = bvecs
    np.savetxt(str(bvec_out), bvecs_out, fmt="%.6f")

    return {
        "nii_path":  str(nii_out),
        "bval_path": str(bval_out),
        "bvec_path": str(bvec_out),
        "shape":     list(data.shape),
    }


def _step6_export_from_mif(
    mif_in:  str,
    out_dir: Path,
) -> dict:
    """Convert final .mif to NIfTI + bval/bvec (MRtrix3 path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nii_out  = out_dir / "dwi_preprocessed.nii.gz"
    bvec_out = out_dir / "dwi_preprocessed.bvec"
    bval_out = out_dir / "dwi_preprocessed.bval"

    _run([
        "mrconvert", mif_in, str(nii_out),
        "-export_grad_fsl", str(bvec_out), str(bval_out),
        "-force",
    ])
    img = nib.load(str(nii_out))
    return {
        "nii_path":  str(nii_out),
        "bval_path": str(bval_out),
        "bvec_path": str(bvec_out),
        "shape":     list(img.shape),
    }


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------

def run_subject_pipeline(
    dti_path:   Path,
    bval_path:  Path,
    bvec_path:  Path,
    output_dir: Path,
    raw_root:   Path,
    backends:   BackendAvailability,
    config:     dict,
    skip_if_exists: bool = True,
) -> SubjectPreprocessingReport:
    """
    Run the full Phase 2R.1 pipeline for one subject.

    Outputs → output_dir/
        dwi_preprocessed.nii.gz
        dwi_preprocessed.bval
        dwi_preprocessed.bvec
        preprocessing_report.json
        qc/
            noise_map.nii.gz   (if dwidenoise used)
            qc_metrics.json
    """
    subject_id = dti_path.parent.parent.parent.name
    dataset    = dti_path.parent.parent.parent.parent.name
    site       = dti_path.parent.parent.parent.parent.parent.name

    guard_no_raw_write(output_dir, raw_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "preprocessing_report.json"
    rec = SubjectPreprocessingReport(
        site=site, dataset=dataset, subject_id=subject_id,
    )

    # Skip if already complete
    if skip_if_exists and report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
            if existing.get("status") == "success":
                logger.info("[%s/%s] already preprocessed — skipping", site, subject_id)
                return _load_report(existing)
        except Exception:
            pass

    t0 = time.time()
    qc_dir   = output_dir / "qc"
    work_dir = output_dir / "_preproc_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # Step 0 — Validate
        # ------------------------------------------------------------------
        v = _step0_validate(dti_path, bval_path, bvec_path)
        rec.original_volume_count = v["original_volume_count"]
        rec.bval_count   = v["bval_count"]
        rec.bvec_count   = v["bvec_count"]
        rec.b0_count     = v["b0_count"]
        rec.dwi_count    = v["dwi_count"]
        rec.unique_bvals = v["unique_bvals"]
        img   = v["img"]
        bvals = v["bvals"]
        bvecs = v["bvecs"]
        rec.steps_completed.append("step0_validation")

        if v["mismatch_pattern"] != "none":
            rec.add_warning(f"Volume mismatch detected: {v['mismatch_pattern']}")

        # ------------------------------------------------------------------
        # Step 1 — Volume correction
        # ------------------------------------------------------------------
        corr = _step1_volume_correction(img, bvals, bvecs)
        rec.corrected_volume_count = corr["corrected_volume_count"]
        bvals = corr["bvals_corrected"]
        bvecs = corr["bvecs_corrected"]

        if corr["action"] == "trailing_volume_stripped":
            rec.add_warning(
                f"Trailing low-signal volume stripped "
                f"(signal_ratio={corr.get('signal_ratio', '?')})"
            )
        rec.steps_completed.append(f"step1_volume_correction:{corr['action']}")

        # After correction, data lives either in img or as corrected array
        data   = corr["img_data"]
        affine = corr["affine"]
        header = corr["header"]

        # ------------------------------------------------------------------
        # Step 2 — Denoise
        # ------------------------------------------------------------------
        pref     = config["backend_preference"]["denoise"]
        fallback = config["fallbacks"].get("denoise", "dipy_patch2self")
        allow_patch2self_experimental = config.get("allow_patch2self_experimental", False)
        backend  = backends.denoise_backend(pref, fallback)
        rec.denoise_backend = backend

        # Safety guard: block patch2self unless explicitly opted in
        if backend == "dipy_patch2self" and not allow_patch2self_experimental:
            raise RuntimeError(
                "PATCH2SELF BLOCKED: dipy_patch2self collapsed directional DWI signal "
                "variance in ABIDE-II ABIDE-II DTI data (BNI FA delta ≈ -0.45, SDSU ≈ -0.54). "
                "Set allow_patch2self_experimental=true in config to override, "
                "or set denoise_backend: none to skip denoising entirely."
            )

        den = {}
        if backend == "mrtrix_dwidenoise":
            tmp_nii = work_dir / "corrected.nii.gz"
            nib.save(nib.Nifti1Image(data, affine, header), str(tmp_nii))
            tmp_bval = work_dir / "corrected.bval"
            tmp_bvec = work_dir / "corrected.bvec"
            np.savetxt(str(tmp_bval), bvals[np.newaxis, :], fmt="%g")
            np.savetxt(str(tmp_bvec), (bvecs.T if bvecs.shape[1] == 3 else bvecs), fmt="%.6f")
            den = _step2_denoise_mrtrix(tmp_nii, tmp_bval, tmp_bvec, work_dir)
            data   = den["data"]
            affine = den["affine"]
            header = den["header"]
            if den.get("noise_map_path"):
                qc_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(den["noise_map_path"], qc_dir / "noise_map.nii.gz")
            rec.steps_completed.append(f"step2_denoise:{backend}")
        elif backend == "dipy_patch2self":
            # Only reachable when allow_patch2self_experimental=true
            den = _step2_denoise_patch2self(data, bvals, affine, header, work_dir)
            data   = den["data"]
            affine = den["affine"]
            header = den["header"]
            rec.steps_completed.append(f"step2_denoise:{backend}")
        else:
            # backend == "none" — skip denoising entirely
            rec.steps_skipped.append("step2_denoise")
            rec.patch2self_disabled_reason = _PATCH2SELF_DISABLED_REASON
            rec.add_warning(
                "Denoising SKIPPED (denoise_backend=none). "
                "Patch2Self disabled: directional signal variance collapse detected in ABIDE-II DTI. "
                "Re-enable with MRtrix3 dwidenoise (MP-PCA) when available."
            )

        logger.info(
            "[%s/%s] denoise=%s  %.1fs",
            site, subject_id, backend, den.get("elapsed_sec", 0)
        )

        # ------------------------------------------------------------------
        # Step 3 — Gibbs ringing correction
        # ------------------------------------------------------------------
        gibbs_pref = config["backend_preference"]["gibbs"]
        gibbs_fall = config["fallbacks"]["gibbs"]
        gibbs_back = backends.gibbs_backend(gibbs_pref, gibbs_fall)
        rec.gibbs_backend = gibbs_back

        if gibbs_back == "mrtrix_mrdegibbs":
            gib = _step3_gibbs_mrtrix(den.get("mif_denoised", ""), work_dir)
            rec.steps_completed.append(f"step3_gibbs:{gibbs_back}")
        else:
            rec.steps_skipped.append("step3_gibbs")
            rec.add_warning(
                f"Gibbs ringing correction SKIPPED — "
                f"MRtrix3 not available (backend={gibbs_back})"
            )
            gib = {}

        # ------------------------------------------------------------------
        # Step 4 — Eddy / motion correction
        # ------------------------------------------------------------------
        eddy_pref = config["backend_preference"]["eddy"]
        eddy_fall = config["fallbacks"]["eddy"]
        eddy_back = backends.eddy_backend(eddy_pref, eddy_fall)
        rec.eddy_backend = eddy_back

        pe_cfg   = config.get("phase_encoding", {})
        pe_dir   = pe_cfg.get("default_pe_dir") or None
        require_manual_pe = pe_cfg.get("require_manual_pe_dir_for_eddy", True)

        if eddy_back == "mrtrix_dwifslpreproc":
            if require_manual_pe and pe_dir is None:
                eddy_back = "skip_with_warning"
                rec.eddy_backend = eddy_back
                rec.add_warning(
                    "Eddy correction SKIPPED — phase-encoding direction unknown. "
                    "Set phase_encoding.default_pe_dir in config to enable."
                )

        if eddy_back == "mrtrix_dwifslpreproc":
            mif_for_eddy = gib.get("mif_out") or den.get("mif_denoised") or ""
            edd = _step4_eddy_mrtrix(mif_for_eddy, work_dir, pe_dir)
            rec.steps_completed.append(f"step4_eddy:{eddy_back}")
        else:
            rec.steps_skipped.append("step4_eddy")
            rec.add_warning(
                f"Eddy-current/motion correction SKIPPED — "
                f"backend={eddy_back} (MRtrix3 or FSL not available, "
                f"or phase-encoding direction unknown)."
            )

        # ------------------------------------------------------------------
        # Step 5 — Bias field correction
        # ------------------------------------------------------------------
        bias_pref = config["backend_preference"]["bias"]
        bias_fall = config["fallbacks"]["bias"]
        bias_back = backends.bias_backend(bias_pref, bias_fall)
        rec.bias_backend = bias_back

        if bias_back.startswith("mrtrix_dwibiascorrect"):
            tool = "ants" if backends.ants_available else "fsl"
            mif_for_bias = (
                locals().get("edd", {}).get("mif_out")
                or gib.get("mif_out")
                or den.get("mif_denoised")
                or ""
            )
            _step5_bias_mrtrix(mif_for_bias, work_dir, backend_tool=tool)
            rec.steps_completed.append(f"step5_bias:{bias_back}_{tool}")
        else:
            rec.steps_skipped.append("step5_bias")
            if bias_fall == "skip_with_warning":
                rec.add_warning(
                    "Bias-field correction SKIPPED — "
                    "MRtrix3 and/or ANTs/FSL not available."
                )

        # ------------------------------------------------------------------
        # Step 6 — Export final outputs
        # ------------------------------------------------------------------
        # Use MRtrix3 path if a final .mif exists, else array path
        final_mif = None
        if bias_back.startswith("mrtrix_dwibiascorrect"):
            final_mif = str(work_dir / "dwi_bias.mif")
        elif eddy_back == "mrtrix_dwifslpreproc":
            final_mif = str(work_dir / "dwi_eddy.mif")
        elif gibbs_back == "mrtrix_mrdegibbs":
            final_mif = gib.get("mif_out")

        if final_mif and Path(final_mif).exists():
            exp = _step6_export_from_mif(final_mif, output_dir)
        else:
            exp = _step6_export_from_array(data, affine, header, bvals, bvecs, output_dir)

        rec.steps_completed.append("step6_export")

        # QC metrics JSON
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_metrics = {
            "subject_id":            subject_id,
            "original_volume_count": rec.original_volume_count,
            "corrected_volume_count": rec.corrected_volume_count,
            "b0_count":              rec.b0_count,
            "dwi_count":             rec.dwi_count,
            "unique_bvals":          rec.unique_bvals,
            "denoise_backend":       rec.denoise_backend,
            "gibbs_backend":         rec.gibbs_backend,
            "eddy_backend":          rec.eddy_backend,
            "bias_backend":          rec.bias_backend,
            "steps_completed":       rec.steps_completed,
            "steps_skipped":         rec.steps_skipped,
            "warning_count":         rec.warning_count,
            "warnings":              rec.warnings,
            "output_shape":          exp["shape"],
        }
        (qc_dir / "qc_metrics.json").write_text(json.dumps(qc_metrics, indent=2))

    except Exception as exc:
        rec.status        = "failed"
        rec.error_message = str(exc)
        logger.error("[%s/%s] pipeline failed: %s", site, subject_id, exc)

    finally:
        # Always clean up tmp dir
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    rec.elapsed_sec = round(time.time() - t0, 1)
    report_path.write_text(json.dumps(rec.to_dict(), indent=2))

    if rec.status == "success":
        logger.info(
            "[%s/%s] done  steps=%s  skipped=%s  warnings=%d  %.1fs",
            site, subject_id,
            ",".join(rec.steps_completed),
            ",".join(rec.steps_skipped) or "none",
            rec.warning_count, rec.elapsed_sec,
        )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch_pipeline(
    raw_root:       Path,
    output_root:    Path,
    sites:          list[str],
    ip_qc_only:     bool = True,
    skip_if_exists: bool = True,
    config:         dict = None,
) -> list[SubjectPreprocessingReport]:
    """
    Run Phase 2R.1 preprocessing for all subjects in the given sites.

    IP_1 note: processed and QC-labelled but NOT automatically included in
    clean tractography. ip_qc_only=True records a QC warning for every IP subject.
    """
    guard_no_raw_write(output_root, raw_root)
    backends = detect_backends()
    records: list[SubjectPreprocessingReport] = []

    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = raw_root / folder
        if not site_dir.exists():
            logger.warning("Site directory not found: %s", site_dir)
            continue

        dwi_files = sorted(site_dir.rglob("dti.nii.gz"))
        if not dwi_files:
            logger.warning("[%s] no dti.nii.gz found", folder)
            continue

        is_ip = folder == "ip"
        logger.info("[%s] %d subjects  ip_qc_only=%s", folder, len(dwi_files), is_ip)

        for dti_path in dwi_files:
            subj_raw_dir = dti_path.parent
            subject_id   = subj_raw_dir.parent.parent.name
            dataset      = subj_raw_dir.parent.parent.parent.name

            bval_path = subj_raw_dir / "dti.bval"
            bvec_path = subj_raw_dir / "dti.bvec"

            out_dir = (
                output_root / "abide_ii" / folder / dataset
                / subject_id / "session_1" / "dti_1"
            )

            try:
                rec = run_subject_pipeline(
                    dti_path=dti_path,
                    bval_path=bval_path,
                    bvec_path=bvec_path,
                    output_dir=out_dir,
                    raw_root=raw_root,
                    backends=backends,
                    config=config or {},
                    skip_if_exists=skip_if_exists,
                )
            except Exception as exc:
                rec = SubjectPreprocessingReport(
                    site=folder, dataset=dataset, subject_id=subject_id,
                    status="failed", error_message=str(exc),
                )
                logger.error("[%s/%s] unexpected error: %s", folder, subject_id, exc)

            if is_ip and ip_qc_only:
                rec.add_warning(
                    "IP_1 site: QC-labelled only — "
                    "NOT automatically included in clean tractography pipeline."
                )

            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# CSV summary helpers
# ---------------------------------------------------------------------------

def save_summary_csvs(
    records:     list[SubjectPreprocessingReport],
    output_root: Path,
) -> tuple[Path, Path]:
    import pandas as pd

    output_root.mkdir(parents=True, exist_ok=True)
    rows = [r.to_summary_row() for r in records]
    df   = pd.DataFrame(rows)

    subj_csv = output_root / "phase2r_1_standard_dwi_preprocessing_summary.csv"
    df.to_csv(subj_csv, index=False)

    ok   = df[df["status"] == "success"]
    site_rows = []
    for site, grp in ok.groupby("site"):
        site_rows.append({
            "site":              site,
            "n_success":         len(grp),
            "n_failed":          int((df[df["site"] == site]["status"] != "success").sum()),
            "n_warnings":        int(grp["warning_count"].sum()),
            "denoise_backends":  "|".join(grp["denoise_backend"].unique()),
            "gibbs_skipped":     int((grp["gibbs_backend"] != "mrtrix_mrdegibbs").sum()),
            "eddy_skipped":      int((grp["eddy_backend"] != "mrtrix_dwifslpreproc").sum()),
            "bias_skipped":      int((grp["bias_backend"].str.startswith("mrtrix_dwibiascorrect") == False).sum()),
            "mean_elapsed_sec":  round(grp["elapsed_sec"].mean(), 1),
        })
    site_csv = output_root / "phase2r_1_site_summary.csv"
    pd.DataFrame(site_rows).to_csv(site_csv, index=False)

    logger.info("Summaries: %s  %s", subj_csv.name, site_csv.name)
    return subj_csv, site_csv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {r.returncode}): {' '.join(cmd)}\n"
            f"stderr: {r.stderr[-500:]}"
        )
    return r


def _load_report(d: dict) -> SubjectPreprocessingReport:
    return SubjectPreprocessingReport(
        site=d.get("site", ""), dataset=d.get("dataset", ""),
        subject_id=d.get("subject_id", ""),
        pipeline_version=d.get("pipeline_version", PIPELINE_VERSION),
        original_volume_count=d.get("original_volume_count", 0),
        corrected_volume_count=d.get("corrected_volume_count", 0),
        bval_count=d.get("bval_count", 0), bvec_count=d.get("bvec_count", 0),
        b0_count=d.get("b0_count", 0), dwi_count=d.get("dwi_count", 0),
        unique_bvals=d.get("unique_bvals", []),
        denoise_backend=d.get("denoise_backend", ""),
        gibbs_backend=d.get("gibbs_backend", ""),
        eddy_backend=d.get("eddy_backend", ""),
        bias_backend=d.get("bias_backend", ""),
        steps_completed=d.get("steps_completed", []),
        steps_skipped=d.get("steps_skipped", []),
        warning_count=d.get("warning_count", 0),
        warnings=d.get("warnings", []),
        status=d.get("status", "success"),
        error_message=d.get("error_message"),
        elapsed_sec=d.get("elapsed_sec"),
        timestamp=d.get("timestamp", ""),
    )
