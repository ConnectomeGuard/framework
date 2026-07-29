"""
Phase 4.2 — Bootstrap Uncertainty Estimation

Runs B deterministic tractography iterations per subject with different random
seeds. Computes per-edge CoV, mean, std, and presence fraction to quantify
edge-weight reproducibility.

Scientific rationale:
  High-count edges should show low CoV (stable across seeds).
  Low-count or marginal edges show high CoV (stochastic appearance).
  Gate G-BOOT-SEED validates this: Spearman rho(CoV, mean_weight) MUST be negative.

Parallelism:
  ProcessPoolExecutor runs B bootstrap runs per subject concurrently.
  Caller controls n_workers; recommended = min(B, cpu_count()).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.peaks import load_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask

from neurofiber.connectome.atlas_registration import load_registered_atlas
from neurofiber.connectome.connectome_builder import build_subject_connectome
from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

BOOTSTRAP_SUBDIR = "bootstrap"
N_NODES = 100
N_POSSIBLE_EDGES = 4950


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BootstrapSummary:
    site:       str
    dataset:    str
    subject_id: str

    B:            int
    base_seed:    int
    runs_ok:      int
    runs_failed:  int

    median_cov:   Optional[float]
    mean_cov:     Optional[float]
    spearman_rho: Optional[float]
    n_detected_50pct: int

    status:          str = "success"
    warning_message: Optional[str] = None
    error_message:   Optional[str] = None
    elapsed_sec:     Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_row(self) -> dict:
        d = self.to_dict()
        for k in ("median_cov", "mean_cov", "spearman_rho", "elapsed_sec"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        return d


# ---------------------------------------------------------------------------
# Single bootstrap run (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _run_one_bootstrap(
    dti_dir_str:       str,
    atlas_dir_str:     str,
    labels_csv_str:    str,
    run_idx:           int,
    base_seed:         int,
    seeds_per_subject: int,
    fa_threshold:      float,
    interface_fa_low:  float,
    interface_fa_high: float,
    step_size:         float,
    max_angle:         float,
    max_cross:         int,
    keep_trk:          bool,
) -> dict:
    """
    Run one bootstrap iteration. Returns a dict with run metadata + adj matrix path.
    Designed to be called in a subprocess (ProcessPoolExecutor).
    """
    dti_dir   = Path(dti_dir_str)
    atlas_dir = Path(atlas_dir_str)
    session_dir = dti_dir.parent

    subject_id = dti_dir.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name

    seed = base_seed + run_idx
    run_dir = session_dir / BOOTSTRAP_SUBDIR / f"run_{run_idx:03d}"

    # Skip if already complete
    adj_path = run_dir / "adjacency_streamline_count.npy"
    meta_path = run_dir / "run_meta.json"
    if adj_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "success":
            return {"run_idx": run_idx, "seed": seed, "status": "skipped",
                    "adj_path": str(adj_path)}

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        # --- Load inputs ------------------------------------------------
        fa_path    = dti_dir / "tensor" / "FA.nii.gz"
        mask_path  = dti_dir / "qc"    / "brain_mask.nii.gz"
        peaks_path = dti_dir / "fod"   / "peaks.pam5"

        for p in [fa_path, mask_path, peaks_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing input: {p.name}")

        pam      = load_peaks(str(peaks_path))
        fa_img   = nib.load(str(fa_path))
        mask_img = nib.load(str(mask_path))
        fa       = fa_img.get_fdata(dtype=np.float32)
        mask     = np.asarray(mask_img.dataobj).astype(bool)

        # Interface seed mask
        interface_mask = (fa >= interface_fa_low) & (fa < interface_fa_high) & mask
        if not interface_mask.any():
            raise ValueError(
                f"Interface seed mask empty (FA [{interface_fa_low}, {interface_fa_high}))"
            )

        # --- Tractography -----------------------------------------------
        np.random.seed(seed)
        seeds = random_seeds_from_mask(
            interface_mask, pam.affine,
            seeds_count=seeds_per_subject,
            seed_count_per_voxel=False,
        )

        getter = DeterministicMaximumDirectionGetter.from_shcoeff(
            pam.shm_coeff,
            max_angle=max_angle,
            sphere=pam.sphere,
        )
        stopping = ThresholdStoppingCriterion(fa, fa_threshold)
        streamlines = Streamlines(
            LocalTracking(
                getter, stopping, seeds, pam.affine,
                step_size=step_size,
                max_cross=max_cross,
                return_all=False,
            )
        )

        n_sl = len(streamlines)
        if n_sl == 0:
            raise ValueError("Zero streamlines generated")

        # Save tractogram (temporary — deleted after connectome unless keep_trk)
        trk_path = run_dir / "streamlines.trk"
        sft = StatefulTractogram(streamlines, fa_img, Space.RASMM)
        save_trk(sft, str(trk_path))

        # --- Connectome -------------------------------------------------
        label_df = pd.read_csv(str(labels_csv_str))
        registered_atlas = load_registered_atlas(atlas_dir, label_df)

        rec = build_subject_connectome(
            dti_dir=dti_dir,
            registered_atlas=registered_atlas,
            output_dir=run_dir,
            trk_path=trk_path,
        )

        if rec.status != "success":
            raise RuntimeError(f"Connectome build failed: {rec.error_message}")

        # Keep only the adjacency matrix; remove bulky outputs to save disk
        if not keep_trk:
            trk_path.unlink(missing_ok=True)
        for bulky in ["edge_table.csv", "graph.graphml", "node_table.csv",
                      "streamline_edge_assignment.csv", "streamline_endpoint_table.csv",
                      "edge_aggregation_report.json"]:
            for p in run_dir.rglob(bulky):
                p.unlink(missing_ok=True)
        baseline_dir = run_dir / "baseline_artifacts"
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir, ignore_errors=True)

        elapsed = time.time() - t0
        meta = {
            "run_idx":         run_idx,
            "seed_used":       seed,
            "n_streamlines":   n_sl,
            "n_edges":         rec.edge_count,
            "graph_density":   rec.graph_density,
            "mapping_ratio":   rec.mapping_success_ratio,
            "elapsed_sec":     round(elapsed, 2),
            "status":          "success",
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return {"run_idx": run_idx, "seed": seed, "status": "success",
                "adj_path": str(adj_path), "n_sl": n_sl, "elapsed": elapsed}

    except Exception as exc:
        elapsed = time.time() - t0
        meta = {
            "run_idx":   run_idx,
            "seed_used": seed,
            "status":    "failed",
            "error":     str(exc),
            "elapsed_sec": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.warning("[%s/%s] bootstrap run %03d failed: %s", site, subject_id, run_idx, exc)
        return {"run_idx": run_idx, "seed": seed, "status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Per-subject bootstrap orchestration
# ---------------------------------------------------------------------------

def run_bootstrap_for_subject(
    dti_dir:           Path,
    raw_root:          Path,
    B:                 int   = 20,
    base_seed:         int   = 1000,
    seeds_per_subject: int   = 10_000,
    fa_threshold:      float = 0.10,
    interface_fa_low:  float = 0.08,
    interface_fa_high: float = 0.20,
    step_size:         float = 0.5,
    max_angle:         float = 30.0,
    max_cross:         int   = 1,
    n_workers:         int   = 10,
    keep_trk:          bool  = False,
    skip_if_exists:    bool  = True,
) -> BootstrapSummary:
    """
    Run B bootstrap det-tractography iterations for one subject and aggregate statistics.

    Returns a BootstrapSummary. Outputs are written to:
        {dti_dir.parent}/bootstrap/
    """
    subject_id = dti_dir.parent.parent.name
    dataset    = dti_dir.parent.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name
    session_dir = dti_dir.parent
    guard_no_raw_write(session_dir, raw_root)

    bootstrap_dir = session_dir / BOOTSTRAP_SUBDIR
    summary_path  = bootstrap_dir / "bootstrap_summary.json"

    def _fail(msg: str, elapsed: float = 0.0) -> BootstrapSummary:
        logger.error("[%s/%s] bootstrap failed: %s", site, subject_id, msg)
        return BootstrapSummary(
            site=site, dataset=dataset, subject_id=subject_id,
            B=B, base_seed=base_seed, runs_ok=0, runs_failed=B,
            median_cov=None, mean_cov=None, spearman_rho=None,
            n_detected_50pct=0,
            status="failed", error_message=msg, elapsed_sec=elapsed,
        )

    # Skip if fully complete
    if skip_if_exists and summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            if s.get("status") == "success" and s.get("runs_ok", 0) == B:
                logger.info("[%s/%s] bootstrap already complete — skipping", site, subject_id)
                return _load_summary(s, site, dataset, subject_id)
        except Exception:
            pass

    # Check atlas
    atlas_dir  = session_dir / "connectome" / "atlas"
    labels_csv = atlas_dir / "atlas_labels.csv"
    if not atlas_dir.exists() or not (atlas_dir / "atlas_subject_space.nii.gz").exists():
        return _fail("atlas_subject_space.nii.gz not found — run Phase 3.4 first")

    t0 = time.time()
    logger.info("[%s/%s] starting bootstrap B=%d  workers=%d", site, subject_id, B, n_workers)

    # Launch all B runs in parallel
    futures_args = [
        dict(
            dti_dir_str=str(dti_dir),
            atlas_dir_str=str(atlas_dir),
            labels_csv_str=str(labels_csv),
            run_idx=b,
            base_seed=base_seed,
            seeds_per_subject=seeds_per_subject,
            fa_threshold=fa_threshold,
            interface_fa_low=interface_fa_low,
            interface_fa_high=interface_fa_high,
            step_size=step_size,
            max_angle=max_angle,
            max_cross=max_cross,
            keep_trk=keep_trk,
        )
        for b in range(B)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one_bootstrap, **kw): kw["run_idx"] for kw in futures_args}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res["status"] in ("success", "skipped"):
                logger.debug(
                    "[%s/%s] run %03d %s (%.1fs)",
                    site, subject_id, res["run_idx"],
                    res["status"],
                    res.get("elapsed", 0),
                )

    runs_ok     = sum(1 for r in results if r["status"] in ("success", "skipped"))
    runs_failed = sum(1 for r in results if r["status"] == "failed")

    if runs_ok == 0:
        return _fail("All bootstrap runs failed", elapsed=time.time() - t0)

    # --- Aggregate statistics -------------------------------------------
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    adjs = []
    for b in range(B):
        adj_path = bootstrap_dir / f"run_{b:03d}" / "adjacency_streamline_count.npy"
        if adj_path.exists():
            A = np.load(str(adj_path)).astype(float)
            A = np.maximum(A, A.T)  # enforce symmetry
            adjs.append(A)
        else:
            adjs.append(np.zeros((N_NODES, N_NODES), dtype=float))

    stack = np.stack(adjs, axis=0)  # (B, 100, 100)

    mean_W     = stack.mean(axis=0)
    std_W      = stack.std(axis=0)
    presence_W = (stack > 0).mean(axis=0)
    cov_W      = std_W / (mean_W + 1e-6)

    np.save(str(bootstrap_dir / "bootstrap_mean.npy"),     mean_W)
    np.save(str(bootstrap_dir / "bootstrap_std.npy"),      std_W)
    np.save(str(bootstrap_dir / "bootstrap_cov.npy"),      cov_W)
    np.save(str(bootstrap_dir / "bootstrap_presence.npy"), presence_W)

    # Scalar QC metrics on upper triangle of detected edges
    upper = np.triu(np.ones((N_NODES, N_NODES), dtype=bool), k=1)
    detected = upper & (mean_W > 0)

    cov_flat  = cov_W[detected]
    mean_flat = mean_W[detected]

    median_cov = float(np.median(cov_flat)) if len(cov_flat) > 0 else None
    mean_cov   = float(np.mean(cov_flat))   if len(cov_flat) > 0 else None

    spearman_rho = None
    if len(cov_flat) >= 5:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(cov_flat, mean_flat)
        spearman_rho = float(rho)

    n_detected_50pct = int((presence_W[upper] >= 0.50).sum())

    elapsed = time.time() - t0

    # Warnings
    warnings = []
    if runs_failed > 0:
        warnings.append(f"{runs_failed}/{B} runs failed")
    if median_cov is not None and not (0.05 <= median_cov <= 1.50):
        warnings.append(f"G-BOOT-COV: median_cov={median_cov:.3f} outside [0.05, 1.50]")
    if spearman_rho is not None and spearman_rho >= 0:
        warnings.append(
            f"G-BOOT-SEED FAILED: rho={spearman_rho:.3f} >= 0 — bootstrap broken"
        )

    rec = BootstrapSummary(
        site=site, dataset=dataset, subject_id=subject_id,
        B=B, base_seed=base_seed,
        runs_ok=runs_ok, runs_failed=runs_failed,
        median_cov=median_cov, mean_cov=mean_cov,
        spearman_rho=spearman_rho,
        n_detected_50pct=n_detected_50pct,
        warning_message="; ".join(warnings) if warnings else None,
        elapsed_sec=round(elapsed, 1),
    )
    summary_path.write_text(json.dumps(rec.to_dict(), indent=2))

    logger.info(
        "[%s/%s] bootstrap done  runs_ok=%d/%d  median_cov=%.3f  rho=%.3f  %.1fs",
        site, subject_id, runs_ok, B,
        median_cov or 0.0, spearman_rho or 0.0, elapsed,
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_bootstrap_batch(
    processed_root:    Path,
    raw_root:          Path,
    sites:             list[str],
    B:                 int   = 20,
    base_seed:         int   = 1000,
    seeds_per_subject: int   = 10_000,
    fa_threshold:      float = 0.10,
    interface_fa_low:  float = 0.08,
    interface_fa_high: float = 0.20,
    step_size:         float = 0.5,
    max_angle:         float = 30.0,
    max_cross:         int   = 1,
    n_workers:         int   = 10,
    n_subjects_parallel: int = 1,
    keep_trk:          bool  = False,
    skip_if_exists:    bool  = True,
) -> list[BootstrapSummary]:
    """
    Run bootstrap for all subjects across the given sites.

    n_subjects_parallel > 1: process multiple subjects at once.
    Each subject uses n_workers internal processes.
    Total processes = n_subjects_parallel × n_workers.
    """
    from neurofiber.tractography.streamline_generation import site_to_folder

    guard_no_raw_write(processed_root, raw_root)
    records: list[BootstrapSummary] = []

    all_dti_dirs = []
    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = processed_root / folder
        if not site_dir.exists():
            logger.warning("[%s] site directory not found", folder)
            continue
        dti_dirs = sorted(site_dir.glob("*/*/session_1/dti_1"))
        if not dti_dirs:
            logger.warning("[%s] no dti_1 dirs found", folder)
            continue
        logger.info("[%s] %d subjects queued for bootstrap", folder, len(dti_dirs))
        all_dti_dirs.extend(dti_dirs)

    logger.info("Bootstrap: %d total subjects  B=%d  workers=%d/subject",
                len(all_dti_dirs), B, n_workers)

    kwargs = dict(
        raw_root=raw_root, B=B, base_seed=base_seed,
        seeds_per_subject=seeds_per_subject,
        fa_threshold=fa_threshold,
        interface_fa_low=interface_fa_low,
        interface_fa_high=interface_fa_high,
        step_size=step_size, max_angle=max_angle, max_cross=max_cross,
        n_workers=n_workers, keep_trk=keep_trk, skip_if_exists=skip_if_exists,
    )

    if n_subjects_parallel == 1:
        for i, dti_dir in enumerate(all_dti_dirs):
            subj = dti_dir.parent.parent.name
            site = dti_dir.parent.parent.parent.parent.name
            logger.info("[%s/%s] subject %d/%d", site, subj, i + 1, len(all_dti_dirs))
            rec = run_bootstrap_for_subject(dti_dir=dti_dir, **kwargs)
            records.append(rec)
    else:
        # Subjects processed in serial batches; parallelism is intra-subject
        # (ProcessPoolExecutor inside run_bootstrap_for_subject already handles that)
        # This outer loop handles n_subjects_parallel > 1 via sequential processing
        # with each subject's internal pool — avoids nested ProcessPools.
        for i, dti_dir in enumerate(all_dti_dirs):
            subj = dti_dir.parent.parent.name
            site = dti_dir.parent.parent.parent.parent.name
            logger.info("[%s/%s] subject %d/%d", site, subj, i + 1, len(all_dti_dirs))
            rec = run_bootstrap_for_subject(dti_dir=dti_dir, **kwargs)
            records.append(rec)

    n_ok   = sum(1 for r in records if r.status == "success")
    n_fail = sum(1 for r in records if r.status == "failed")
    logger.info("Bootstrap batch done: %d success / %d failed", n_ok, n_fail)
    return records


# ---------------------------------------------------------------------------
# Cohort gate: G-BOOT-SEED
# ---------------------------------------------------------------------------

def check_gate_boot_seed(processed_root: Path) -> float:
    """
    G-BOOT-SEED: cohort-level check that Spearman rho(CoV, mean_weight) < 0.
    Returns the cohort mean rho. Raises AssertionError if >= 0.
    """
    from scipy.stats import spearmanr

    upper = np.triu(np.ones((N_NODES, N_NODES), dtype=bool), k=1)
    rhos = []

    for cov_path in sorted(processed_root.glob("*/*/*/session_1/bootstrap/bootstrap_cov.npy")):
        mean_path = cov_path.parent / "bootstrap_mean.npy"
        if not mean_path.exists():
            continue
        cov_W  = np.load(str(cov_path))
        mean_W = np.load(str(mean_path))
        detected = upper & (mean_W > 0)
        if detected.sum() < 5:
            continue
        rho, _ = spearmanr(cov_W[detected], mean_W[detected])
        rhos.append(float(rho))

    if not rhos:
        raise AssertionError("G-BOOT-SEED: no bootstrap results found")

    cohort_rho = float(np.mean(rhos))
    assert cohort_rho < 0, (
        f"G-BOOT-SEED FAILED: mean Spearman rho(CoV, streamlines) = {cohort_rho:.3f}. "
        "MUST be negative. Bootstrap implementation is broken."
    )
    logger.info("G-BOOT-SEED PASS: cohort rho = %.3f (negative as expected)", cohort_rho)
    return cohort_rho


# ---------------------------------------------------------------------------
# CSV summary helper
# ---------------------------------------------------------------------------

def save_bootstrap_summary_csv(
    records:    list[BootstrapSummary],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_summary_row() for r in records])
    csv_path = output_dir / "phase42_bootstrap_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Bootstrap summary CSV → %s  (%d rows)", csv_path.name, len(df))
    return csv_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_summary(s: dict, site: str, dataset: str, subject_id: str) -> BootstrapSummary:
    return BootstrapSummary(
        site=site, dataset=dataset, subject_id=subject_id,
        B=s.get("B", 0), base_seed=s.get("base_seed", 0),
        runs_ok=s.get("runs_ok", 0), runs_failed=s.get("runs_failed", 0),
        median_cov=s.get("median_cov"), mean_cov=s.get("mean_cov"),
        spearman_rho=s.get("spearman_rho"),
        n_detected_50pct=s.get("n_detected_50pct", 0),
        status=s.get("status", "success"),
        warning_message=s.get("warning_message"),
        elapsed_sec=s.get("elapsed_sec"),
        timestamp=s.get("timestamp", ""),
    )
