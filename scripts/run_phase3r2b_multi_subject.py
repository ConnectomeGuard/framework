"""
NeuroFiber Phase 3R.2B — Multi-Subject Seed Saturation Validation

Goal: Determine whether the BNI 29006 saturation curve generalises across sites.

Subjects: 2 per site × 4 sites = 8 subjects
  BNI:    29006, 29007
  NYU_1:  29177, 29180
  SDSU_1: 28853, 28854
  TCD_1:  29096, 29097

Seed counts: 10k, 50k, 100k, 250k, 500k

Per subject × seed count:
  - Regenerate peaks.pam5 if missing (CSD, deterministic)
  - Run tractography
  - Build Schaefer-100 connectome
  - Compute: nonzero edges, density, Jaccard vs 500k, edge recovery, runtime

Outputs:
  data/experiments/seed_saturation_multi/
    {site}_{subject_id}/
      {seed_count}/
        streamlines.trk
        run_report.json
        connectome_metrics.json
    seed_saturation_multi_summary.csv
    seed_saturation_multi_report.md
    plots/
      pooled_edges.png
      pooled_density.png
      pooled_recovery.png
      pooled_runtime.png
      site_saturation.png

Resume: already-computed runs are skipped automatically (check connectome_metrics.json).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
import pandas as pd

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import DeterministicMaximumDirectionGetter, peaks_from_model
from dipy.io.peaks import load_peaks, save_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_trk
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask

# ── constants ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_V2B = ROOT / "data/processed_v2b/abide_ii"
V1_PROCESSED  = ROOT / "data/processed/abide_ii"
SINGLE_STUDY  = ROOT / "data/processed_v2b/phase3r2b_seed_density"   # reuse 29006
OUT_ROOT      = ROOT / "data/experiments/seed_saturation_multi"

SEED_COUNTS   = [10_000, 50_000, 100_000, 250_000, 500_000]
REFERENCE_SC  = 500_000
RECOVERY_THR  = 0.95
N_ROIS        = 100

FA_THRESHOLD  = 0.15
MIN_LENGTH_MM = 20.0
STEP_SIZE     = 0.5
MAX_ANGLE     = 30.0
MAX_CROSS     = 1
SH_ORDER      = 6
B0_THRESHOLD  = 50
RANDOM_SEED   = 42

SUBJECTS = [
    {"site": "BNI",    "folder": "bni",  "dataset": "ABIDEII-BNI_1",    "subject_id": "29006"},
    {"site": "BNI",    "folder": "bni",  "dataset": "ABIDEII-BNI_1",    "subject_id": "29007"},
    {"site": "NYU_1",  "folder": "nyu1", "dataset": "ABIDEII-NYU_1",    "subject_id": "29177"},
    {"site": "NYU_1",  "folder": "nyu1", "dataset": "ABIDEII-NYU_1",    "subject_id": "29180"},
    {"site": "SDSU_1", "folder": "sdsu", "dataset": "ABIDEII-SDSU_1",   "subject_id": "28853"},
    {"site": "SDSU_1", "folder": "sdsu", "dataset": "ABIDEII-SDSU_1",   "subject_id": "28854"},
    {"site": "TCD_1",  "folder": "tcd",  "dataset": "ABIDEII-TCD_1",    "subject_id": "29096"},
    {"site": "TCD_1",  "folder": "tcd",  "dataset": "ABIDEII-TCD_1",    "subject_id": "29097"},
]

PLOT_STYLE = {
    "figure.facecolor": "#0e0e12",
    "axes.facecolor":   "#0e0e12",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#cccccc",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "lines.linewidth":  2.0,
}

SITE_COLORS = {
    "BNI":    "#4db8ff",
    "NYU_1":  "#ff9f40",
    "SDSU_1": "#7dde92",
    "TCD_1":  "#e87dde",
}

SITE_MARKERS = {"BNI": "o", "NYU_1": "s", "SDSU_1": "^", "TCD_1": "D"}


# ── path helpers ──────────────────────────────────────────────────────────────

def dti_dir(s: dict) -> Path:
    return (PROCESSED_V2B / s["folder"] / s["dataset"]
            / s["subject_id"] / "session_1" / "dti_1")


def atlas_path(s: dict) -> Path:
    return (V1_PROCESSED / s["folder"] / s["dataset"]
            / s["subject_id"] / "session_1/connectome/atlas/atlas_subject_space.nii.gz")


def subject_out_dir(s: dict) -> Path:
    return OUT_ROOT / f"{s['site']}_{s['subject_id']}"


# ── peaks regeneration ────────────────────────────────────────────────────────

def ensure_peaks(s: dict) -> Path:
    dti      = dti_dir(s)
    peaks_p  = dti / "fod" / "peaks.pam5"

    if peaks_p.exists():
        return peaks_p

    sid = s["subject_id"]
    print(f"    [peaks] regenerating for {sid} ...", flush=True)
    t0 = time.time()

    dwi_path  = dti / "dwi_preprocessed.nii.gz"
    bval_path = dti / "dwi_preprocessed.bval"
    bvec_path = dti / "dwi_preprocessed.bvec"
    mask_path = dti / "qc" / "brain_mask.nii.gz"

    dwi_img  = nib.load(str(dwi_path))
    mask_img = nib.load(str(mask_path))
    dwi_data = dwi_img.get_fdata(dtype=np.float32)
    mask     = np.asarray(mask_img.dataobj).astype(bool)

    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    gtab   = gradient_table(bvals, bvecs, b0_threshold=B0_THRESHOLD)
    sphere = get_sphere("repulsion724")
    model  = CsaOdfModel(gtab, sh_order=SH_ORDER)
    pam    = peaks_from_model(
        model, dwi_data, sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask,
        return_sh=True,
        return_odf=False,
        npeaks=5,
        normalize_peaks=True,
        parallel=False,
    )

    peaks_p.parent.mkdir(parents=True, exist_ok=True)
    save_peaks(str(peaks_p), pam, affine=dwi_img.affine)
    print(f"    [peaks] done in {time.time()-t0:.0f}s")
    return peaks_p


# ── tractography ──────────────────────────────────────────────────────────────

def run_tractography(s: dict, pam, sc: int, out_trk: Path) -> dict:
    dti = dti_dir(s)
    t0  = time.time()

    fa_img   = nib.load(str(dti / "tensor" / "FA.nii.gz"))
    mask_img = nib.load(str(dti / "qc" / "brain_mask.nii.gz"))
    fa       = fa_img.get_fdata(dtype=np.float32)
    mask     = np.asarray(mask_img.dataobj).astype(bool)
    wm_mask  = (fa >= FA_THRESHOLD) & mask

    np.random.seed(RANDOM_SEED)
    seeds = random_seeds_from_mask(wm_mask, pam.affine,
                                   seeds_count=sc, seed_count_per_voxel=False)

    getter   = DeterministicMaximumDirectionGetter.from_shcoeff(
        pam.shm_coeff, max_angle=MAX_ANGLE, sphere=pam.sphere)
    stopping = ThresholdStoppingCriterion(fa, FA_THRESHOLD)

    raw = Streamlines(LocalTracking(
        getter, stopping, seeds, pam.affine,
        step_size=STEP_SIZE, max_cross=MAX_CROSS, return_all=False,
    ))

    lengths  = np.array([len(sl) * STEP_SIZE for sl in raw])
    keep     = lengths >= MIN_LENGTH_MM
    retained = Streamlines(sl for sl, k in zip(raw, keep) if k)
    ret_lens = lengths[keep]

    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram(retained, fa_img, Space.RASMM)
    save_trk(sft, str(out_trk))

    rep = {
        "site":           s["site"],
        "subject_id":     s["subject_id"],
        "seed_count":     sc,
        "raw_streamlines": len(raw),
        "retained":       len(retained),
        "rejected_short": int((~keep).sum()),
        "mean_length":    float(np.mean(ret_lens)) if len(ret_lens) else 0.0,
        "runtime_s":      round(time.time() - t0, 2),
    }
    (out_trk.parent / "run_report.json").write_text(json.dumps(rep, indent=2))
    return rep


# ── connectome ────────────────────────────────────────────────────────────────

def build_connectome(trk_path: Path, atlas_data: np.ndarray,
                     atlas_affine: np.ndarray) -> dict:
    sft = load_tractogram(str(trk_path), str(trk_path), to_space=Space.RASMM)
    streamlines = list(sft.streamlines)

    inv = np.linalg.inv(atlas_affine)
    mat = np.zeros((N_ROIS + 1, N_ROIS + 1), dtype=np.int32)
    used = disc = 0

    for sl in streamlines:
        ep  = np.array([sl[0], sl[-1]])
        vc  = np.round(nib.affines.apply_affine(inv, ep)).astype(int)
        lbs = []
        for v in vc:
            lbs.append(int(atlas_data[v[0], v[1], v[2]])
                       if all(0 <= v[d] < atlas_data.shape[d] for d in range(3))
                       else 0)
        i, j = lbs
        if i > 0 and j > 0 and i != j:
            mat[i, j] += 1; mat[j, i] += 1; used += 1
        else:
            disc += 1

    upper     = np.triu(mat[1:, 1:], k=1)
    max_edges = N_ROIS * (N_ROIS - 1) // 2
    nonzero   = int(np.sum(upper > 0))
    density   = nonzero / max_edges
    wts       = upper[upper > 0]
    edge_set  = frozenset(zip(*np.where(upper > 0))) if nonzero else frozenset()

    return {
        "nonzero_edges": nonzero,
        "density":       round(density, 6),
        "mean_weight":   float(np.mean(wts)) if len(wts) else 0.0,
        "edge_set":      edge_set,
        "used":          used,
        "discarded":     disc,
    }


def edge_similarity(edge_set: frozenset, ref_set: frozenset) -> dict:
    inter = edge_set & ref_set
    union = edge_set | ref_set
    return {
        "jaccard":           round(len(inter) / len(union), 4) if union else 1.0,
        "edge_recovery_pct": round(len(inter) / len(ref_set) * 100, 2) if ref_set else 100.0,
        "active_overlap":    len(inter),
    }


# ── per-subject pipeline ──────────────────────────────────────────────────────

def process_subject(s: dict) -> list[dict]:
    sid     = s["subject_id"]
    site    = s["site"]
    out_dir = subject_out_dir(s)
    atl_p   = atlas_path(s)

    print(f"\n  ── {site} {sid} ──────────────────────────────────────────")

    atlas_img  = nib.load(str(atl_p))
    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)

    # collect all edge sets first pass (to compute vs-500k reference)
    runs: dict[int, dict] = {}

    for sc in SEED_COUNTS:
        sc_dir    = out_dir / str(sc)
        trk_path  = sc_dir / "streamlines.trk"
        rep_path  = sc_dir / "run_report.json"
        cm_path   = sc_dir / "connectome_metrics.json"

        # check if we can reuse from the single-subject study (29006)
        if not trk_path.exists() and sid == "29006":
            src = SINGLE_STUDY / sid / str(sc) / "streamlines.trk"
            if src.exists():
                sc_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(str(src), str(trk_path))
                src_rep = SINGLE_STUDY / sid / str(sc) / "run_report.json"
                if src_rep.exists():
                    shutil.copy2(str(src_rep), str(rep_path))

        # resume: skip if connectome already computed
        if cm_path.exists() and trk_path.exists() and rep_path.exists():
            with open(rep_path) as f:
                rep = json.load(f)
            with open(cm_path) as f:
                cm_raw = json.load(f)
            # restore edge_set from stored edge list
            edge_set = frozenset(tuple(e) for e in cm_raw.get("edge_list", []))
            runs[sc] = {**rep, **cm_raw, "edge_set": edge_set}
            print(f"    [{sc:>8,}]  resume — edges={cm_raw['nonzero_edges']}", flush=True)
            continue

        # peaks.pam5
        if not hasattr(process_subject, "_pam_cache"):
            process_subject._pam_cache = {}
        cache_key = str(dti_dir(s) / "fod" / "peaks.pam5")
        if cache_key not in process_subject._pam_cache:
            pp = ensure_peaks(s)
            process_subject._pam_cache[cache_key] = load_peaks(str(pp))
        pam = process_subject._pam_cache[cache_key]

        # tractography
        print(f"    [{sc:>8,}]  tracking ...", end="  ", flush=True)
        rep = run_tractography(s, pam, sc, trk_path)
        print(f"retained={rep['retained']:,}  t={rep['runtime_s']:.0f}s", flush=True)

        # connectome
        cm = build_connectome(trk_path, atlas_data, atlas_img.affine)

        # save connectome metrics (edge_set as list for JSON)
        cm_save = {k: v for k, v in cm.items() if k != "edge_set"}
        cm_save["edge_list"] = [[int(i), int(j)] for i, j in cm["edge_set"]]
        sc_dir.mkdir(parents=True, exist_ok=True)
        cm_path.write_text(json.dumps(cm_save, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x))

        runs[sc] = {**rep, **cm}
        print(f"           edges={cm['nonzero_edges']}  density={cm['density']:.4f}")

    # compute vs-500k similarity
    ref_set = runs[REFERENCE_SC]["edge_set"]
    rows = []
    for sc in SEED_COUNTS:
        r   = runs[sc]
        sim = edge_similarity(r["edge_set"], ref_set)
        rows.append({
            "site":              site,
            "subject_id":        sid,
            "seed_count":        sc,
            "retained":          r["retained"],
            "runtime_s":         r["runtime_s"],
            "mean_length_mm":    round(r["mean_length"], 1),
            "nonzero_edges":     r["nonzero_edges"],
            "density":           r["density"],
            "mean_weight":       r["mean_weight"],
            **sim,
        })
    return rows


# ── plots ─────────────────────────────────────────────────────────────────────

def _fmt_k(x, _): return f"{x:.0f}k"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _mean_band(ax, df: pd.DataFrame, col: str, seeds_k, color: str,
               label: str = "Pooled mean") -> None:
    pooled = df.groupby("seed_count")[col].agg(["mean", "std"]).reset_index()
    sk = pooled["seed_count"] / 1000
    ax.plot(sk, pooled["mean"], "o-", color=color, linewidth=2.5,
            markersize=8, label=label, zorder=5)
    ax.fill_between(sk,
                    pooled["mean"] - pooled["std"],
                    pooled["mean"] + pooled["std"],
                    color=color, alpha=0.15)


def plot_pooled_edges(df: pd.DataFrame) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        for site, grp in df.groupby("site"):
            for sid, sgrp in grp.groupby("subject_id"):
                sk = sgrp["seed_count"] / 1000
                ax.plot(sk, sgrp["nonzero_edges"],
                        color=SITE_COLORS[site], alpha=0.35,
                        marker=SITE_MARKERS[site], markersize=4, linewidth=1)
        # site means
        for site, grp in df.groupby("site"):
            mn = grp.groupby("seed_count")["nonzero_edges"].mean().reset_index()
            ax.plot(mn["seed_count"] / 1000, mn["nonzero_edges"], "o-",
                    color=SITE_COLORS[site], linewidth=2.2, markersize=7,
                    label=site)
        # pooled mean
        _mean_band(ax, df, "nonzero_edges", None, "#ffffff", "Pooled mean ± SD")
        ax.set_title("Seed Count vs Nonzero Edges — 4 Sites × 2 Subjects", color="white", fontsize=13)
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Nonzero Edges")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
        ax.grid(True)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        _save(fig, OUT_ROOT / "plots/pooled_edges.png")
    print("  [plot] pooled edges")


def plot_pooled_density(df: pd.DataFrame) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        for site, grp in df.groupby("site"):
            for sid, sgrp in grp.groupby("subject_id"):
                ax.plot(sgrp["seed_count"] / 1000, sgrp["density"] * 100,
                        color=SITE_COLORS[site], alpha=0.35,
                        marker=SITE_MARKERS[site], markersize=4, linewidth=1)
        for site, grp in df.groupby("site"):
            mn = grp.groupby("seed_count")["density"].mean().reset_index()
            ax.plot(mn["seed_count"] / 1000, mn["density"] * 100, "o-",
                    color=SITE_COLORS[site], linewidth=2.2, markersize=7, label=site)
        _mean_band(ax, df, "density", None, "#ffffff", "Pooled mean ± SD")
        ax.set_title("Seed Count vs Connectome Density — 4 Sites × 2 Subjects", color="white", fontsize=13)
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Connectome Density (%)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
        ax.grid(True)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        _save(fig, OUT_ROOT / "plots/pooled_density.png")
    print("  [plot] pooled density")


def plot_pooled_recovery(df: pd.DataFrame) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        for site, grp in df.groupby("site"):
            for sid, sgrp in grp.groupby("subject_id"):
                ax.plot(sgrp["seed_count"] / 1000, sgrp["edge_recovery_pct"],
                        color=SITE_COLORS[site], alpha=0.35,
                        marker=SITE_MARKERS[site], markersize=4, linewidth=1)
        for site, grp in df.groupby("site"):
            mn = grp.groupby("seed_count")["edge_recovery_pct"].mean().reset_index()
            ax.plot(mn["seed_count"] / 1000, mn["edge_recovery_pct"], "o-",
                    color=SITE_COLORS[site], linewidth=2.2, markersize=7, label=site)
        _mean_band(ax, df, "edge_recovery_pct", None, "#ffffff", "Pooled mean ± SD")
        ax.axhline(RECOVERY_THR * 100, color="#ffffff", linestyle=":",
                   alpha=0.5, linewidth=1.2, label=f"{int(RECOVERY_THR*100)}% threshold")
        ax.set_ylim(0, 105)
        ax.set_title("Edge Recovery vs 500k Reference — 4 Sites × 2 Subjects", color="white", fontsize=13)
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("% of 500k Edges Recovered")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
        ax.grid(True)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        _save(fig, OUT_ROOT / "plots/pooled_recovery.png")
    print("  [plot] pooled recovery")


def plot_pooled_runtime(df: pd.DataFrame) -> None:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        for site, grp in df.groupby("site"):
            mn = grp.groupby("seed_count")["runtime_s"].mean().reset_index()
            ax.plot(mn["seed_count"] / 1000, mn["runtime_s"] / 60, "o-",
                    color=SITE_COLORS[site], linewidth=2.2, markersize=7, label=site)
        _mean_band(ax, df.assign(runtime_min=df["runtime_s"] / 60),
                   "runtime_min", None, "#ffffff", "Pooled mean ± SD")
        ax.set_title("Seed Count vs Runtime — 4 Sites × 2 Subjects", color="white", fontsize=13)
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Runtime (minutes)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
        ax.grid(True)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        _save(fig, OUT_ROOT / "plots/pooled_runtime.png")
    print("  [plot] pooled runtime")


def plot_site_saturation(df: pd.DataFrame) -> None:
    """2×2 grid: one panel per site, each showing both subjects + site mean."""
    sites = ["BNI", "NYU_1", "SDSU_1", "TCD_1"]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Site-Wise Edge Recovery vs 500k Reference",
                     color="white", fontsize=14, y=0.98)
        for ax, site in zip(axes.flat, sites):
            grp   = df[df["site"] == site]
            color = SITE_COLORS[site]
            for sid, sgrp in grp.groupby("subject_id"):
                ax.plot(sgrp["seed_count"] / 1000, sgrp["edge_recovery_pct"],
                        "o--", color=color, alpha=0.5, markersize=5,
                        label=f"subj {sid}")
            mn = grp.groupby("seed_count")["edge_recovery_pct"].mean().reset_index()
            ax.plot(mn["seed_count"] / 1000, mn["edge_recovery_pct"], "o-",
                    color=color, linewidth=2.5, markersize=8, label="Site mean")
            ax.axhline(RECOVERY_THR * 100, color="#ffffff", linestyle=":",
                       alpha=0.45, linewidth=1.2)
            ax.set_title(site, color=color, fontsize=12)
            ax.set_xlabel("Seed Count (thousands)")
            ax.set_ylabel("% 500k Edges Recovered")
            ax.set_ylim(0, 105)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
            ax.grid(True)
            ax.legend(facecolor="#1a1a22", edgecolor="#333333",
                      labelcolor="#cccccc", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, OUT_ROOT / "plots/site_saturation.png")
    print("  [plot] site saturation")


# ── markdown report ───────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, decision_sc: Optional[int]) -> None:
    pooled = (df.groupby("seed_count")[
        ["retained", "runtime_s", "nonzero_edges", "density",
         "edge_recovery_pct", "jaccard"]
    ].mean().reset_index())

    lines = [
        "# NeuroFiber Phase 3R.2B — Multi-Subject Seed Saturation Validation",
        "",
        f"**Subjects:** 8 (2 per site × 4 sites)  ",
        f"**Sites:** BNI, NYU_1, SDSU_1, TCD_1  ",
        f"**Seed counts:** {', '.join(str(s) for s in SEED_COUNTS)}  ",
        f"**Reference:** {REFERENCE_SC:,} seeds  ",
        f"**Decision threshold:** mean edge recovery ≥ {int(RECOVERY_THR*100)}% across subjects  ",
        f"**1M seeds:** excluded — computationally impractical on current hardware  ",
        "",
        "---",
        "",
        "## Pooled Results (mean across 8 subjects)",
        "",
        "| Seeds | Retained | Runtime | Nonzero Edges | Density | Recovery vs 500k | Jaccard |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in pooled.iterrows():
        lines.append(
            f"| {int(r['seed_count']):,} "
            f"| {r['retained']:,.0f} "
            f"| {r['runtime_s']/60:.1f} min "
            f"| {r['nonzero_edges']:.0f} "
            f"| {r['density']*100:.3f}% "
            f"| {r['edge_recovery_pct']:.1f}% "
            f"| {r['jaccard']:.4f} |"
        )

    lines += ["", "---", "", "## Per-Subject Results", ""]
    for site in ["BNI", "NYU_1", "SDSU_1", "TCD_1"]:
        lines.append(f"### {site}")
        lines += [
            "",
            "| Subject | Seeds | Retained | Runtime | Edges | Density | Recovery |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in df[df["site"] == site].iterrows():
            lines.append(
                f"| {r['subject_id']} "
                f"| {int(r['seed_count']):,} "
                f"| {int(r['retained']):,} "
                f"| {r['runtime_s']/60:.1f} min "
                f"| {int(r['nonzero_edges'])} "
                f"| {r['density']*100:.3f}% "
                f"| {r['edge_recovery_pct']:.1f}% |"
            )
        lines.append("")

    lines += ["---", "", "## Decision", ""]
    if decision_sc:
        dec = pooled[pooled["seed_count"] == decision_sc].iloc[0]
        ref = pooled[pooled["seed_count"] == REFERENCE_SC].iloc[0]
        speedup = ref["runtime_s"] / dec["runtime_s"]
        lines += [
            f"**Optimal seed count: {decision_sc:,}** (smallest with mean recovery ≥ {int(RECOVERY_THR*100)}%)  ",
            f"- Mean edge recovery: **{dec['edge_recovery_pct']:.1f}%**  ",
            f"- Mean Jaccard: **{dec['jaccard']:.4f}**  ",
            f"- Mean nonzero edges: **{dec['nonzero_edges']:.0f}** "
            f"(vs {ref['nonzero_edges']:.0f} at 500k)  ",
            f"- Mean runtime: **{dec['runtime_s']/60:.1f} min** "
            f"(vs {ref['runtime_s']/60:.1f} min at 500k)  ",
            f"- Speedup vs 500k: **{speedup:.1f}×**  ",
            f"- 229-subject batch estimate: "
            f"**{dec['runtime_s']/60*229/60:.1f} hours** "
            f"(vs {ref['runtime_s']/60*229/60:.1f} hours at 500k)  ",
        ]
    else:
        lines += [
            "**No seed count meets the ≥95% mean recovery threshold within the tested range.**  ",
            "500k seeds is the best available option within tested range.  ",
        ]

    lines += [
        "",
        "---",
        "",
        "## Notes on 1M Seeds",
        "",
        "A 1M seed run on BNI 29006 stalled after ~116 minutes without completing. "
        "Estimated 229-subject batch cost: ~444 hours (single-core). Excluded.",
    ]

    out = OUT_ROOT / "seed_saturation_multi_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  [md]   → {out.relative_to(ROOT)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("NeuroFiber — Phase 3R.2B Multi-Subject Seed Saturation")
    print("=" * 62)
    print(f"  Subjects : {len(SUBJECTS)} ({', '.join(s['site'] for s in SUBJECTS)})")
    print(f"  Seeds    : {SEED_COUNTS}")
    print(f"  Output   : {OUT_ROOT.relative_to(ROOT)}")
    print()

    all_rows: list[dict] = []

    for s in SUBJECTS:
        rows = process_subject(s)
        all_rows.extend(rows)
        # save intermediate CSV after each subject
        df_interim = pd.DataFrame(all_rows)
        csv_out = OUT_ROOT / "seed_saturation_multi_summary.csv"
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df_interim.to_csv(str(csv_out), index=False)

    df = pd.DataFrame(all_rows).sort_values(
        ["site", "subject_id", "seed_count"]).reset_index(drop=True)

    # ── decision ──────────────────────────────────────────────────────────
    pooled_recovery = df.groupby("seed_count")["edge_recovery_pct"].mean()
    candidates = pooled_recovery[pooled_recovery >= RECOVERY_THR * 100]
    decision_sc = int(candidates.index.min()) if not candidates.empty else None

    print()
    print("Pooled mean edge recovery vs 500k:")
    for sc, val in pooled_recovery.items():
        flag = " ← OPTIMAL" if sc == decision_sc else ""
        print(f"  {int(sc):>8,} seeds:  {val:.1f}%{flag}")
    print()

    if decision_sc:
        print(f"  DECISION: optimal seed count = {decision_sc:,}")
    else:
        print("  DECISION: no seed count meets >=95% threshold — 500k recommended")

    # ── save final CSV ────────────────────────────────────────────────────
    csv_out = OUT_ROOT / "seed_saturation_multi_summary.csv"
    df.to_csv(str(csv_out), index=False)
    print(f"\n  [csv]  → {csv_out.relative_to(ROOT)}")

    # ── plots ─────────────────────────────────────────────────────────────
    print()
    plot_pooled_edges(df)
    plot_pooled_density(df)
    plot_pooled_recovery(df)
    plot_pooled_runtime(df)
    plot_site_saturation(df)

    # ── report ────────────────────────────────────────────────────────────
    write_report(df, decision_sc)

    print()
    print("=" * 62)
    print("Done.")
    print(f"  CSV    : {csv_out.relative_to(ROOT)}")
    print(f"  Report : {(OUT_ROOT / 'seed_saturation_multi_report.md').relative_to(ROOT)}")
    print(f"  Plots  : {(OUT_ROOT / 'plots').relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
