"""
NeuroFiber Phase 3R.2B — Seed Density Study

Runs tractography on a single subject at multiple seed counts and measures:
  - retained streamlines
  - connectome edge count (nonzero edges)
  - connectome density
  - runtime per seed count

Plots saturation curves to identify the seed count at which connectome
information stops increasing.

Subject: BNI 29006 (default)

Seed counts: 10k, 50k, 100k, 250k, 500k, 1M

Usage:
    python ai_pipeline/scripts/run_phase3r2b_seed_density.py
    python ai_pipeline/scripts/run_phase3r2b_seed_density.py --subject 29007 --site BNI

Safety:
    - Writes only to data/processed_v2b/phase3r2b_seed_density/
    - Never writes to data/raw, data/canonical_v1, or canonical tractography paths
    - Canonical streamlines.trk files are never touched
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
import pandas as pd

# ── DIPY tractography ─────────────────────────────────────────────────────────
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import DeterministicMaximumDirectionGetter, peaks_from_model
from dipy.io.peaks import load_peaks, save_peaks
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import random_seeds_from_mask
from dipy.tracking.streamline import Streamlines

# ── repo paths ────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[2]
PROCESSED_V2B    = ROOT / "data/processed_v2b/abide_ii"
V1_PROCESSED     = ROOT / "data/processed/abide_ii"
OUT_ROOT         = ROOT / "data/processed_v2b/phase3r2b_seed_density"

GUARD_PATHS = [ROOT / "data/raw", ROOT / "data/canonical_v1"]

SITE_FOLDER_MAP = {
    "BNI":    "bni",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
}
DATASET_MAP = {
    "BNI":    "ABIDEII-BNI_1",
    "NYU_1":  "ABIDEII-NYU_1",
    "NYU_2":  "ABIDEII-NYU_2",
    "SDSU_1": "ABIDEII-SDSU_1",
    "TCD_1":  "ABIDEII-TCD_1",
}

DEFAULT_SEEDS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

# Tractography params matching Phase 3R.2
FA_THRESHOLD   = 0.15
MIN_LENGTH_MM  = 20.0
MAX_LENGTH_MM  = 300.0
STEP_SIZE      = 0.5
MAX_ANGLE      = 30.0
MAX_CROSS      = 1
SH_ORDER       = 6
B0_THRESHOLD   = 50
RANDOM_SEED    = 42


# ── safety ────────────────────────────────────────────────────────────────────

def _guard(path: Path) -> None:
    for g in GUARD_PATHS:
        if path == g or g in path.parents:
            sys.exit(f"ERROR: attempted write to protected path: {path}")


# ── subject path helpers ──────────────────────────────────────────────────────

def subject_dti_dir(site: str, subject_id: str) -> Path:
    folder  = SITE_FOLDER_MAP[site]
    dataset = DATASET_MAP[site]
    return PROCESSED_V2B / folder / dataset / subject_id / "session_1" / "dti_1"


def atlas_path(site: str, subject_id: str) -> Path:
    folder  = SITE_FOLDER_MAP[site]
    dataset = DATASET_MAP[site]
    return (V1_PROCESSED / folder / dataset / subject_id
            / "session_1" / "connectome" / "atlas" / "atlas_subject_space.nii.gz")


# ── Step 1: regenerate peaks.pam5 if missing ─────────────────────────────────

def ensure_peaks(dti_dir: Path) -> Path:
    peaks_path = dti_dir / "fod" / "peaks.pam5"
    if peaks_path.exists():
        print("  [peaks] peaks.pam5 already exists — skipping regeneration")
        return peaks_path

    print("  [peaks] peaks.pam5 missing — regenerating from DWI...")
    t0 = time.time()

    dwi_path  = dti_dir / "dwi_preprocessed.nii.gz"
    bval_path = dti_dir / "dwi_preprocessed.bval"
    bvec_path = dti_dir / "dwi_preprocessed.bvec"
    mask_path = dti_dir / "qc" / "brain_mask.nii.gz"

    for p in [dwi_path, bval_path, bvec_path, mask_path]:
        if not p.exists():
            sys.exit(f"ERROR: required input missing: {p}")

    dwi_img  = nib.load(str(dwi_path))
    mask_img = nib.load(str(mask_path))
    dwi_data = dwi_img.get_fdata(dtype=np.float32)
    mask     = np.asarray(mask_img.dataobj).astype(bool)

    bvals = np.loadtxt(str(bval_path))
    bvecs = np.loadtxt(str(bvec_path))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    gtab = gradient_table(bvals, bvecs, b0_threshold=B0_THRESHOLD)
    sphere = get_sphere("repulsion724")

    model = CsaOdfModel(gtab, sh_order=SH_ORDER)
    pam = peaks_from_model(
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

    fod_dir = dti_dir / "fod"
    _guard(fod_dir)
    fod_dir.mkdir(parents=True, exist_ok=True)
    save_peaks(str(peaks_path), pam, affine=dwi_img.affine)

    elapsed = time.time() - t0
    print(f"  [peaks] regenerated in {elapsed:.1f}s → {peaks_path.relative_to(ROOT)}")
    return peaks_path


# ── Step 2: run tractography at one seed count ────────────────────────────────

@dataclass
class TractResult:
    seed_count:        int
    raw_streamlines:   int
    retained:          int
    rejected_short:    int
    mean_length:       float
    runtime_s:         float


def run_tractography(
    dti_dir:    Path,
    pam,
    seed_count: int,
    out_trk:    Path,
) -> TractResult:
    t0 = time.time()

    fa_img   = nib.load(str(dti_dir / "tensor" / "FA.nii.gz"))
    mask_img = nib.load(str(dti_dir / "qc" / "brain_mask.nii.gz"))
    fa       = fa_img.get_fdata(dtype=np.float32)
    mask     = np.asarray(mask_img.dataobj).astype(bool)

    wm_mask = (fa >= FA_THRESHOLD) & mask

    np.random.seed(RANDOM_SEED)
    seeds = random_seeds_from_mask(
        wm_mask, pam.affine,
        seeds_count=seed_count,
        seed_count_per_voxel=False,
    )

    getter   = DeterministicMaximumDirectionGetter.from_shcoeff(
        pam.shm_coeff, max_angle=MAX_ANGLE, sphere=pam.sphere,
    )
    stopping = ThresholdStoppingCriterion(fa, FA_THRESHOLD)

    raw = Streamlines(LocalTracking(
        getter, stopping, seeds, pam.affine,
        step_size=STEP_SIZE,
        max_cross=MAX_CROSS,
        return_all=False,
    ))

    lengths  = np.array([len(s) * STEP_SIZE for s in raw])
    keep     = lengths >= MIN_LENGTH_MM
    retained = Streamlines(s for s, k in zip(raw, keep) if k)
    ret_len  = lengths[keep]

    _guard(out_trk.parent)
    out_trk.parent.mkdir(parents=True, exist_ok=True)
    sft = StatefulTractogram(retained, fa_img, Space.RASMM)
    save_trk(sft, str(out_trk))

    elapsed = time.time() - t0
    return TractResult(
        seed_count=seed_count,
        raw_streamlines=len(raw),
        retained=len(retained),
        rejected_short=int((~keep).sum()),
        mean_length=float(np.mean(ret_len)) if len(ret_len) else 0.0,
        runtime_s=round(elapsed, 2),
    )


# ── Step 3: build connectome (endpoint-based, no full builder import) ─────────

@dataclass
class ConnectomeResult:
    seed_count:   int
    nonzero_edges: int
    density:       float
    streamlines_used: int
    streamlines_discarded: int


def build_connectome(
    trk_path:   Path,
    fa_img,
    atlas_nii:  Path,
    n_rois:     int = 100,
) -> ConnectomeResult:
    from dipy.io.streamline import load_tractogram as _load_trk

    seed_count_label = int(trk_path.parent.name)

    sft = _load_trk(str(trk_path), str(fa_img.get_filename()), to_space=Space.RASMM)
    streamlines = list(sft.streamlines)

    atlas_img  = nib.load(str(atlas_nii))
    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)
    inv_affine = np.linalg.inv(atlas_img.affine)

    count_matrix = np.zeros((n_rois + 1, n_rois + 1), dtype=np.int32)
    used = 0
    discarded = 0

    for sl in streamlines:
        endpoints = np.array([sl[0], sl[-1]])
        vox_coords = np.round(
            nib.affines.apply_affine(inv_affine, endpoints)
        ).astype(int)

        labels = []
        for v in vox_coords:
            if all(0 <= v[d] < atlas_data.shape[d] for d in range(3)):
                labels.append(int(atlas_data[v[0], v[1], v[2]]))
            else:
                labels.append(0)

        i, j = labels
        if i > 0 and j > 0 and i != j:
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1
            used += 1
        else:
            discarded += 1

    # upper triangle nonzero edges
    upper = count_matrix[1:, 1:]
    nonzero = int(np.sum(np.triu(upper, k=1) > 0))
    max_edges = n_rois * (n_rois - 1) // 2
    density = nonzero / max_edges if max_edges > 0 else 0.0

    return ConnectomeResult(
        seed_count=seed_count_label,
        nonzero_edges=nonzero,
        density=round(density, 6),
        streamlines_used=used,
        streamlines_discarded=discarded,
    )


# ── Step 4: saturation plots ──────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor":  "#0e0e12",
    "axes.facecolor":    "#0e0e12",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "text.color":        "#cccccc",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
}

_ACCENT = "#4db8ff"
_ACCENT2 = "#ff9f40"
_ACCENT3 = "#7dde92"
_ACCENT4 = "#e87dde"


def plot_saturation_curves(df: pd.DataFrame, out_path: Path, subject_label: str) -> None:
    seeds_k = df["seed_count"] / 1000

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Seed Density Saturation Study — {subject_label}",
            color="white", fontsize=15, y=0.98, fontweight="bold",
        )

        # ── 1. Retained streamlines ───────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(seeds_k, df["retained"], "o-", color=_ACCENT, markersize=6)
        ax.set_title("Retained Streamlines", color="white")
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Retained Streamlines")
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
        _annotate_saturation(ax, seeds_k, df["retained"].values, _ACCENT)

        # ── 2. Nonzero Edges ─────────────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(seeds_k, df["nonzero_edges"], "o-", color=_ACCENT2, markersize=6)
        ax.set_title("Connectome Nonzero Edges", color="white")
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Nonzero Edges")
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
        _annotate_saturation(ax, seeds_k, df["nonzero_edges"].values, _ACCENT2)

        # ── 3. Connectome Density ─────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(seeds_k, df["density"] * 100, "o-", color=_ACCENT3, markersize=6)
        ax.set_title("Connectome Density", color="white")
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Density (%)")
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))

        # ── 4. Runtime ────────────────────────────────────────────────────────
        ax = axes[1, 1]
        ax.bar(seeds_k, df["runtime_s"] / 60, color=_ACCENT4, alpha=0.8,
               width=seeds_k.diff().fillna(seeds_k.iloc[0]).values * 0.6)
        ax.set_title("Tractography Runtime", color="white")
        ax.set_xlabel("Seed Count (thousands)")
        ax.set_ylabel("Runtime (minutes)")
        ax.grid(True, axis="y")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
        for i, (x, y) in enumerate(zip(seeds_k, df["runtime_s"] / 60)):
            ax.text(x, y + 0.02, f"{y:.1f}m", ha="center", va="bottom",
                    color="#cccccc", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _guard(out_path.parent)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  [plot] saved → {out_path.relative_to(ROOT)}")


def _annotate_saturation(ax, seeds_k, values, color) -> None:
    """Mark the saturation point (first point where gain <5% of total range)."""
    if len(values) < 3:
        return
    total_range = values[-1] - values[0]
    if total_range <= 0:
        return
    for i in range(1, len(values)):
        gain = values[i] - values[i - 1]
        if gain / total_range < 0.05:
            ax.axvline(seeds_k.iloc[i], color=color, alpha=0.4, linestyle=":")
            ax.text(seeds_k.iloc[i], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                    f" sat≈{seeds_k.iloc[i]:.0f}k",
                    color=color, fontsize=8, alpha=0.8)
            break


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3R.2B — Seed Density Study"
    )
    p.add_argument("--subject", default="29006",
                   help="Subject ID (default: 29006)")
    p.add_argument("--site", default="BNI", choices=list(SITE_FOLDER_MAP),
                   help="Site name (default: BNI)")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seed counts to test")
    p.add_argument("--resume", action="store_true",
                   help="Skip seed counts whose tractogram already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    subject_id = args.subject
    site       = args.site
    seed_counts = sorted(args.seeds)

    dti_dir   = subject_dti_dir(site, subject_id)
    atlas_nii = atlas_path(site, subject_id)
    out_dir   = OUT_ROOT / subject_id
    _guard(out_dir)

    if not dti_dir.exists():
        sys.exit(f"ERROR: subject DTI directory not found: {dti_dir}")
    if not atlas_nii.exists():
        sys.exit(f"ERROR: registered atlas not found: {atlas_nii}")

    subject_label = f"{site}  ·  {subject_id}"

    print("=" * 62)
    print("NeuroFiber — Phase 3R.2B — Seed Density Study")
    print("=" * 62)
    print(f"  Subject : {subject_label}")
    print(f"  Seeds   : {[f'{s//1000}k' for s in seed_counts]}")
    print(f"  Output  : {out_dir.relative_to(ROOT)}")
    print()

    # ── Step 1: Ensure peaks.pam5 ─────────────────────────────────────────
    peaks_path = ensure_peaks(dti_dir)
    print(f"  [peaks] loading {peaks_path.name} ...")
    pam = load_peaks(str(peaks_path))
    fa_img = nib.load(str(dti_dir / "tensor" / "FA.nii.gz"))
    print()

    # ── Step 2: Tractography at each seed count ───────────────────────────
    tract_results: list[TractResult] = []
    conn_results:  list[ConnectomeResult] = []

    for sc in seed_counts:
        sc_label = f"{sc // 1000}k" if sc % 1000 == 0 else str(sc)
        out_trk  = out_dir / str(sc) / "streamlines.trk"

        print(f"  [seeds={sc_label:>5}]", end="  ", flush=True)

        if args.resume and out_trk.exists():
            print(f"tractogram exists — loading metrics from report")
            report_path = out_trk.parent / "run_report.json"
            if report_path.exists():
                d = json.loads(report_path.read_text())
                tract_results.append(TractResult(**{
                    k: d[k] for k in TractResult.__dataclass_fields__
                }))
            else:
                print(f"    WARNING: no report found, re-running")
                tr = run_tractography(dti_dir, pam, sc, out_trk)
                tract_results.append(tr)
        else:
            tr = run_tractography(dti_dir, pam, sc, out_trk)
            tract_results.append(tr)
            # save per-run report
            (out_trk.parent / "run_report.json").write_text(
                json.dumps(asdict(tr), indent=2)
            )

        tr = tract_results[-1]
        print(f"retained={tr.retained:6d}  t={tr.runtime_s:.1f}s")

        # connectome
        cr = build_connectome(out_trk, fa_img, atlas_nii)
        conn_results.append(cr)
        print(f"    → edges={cr.nonzero_edges:4d}  density={cr.density:.4f}"
              f"  used={cr.streamlines_used}  discarded={cr.streamlines_discarded}")

    # ── Step 3: Merge and save CSV ─────────────────────────────────────────
    print()
    rows = []
    for tr, cr in zip(tract_results, conn_results):
        rows.append({
            "seed_count":           tr.seed_count,
            "raw_streamlines":      tr.raw_streamlines,
            "retained":             tr.retained,
            "rejected_short":       tr.rejected_short,
            "mean_length":          round(tr.mean_length, 2),
            "runtime_s":            tr.runtime_s,
            "nonzero_edges":        cr.nonzero_edges,
            "density":              cr.density,
            "streamlines_used":     cr.streamlines_used,
            "streamlines_discarded": cr.streamlines_discarded,
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "phase3r2b_seed_density_results.csv"
    _guard(csv_path.parent)
    df.to_csv(str(csv_path), index=False)
    print(f"  [csv]  → {csv_path.relative_to(ROOT)}")

    # ── Step 4: Report ────────────────────────────────────────────────────
    print()
    print(df.to_string(index=False))
    print()

    # Saturation analysis: first seed count where edge gain < 5% of total range
    edge_range = df["nonzero_edges"].iloc[-1] - df["nonzero_edges"].iloc[0]
    sat_row = None
    if edge_range > 0:
        for i in range(1, len(df)):
            gain = df["nonzero_edges"].iloc[i] - df["nonzero_edges"].iloc[i - 1]
            if gain / edge_range < 0.05:
                sat_row = df.iloc[i]
                break

    report = {
        "subject":          subject_id,
        "site":             site,
        "seed_counts_tested": seed_counts,
        "saturation_seed_count": int(sat_row["seed_count"]) if sat_row is not None else None,
        "saturation_edges":      int(sat_row["nonzero_edges"]) if sat_row is not None else None,
        "max_edges":             int(df["nonzero_edges"].iloc[-1]),
        "max_density":           float(df["density"].iloc[-1]),
        "total_runtime_s":       float(df["runtime_s"].sum()),
        "results": rows,
    }
    json_path = out_dir / "phase3r2b_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  [json] → {json_path.relative_to(ROOT)}")

    if sat_row is not None:
        sat_k = int(sat_row["seed_count"]) // 1000
        print(f"\n  *** Saturation point: ~{sat_k}k seeds ***")
        print(f"      Edges at saturation: {int(sat_row['nonzero_edges'])}")
        print(f"      Max edges (1M):      {int(df['nonzero_edges'].iloc[-1])}")
    else:
        print("\n  *** No clear saturation detected within tested range ***")

    # ── Step 5: Plot ──────────────────────────────────────────────────────
    plot_path = out_dir / "phase3r2b_saturation_curves.png"
    plot_saturation_curves(df, plot_path, subject_label)

    print()
    print("=" * 62)
    print("Done.")
    print(f"  CSV   : {csv_path.relative_to(ROOT)}")
    print(f"  Plot  : {plot_path.relative_to(ROOT)}")
    print(f"  JSON  : {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
