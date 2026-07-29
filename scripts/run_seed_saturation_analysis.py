"""
NeuroFiber Phase 3R.2B — Seed Saturation Analysis

Uses existing tractography outputs (no re-running tractography).
Builds connectomes for each seed count, computes metrics, and identifies
the smallest seed count that recovers >=95% of 500k nonzero edges.

1M seeds excluded: computationally impractical on current hardware (stalled at ~116 min).

Inputs:
    data/processed_v2b/phase3r2b_seed_density/29006/{seed_count}/streamlines.trk
    data/processed/abide_ii/bni/ABIDEII-BNI_1/29006/session_1/connectome/atlas/atlas_subject_space.nii.gz

Outputs:
    data/experiments/seed_saturation/seed_saturation_summary.csv
    data/experiments/seed_saturation/seed_saturation_report.md
    data/experiments/seed_saturation/plot_retained_streamlines.png
    data/experiments/seed_saturation/plot_runtime.png
    data/experiments/seed_saturation/plot_nonzero_edges.png
    data/experiments/seed_saturation/plot_density.png
    data/experiments/seed_saturation/plot_edge_recovery.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
import pandas as pd

from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
TRK_ROOT   = ROOT / "data/processed_v2b/phase3r2b_seed_density/29006"
ATLAS_PATH = (ROOT / "data/processed/abide_ii/bni/ABIDEII-BNI_1/29006"
              / "session_1/connectome/atlas/atlas_subject_space.nii.gz")
FA_PATH    = (ROOT / "data/processed_v2b/abide_ii/bni/ABIDEII-BNI_1/29006"
              / "session_1/dti_1/tensor/FA.nii.gz")
OUT_DIR    = ROOT / "data/experiments/seed_saturation"

SEED_COUNTS  = [10_000, 50_000, 100_000, 250_000, 500_000]
REFERENCE_SC = 500_000
N_ROIS       = 100
RECOVERY_THR = 0.95     # >= 95% of reference edges required

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
    "lines.linewidth":  2.2,
}

COLORS = {
    "retained":  "#4db8ff",
    "runtime":   "#7dde92",
    "edges":     "#ff9f40",
    "density":   "#4db8ff",
    "recovery":  "#e87dde",
    "jaccard":   "#ff9f40",
    "ref":       "#ff6b6b",
    "threshold": "#ffffff",
}


# ── connectome builder ────────────────────────────────────────────────────────

def build_connectome(trk_path: Path, atlas_data: np.ndarray,
                     atlas_affine: np.ndarray) -> dict:
    """
    Endpoint-based connectome from a .trk file.
    Returns count matrix and derived metrics.
    """
    sft = load_tractogram(str(trk_path), str(trk_path), to_space=Space.RASMM)
    streamlines = list(sft.streamlines)

    inv_affine   = np.linalg.inv(atlas_affine)
    count_matrix = np.zeros((N_ROIS + 1, N_ROIS + 1), dtype=np.int32)
    used = discarded = 0

    for sl in streamlines:
        ep  = np.array([sl[0], sl[-1]])
        vc  = np.round(nib.affines.apply_affine(inv_affine, ep)).astype(int)
        lbs = []
        for v in vc:
            if all(0 <= v[d] < atlas_data.shape[d] for d in range(3)):
                lbs.append(int(atlas_data[v[0], v[1], v[2]]))
            else:
                lbs.append(0)
        i, j = lbs
        if i > 0 and j > 0 and i != j:
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1
            used += 1
        else:
            discarded += 1

    upper      = count_matrix[1:, 1:]
    upper_tri  = np.triu(upper, k=1)
    max_edges  = N_ROIS * (N_ROIS - 1) // 2
    nonzero    = int(np.sum(upper_tri > 0))
    density    = nonzero / max_edges
    nonzero_weights = upper_tri[upper_tri > 0]
    mean_weight     = float(np.mean(nonzero_weights)) if len(nonzero_weights) else 0.0

    # edge set as frozenset of (i,j) pairs for Jaccard computation
    edge_set = frozenset(
        zip(*np.where(upper_tri > 0))
    )

    return {
        "used":         used,
        "discarded":    discarded,
        "nonzero":      nonzero,
        "density":      density,
        "mean_weight":  mean_weight,
        "edge_set":     edge_set,
        "count_matrix": upper_tri,
    }


# ── comparison vs reference ───────────────────────────────────────────────────

def edge_similarity(edge_set: frozenset, ref_set: frozenset) -> dict:
    inter  = edge_set & ref_set
    union  = edge_set | ref_set
    jaccard  = len(inter) / len(union) if union else 1.0
    recovery = len(inter) / len(ref_set) if ref_set else 1.0
    overlap  = len(inter)
    return {
        "active_edge_overlap": overlap,
        "jaccard":             round(jaccard, 4),
        "edge_recovery_pct":   round(recovery * 100, 2),
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def _fmt_k(x, _):
    return f"{x:.0f}k"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _base_ax(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.grid(True)
    return fig, ax


def plot_retained(df: pd.DataFrame) -> Path:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = _base_ax(
            "Seed Count vs Retained Streamlines\nBNI · 29006",
            "Seed Count (thousands)", "Retained Streamlines",
        )
        sk = df["seed_count"] / 1000
        ax.plot(sk, df["retained"], "o-", color=COLORS["retained"], markersize=7)
        for x, y in zip(sk, df["retained"]):
            ax.text(x, y * 1.01, f"{y:,}", ha="center", va="bottom",
                    color="#aaaaaa", fontsize=8)
        out = OUT_DIR / "plot_retained_streamlines.png"
        _save(fig, out)
    print(f"  [plot] retained → {out.relative_to(ROOT)}")
    return out


def plot_runtime(df: pd.DataFrame) -> Path:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = _base_ax(
            "Seed Count vs Runtime\nBNI · 29006  (1M excluded — impractical on current hardware)",
            "Seed Count (thousands)", "Runtime (minutes)",
        )
        sk = df["seed_count"] / 1000
        rt = df["runtime_s"] / 60
        ax.bar(sk, rt, color=COLORS["runtime"], alpha=0.82,
               width=sk.diff().fillna(sk.iloc[0]).values * 0.55)
        for x, y in zip(sk, rt):
            ax.text(x, y + rt.max() * 0.02, f"{y:.1f}m",
                    ha="center", va="bottom", color="#dddddd",
                    fontsize=9, fontweight="bold")
        # annotate 1M as excluded
        ax.text(0.98, 0.92, "1M seeds: stalled at ~116 min\n(excluded from analysis)",
                transform=ax.transAxes, ha="right", va="top",
                color="#888888", fontsize=8,
                bbox=dict(facecolor="#1a1a22", edgecolor="#444444", pad=4))
        out = OUT_DIR / "plot_runtime.png"
        _save(fig, out)
    print(f"  [plot] runtime → {out.relative_to(ROOT)}")
    return out


def plot_nonzero_edges(df: pd.DataFrame) -> Path:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = _base_ax(
            "Seed Count vs Nonzero Edges\nBNI · 29006",
            "Seed Count (thousands)", "Nonzero Edges",
        )
        sk = df["seed_count"] / 1000
        ax.plot(sk, df["nonzero_edges"], "o-", color=COLORS["edges"], markersize=7)
        # reference line at 500k
        ref_val = df.loc[df["seed_count"] == REFERENCE_SC, "nonzero_edges"].iloc[0]
        ax.axhline(ref_val, color=COLORS["ref"], linestyle="--", alpha=0.6,
                   label=f"500k reference ({ref_val} edges)")
        for x, y in zip(sk, df["nonzero_edges"]):
            ax.text(x, y + ref_val * 0.01, str(y), ha="center", va="bottom",
                    color="#aaaaaa", fontsize=8)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        out = OUT_DIR / "plot_nonzero_edges.png"
        _save(fig, out)
    print(f"  [plot] edges → {out.relative_to(ROOT)}")
    return out


def plot_density(df: pd.DataFrame) -> Path:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = _base_ax(
            "Seed Count vs Connectome Density\nBNI · 29006",
            "Seed Count (thousands)", "Connectome Density (%)",
        )
        sk = df["seed_count"] / 1000
        ax.plot(sk, df["density"] * 100, "o-", color=COLORS["density"], markersize=7)
        ref_den = df.loc[df["seed_count"] == REFERENCE_SC, "density"].iloc[0]
        ax.axhline(ref_den * 100, color=COLORS["ref"], linestyle="--", alpha=0.6,
                   label=f"500k reference ({ref_den*100:.2f}%)")
        for x, y in zip(sk, df["density"] * 100):
            ax.text(x, y + ref_den * 100 * 0.01, f"{y:.2f}%",
                    ha="center", va="bottom", color="#aaaaaa", fontsize=8)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        out = OUT_DIR / "plot_density.png"
        _save(fig, out)
    print(f"  [plot] density → {out.relative_to(ROOT)}")
    return out


def plot_edge_recovery(df: pd.DataFrame) -> Path:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = _base_ax(
            "Edge Recovery vs 500k Reference\nBNI · 29006",
            "Seed Count (thousands)", "Recovery / Similarity (%)",
        )
        sk = df["seed_count"] / 1000

        ax.plot(sk, df["edge_recovery_pct"], "o-",
                color=COLORS["recovery"], markersize=7, label="% of 500k edges recovered")
        ax.plot(sk, df["jaccard"] * 100, "s--",
                color=COLORS["jaccard"], markersize=6, label="Jaccard similarity (%)")

        # 95% threshold line
        ax.axhline(RECOVERY_THR * 100, color=COLORS["threshold"], linestyle=":",
                   alpha=0.5, linewidth=1.2, label=f"{int(RECOVERY_THR*100)}% threshold")

        # mark decision point
        decision = df[df["edge_recovery_pct"] >= RECOVERY_THR * 100]
        if not decision.empty:
            dp = decision.iloc[0]
            ax.axvline(dp["seed_count"] / 1000, color=COLORS["recovery"],
                       alpha=0.4, linestyle=":", linewidth=1.4)
            ax.text(dp["seed_count"] / 1000 * 1.02,
                    df["edge_recovery_pct"].min() + 2,
                    f"  optimal: {int(dp['seed_count'])//1000}k seeds\n"
                    f"  recovery: {dp['edge_recovery_pct']:.1f}%",
                    color=COLORS["recovery"], fontsize=8)

        ax.set_ylim(0, 105)
        ax.legend(facecolor="#1a1a22", edgecolor="#333333", labelcolor="#cccccc")
        out = OUT_DIR / "plot_edge_recovery.png"
        _save(fig, out)
    print(f"  [plot] recovery → {out.relative_to(ROOT)}")
    return out


# ── markdown report ───────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, decision_sc: int | None) -> Path:
    lines = [
        "# NeuroFiber — Seed Density Saturation Study",
        "",
        "**Subject:** BNI · 29006  ",
        "**Atlas:** Schaefer 100 Parcels / 7 Networks  ",
        "**Reference:** 500k seeds  ",
        "**Decision threshold:** ≥95% of 500k nonzero edges recovered  ",
        "**1M seeds:** excluded — stalled at ~116 min on current hardware (CPU-only macOS); computationally impractical for batch pipeline  ",
        "",
        "---",
        "",
        "## Metrics Table",
        "",
    ]

    # table header
    cols = [
        ("Seed Count", "seed_count", lambda x: f"{int(x):,}"),
        ("Retained", "retained", lambda x: f"{int(x):,}"),
        ("Runtime", "runtime_s", lambda x: f"{x/60:.1f} min"),
        ("Mean Length (mm)", "mean_length_mm", lambda x: f"{x:.1f}"),
        ("Nonzero Edges", "nonzero_edges", lambda x: str(int(x))),
        ("Density", "density", lambda x: f"{x*100:.3f}%"),
        ("Mean Edge Wt", "mean_weight", lambda x: f"{x:.2f}"),
        ("Active Overlap", "active_edge_overlap", lambda x: str(int(x))),
        ("Jaccard", "jaccard", lambda x: f"{x:.4f}"),
        ("Recovery vs 500k", "edge_recovery_pct", lambda x: f"{x:.1f}%"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines += [header, sep]

    for _, row in df.iterrows():
        cells = [fmt(row[col]) for _, col, fmt in cols]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## Decision",
        "",
    ]

    if decision_sc is not None:
        ref_row = df[df["seed_count"] == REFERENCE_SC].iloc[0]
        dec_row = df[df["seed_count"] == decision_sc].iloc[0]
        lines += [
            f"**Optimal seed count: {decision_sc:,}**  ",
            f"- Recovers **{dec_row['edge_recovery_pct']:.1f}%** of 500k nonzero edges  ",
            f"- Jaccard similarity: **{dec_row['jaccard']:.4f}**  ",
            f"- Nonzero edges: **{int(dec_row['nonzero_edges'])}** "
            f"(vs {int(ref_row['nonzero_edges'])} at 500k)  ",
            f"- Runtime: **{dec_row['runtime_s']/60:.1f} min** "
            f"(vs {ref_row['runtime_s']/60:.1f} min at 500k)  ",
            f"- Speedup vs 500k: **{ref_row['runtime_s']/dec_row['runtime_s']:.1f}×**  ",
            "",
        ]
    else:
        lines += [
            "**No seed count within the tested range meets the ≥95% recovery threshold.**  ",
            "Consider testing higher seed counts (>500k) if hardware permits.  ",
            "",
        ]

    lines += [
        "---",
        "",
        "## Notes on 1M Seeds",
        "",
        "A 1M seed run was attempted but stalled at approximately 116 minutes elapsed "
        "without completing. The 500k run took 29.2 minutes. The 1M run is estimated "
        "to require ~58–60 minutes on this hardware configuration (single-core CPU, "
        "macOS). For a 229-subject batch pipeline this would be computationally "
        "impractical. The optimal seed count identified above is recommended for "
        "production use.",
        "",
        "---",
        "",
        "## Plots",
        "",
        "- `plot_retained_streamlines.png` — retained streamlines vs seed count",
        "- `plot_runtime.png` — runtime vs seed count",
        "- `plot_nonzero_edges.png` — nonzero edges vs seed count",
        "- `plot_density.png` — connectome density vs seed count",
        "- `plot_edge_recovery.png` — edge recovery and Jaccard similarity vs 500k reference",
    ]

    out = OUT_DIR / "seed_saturation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  [md]   → {out.relative_to(ROOT)}")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("NeuroFiber — Seed Saturation Analysis")
    print("=" * 62)
    print(f"  Atlas  : {ATLAS_PATH.relative_to(ROOT)}")
    print(f"  Output : {OUT_DIR.relative_to(ROOT)}")
    print()

    for p in [ATLAS_PATH, FA_PATH]:
        if not p.exists():
            sys.exit(f"ERROR: required file missing: {p}")

    atlas_img  = nib.load(str(ATLAS_PATH))
    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)

    # ── build connectomes ──────────────────────────────────────────────────
    results   = []
    edge_sets = {}

    for sc in SEED_COUNTS:
        trk_path    = TRK_ROOT / str(sc) / "streamlines.trk"
        report_path = TRK_ROOT / str(sc) / "run_report.json"

        if not trk_path.exists():
            print(f"  [SKIP] {sc:,} seeds — streamlines.trk not found")
            continue
        if not report_path.exists():
            print(f"  [SKIP] {sc:,} seeds — run_report.json not found")
            continue

        with open(report_path) as f:
            rep = json.load(f)

        print(f"  [{sc:>8,} seeds]  building connectome ...", flush=True)
        cm = build_connectome(trk_path, atlas_data, atlas_img.affine)
        edge_sets[sc] = cm["edge_set"]

        results.append({
            "seed_count":    sc,
            "retained":      rep["retained"],
            "runtime_s":     rep["runtime_s"],
            "mean_length_mm": round(rep["mean_length"], 1),
            "nonzero_edges": cm["nonzero"],
            "density":       cm["density"],
            "mean_weight":   round(cm["mean_weight"], 4),
        })
        print(f"    → edges={cm['nonzero']}  density={cm['density']:.4f}  "
              f"used={cm['used']}  discarded={cm['discarded']}")

    if not results:
        sys.exit("ERROR: no completed seed counts found.")

    df = pd.DataFrame(results).sort_values("seed_count").reset_index(drop=True)

    # ── reference comparison ───────────────────────────────────────────────
    ref_set = edge_sets.get(REFERENCE_SC)
    if ref_set is None:
        sys.exit(f"ERROR: reference seed count {REFERENCE_SC:,} not found.")

    print()
    print("  Computing edge similarity vs 500k reference ...")
    sim_rows = []
    for sc in df["seed_count"]:
        sim = edge_similarity(edge_sets[sc], ref_set)
        sim_rows.append(sim)

    sim_df = pd.DataFrame(sim_rows)
    df     = pd.concat([df.reset_index(drop=True), sim_df], axis=1)

    # ── print table ────────────────────────────────────────────────────────
    print()
    display = df[[
        "seed_count", "retained", "runtime_s", "nonzero_edges",
        "density", "mean_weight", "active_edge_overlap", "jaccard", "edge_recovery_pct",
    ]].copy()
    display["runtime_min"]  = (display["runtime_s"] / 60).round(1)
    display["density_pct"]  = (display["density"] * 100).round(3)
    print(display[[
        "seed_count", "retained", "runtime_min", "mean_weight",
        "nonzero_edges", "density_pct", "active_edge_overlap", "jaccard", "edge_recovery_pct",
    ]].to_string(index=False))

    # ── decision ───────────────────────────────────────────────────────────
    print()
    decision_row = df[df["edge_recovery_pct"] >= RECOVERY_THR * 100]
    if not decision_row.empty:
        decision_sc = int(decision_row.iloc[0]["seed_count"])
        dec = decision_row.iloc[0]
        ref = df[df["seed_count"] == REFERENCE_SC].iloc[0]
        print(f"  DECISION: optimal seed count = {decision_sc:,}")
        print(f"    recovery  = {dec['edge_recovery_pct']:.1f}%")
        print(f"    jaccard   = {dec['jaccard']:.4f}")
        print(f"    speedup   = {ref['runtime_s']/dec['runtime_s']:.1f}× vs 500k")
    else:
        decision_sc = None
        print("  DECISION: no seed count meets >=95% recovery within tested range")

    # ── save CSV ───────────────────────────────────────────────────────────
    csv_out = OUT_DIR / "seed_saturation_summary.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["density"]).assign(
        density=df["density"]
    ).to_csv(str(csv_out), index=False)
    print(f"\n  [csv]  → {csv_out.relative_to(ROOT)}")

    # ── plots ──────────────────────────────────────────────────────────────
    print()
    plot_retained(df)
    plot_runtime(df)
    plot_nonzero_edges(df)
    plot_density(df)
    plot_edge_recovery(df)

    # ── report ─────────────────────────────────────────────────────────────
    write_report(df, decision_sc)

    print()
    print("=" * 62)
    print("Done.")
    print(f"  CSV    : {csv_out.relative_to(ROOT)}")
    print(f"  Report : {(OUT_DIR / 'seed_saturation_report.md').relative_to(ROOT)}")
    print(f"  Plots  : {OUT_DIR.relative_to(ROOT)}/plot_*.png")


if __name__ == "__main__":
    main()
