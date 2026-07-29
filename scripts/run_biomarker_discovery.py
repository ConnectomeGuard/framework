"""
Phase 4.1 — ASD Biomarker Discovery

Runs edge-level statistical association analysis on ComBat-harmonized
FA, MD, and length connectome features.

Scientific context
──────────────────
This is biomarker discovery, NOT clinical diagnosis.
Do NOT interpret outputs as diagnostic criteria.
Candidate edges are research findings requiring external validation.

Usage
─────
  python scripts/run_biomarker_discovery.py [--perms N] [--feature md_edges fa_edges length_edges]

  Quick run (50 perms):
    python scripts/run_biomarker_discovery.py --perms 50

  MD only (faster):
    python scripts/run_biomarker_discovery.py --feature md_edges --perms 100

Outputs
───────
  data/canonical_v1/biomarkers/
    md_edge_statistics.csv
    fa_edge_statistics.csv
    length_edge_statistics.csv
    candidate_biomarkers.csv
    site_robustness.csv
    leave_one_site_out.csv
    biomarker_summary.json
    figures/
      volcano_md.png
      top20_md_edges.png
      effect_size_distribution.png
      site_consistency_heatmap.png
      top10_boxplots_md.png
      manhattan_md.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.biomarkers.biomarker_discovery import (
    BiomarkerDataset,
    DiscoveryResult,
    BALANCED_SITES,
    FDR_STRICT,
    FDR_EXPLORATORY,
    load_dataset,
    build_edge_index_table,
    compute_edge_stats,
    apply_fdr,
    compute_site_robustness,
    stability_score,
    permutation_test,
    network_summary,
    run_biomarker_discovery,
    _stats_to_df,
    _robustness_to_df,
    _loso_to_df,
)

OUT_DIR = REPO_ROOT / "data" / "canonical_v1" / "biomarkers"
FIG_DIR = OUT_DIR / "figures"

log = logging.getLogger("run_biomarker_discovery")


# ── visualizations ─────────────────────────────────────────────────────────────

def plot_volcano(stats_df: pd.DataFrame, feature_name: str, out_path: Path) -> None:
    """Volcano plot: Cohen's d vs -log10(p_B), colored by FDR category."""
    fig, ax = plt.subplots(figsize=(9, 7))

    df = stats_df.copy()
    df["neg_log_p"] = -np.log10(df["p_B"].clip(lower=1e-300))
    df["abs_d"] = df["cohen_d"].abs()

    # Color coding
    colors = []
    for _, row in df.iterrows():
        if row["q_B_strict"]:
            colors.append("#d62728")   # red: q≤0.05
        elif row["q_B_exploratory"]:
            colors.append("#ff7f0e")   # orange: q≤0.10
        elif row["p_B"] < 0.05:
            colors.append("#aec7e8")   # light blue: nominal
        else:
            colors.append("#cccccc")   # gray: not significant

    ax.scatter(df["cohen_d"], df["neg_log_p"], c=colors, s=12, alpha=0.7, linewidths=0)

    # Thresholds
    ax.axvline(x=0, color="black", lw=0.5, linestyle="--")
    p_bh_line = df.loc[df["q_B_exploratory"], "p_B"].max() if df["q_B_exploratory"].any() else None
    if p_bh_line is not None:
        ax.axhline(y=-np.log10(p_bh_line), color="#ff7f0e", lw=1, linestyle="--",
                   label=f"FDR q≤0.10 (p={p_bh_line:.3f})")

    # Top 5 labels
    top = df.nlargest(5, "neg_log_p")
    for _, row in top.iterrows():
        ax.annotate(
            row["edge_id"],
            xy=(row["cohen_d"], row["neg_log_p"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=6, color="#333333",
        )

    patches = [
        mpatches.Patch(color="#d62728", label=f"q ≤ {FDR_STRICT} ({df['q_B_strict'].sum()})"),
        mpatches.Patch(color="#ff7f0e", label=f"q ≤ {FDR_EXPLORATORY} ({df['q_B_exploratory'].sum()})"),
        mpatches.Patch(color="#aec7e8", label="p < 0.05 nominal"),
        mpatches.Patch(color="#cccccc", label="NS"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("Cohen's d (ASD − Control)", fontsize=11)
    ax.set_ylabel("−log₁₀(p_B)", fontsize=11)
    ax.set_title(f"Volcano Plot — {feature_name.replace('_', ' ').title()}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_top20_effects(stats_df: pd.DataFrame, feature_name: str, out_path: Path) -> None:
    """Horizontal bar chart of top-20 edges by |Cohen's d|."""
    df = stats_df.dropna(subset=["cohen_d"]).copy()
    df["abs_d"] = df["cohen_d"].abs()
    top = df.nlargest(20, "abs_d").sort_values("cohen_d")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if d >= 0 else "#1f77b4" for d in top["cohen_d"]]
    labels = [f"{r['edge_id']}  ({r['roi_i_name']}—{r['roi_j_name']})" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["cohen_d"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Cohen's d (ASD − Control)", fontsize=11)
    ax.set_title(f"Top 20 ASD-Associated Edges — {feature_name.replace('_', ' ').title()}", fontsize=12)
    ax.text(0.98, 0.02, "Red = ASD higher   Blue = ASD lower",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_effect_distribution(results: dict, out_path: Path) -> None:
    """Overlaid Cohen's d distributions for MD, FA, length."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_map = {"md_edges": "#1f77b4", "fa_edges": "#2ca02c", "length_edges": "#ff7f0e"}
    for fn, res in results.items():
        df = res.edge_stats.dropna(subset=["cohen_d"])
        label = fn.replace("_edges", "").upper()
        ax.hist(df["cohen_d"], bins=60, alpha=0.5, color=colors_map.get(fn, "gray"),
                label=f"{label} (n={len(df)})", density=True)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Cohen's d (ASD − Control)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Effect Size Distribution Across Feature Types", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_site_consistency_heatmap(rob_df: pd.DataFrame, stats_df: pd.DataFrame,
                                   feature_name: str, out_path: Path) -> None:
    """Heatmap of per-site Cohen's d for the top candidates."""
    df = stats_df.merge(rob_df[["edge_id"] + [f"cohen_d_{s}" for s in BALANCED_SITES]],
                        on="edge_id")
    df = df.sort_values("stability_score", ascending=False).head(30)

    if df.empty:
        log.warning("  No candidates for site consistency heatmap.")
        return

    heat = df[[f"cohen_d_{s}" for s in BALANCED_SITES]].values.T  # (4, n_edges)
    labels_x = df["edge_id"].tolist()

    vmax = np.nanmax(np.abs(heat)) if not np.all(np.isnan(heat)) else 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(labels_x) * 0.4 + 2), 4))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(BALANCED_SITES)))
    ax.set_yticklabels(BALANCED_SITES, fontsize=9)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=90, fontsize=6)
    ax.set_title(f"Site Consistency (Cohen's d) — Top 30 Candidates — {feature_name.replace('_', ' ').title()}",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Cohen's d")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_top10_boxplots(
    ds: BiomarkerDataset,
    stats_df: pd.DataFrame,
    feature_name: str,
    out_path: Path,
) -> None:
    """ASD vs Control boxplots for top 10 edges by stability score."""
    top10 = stats_df.nlargest(10, "stability_score")
    if top10.empty:
        return

    meta = ds.meta
    dx = meta["diagnosis"].values

    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()

    edge_name_to_pos = {ec: pos for pos, ec in enumerate(ds.edge_cols)}
    X = ds.X_combat

    for k, (_, row) in enumerate(top10.iterrows()):
        ax = axes[k]
        pos = edge_name_to_pos.get(row["edge_id"])
        if pos is None:
            ax.axis("off")
            continue
        y = X[:, pos]
        valid = ~np.isnan(y)
        y_v = y[valid]; dx_v = dx[valid]
        asd_vals  = y_v[dx_v == "ASD"]
        ctrl_vals = y_v[dx_v == "CONTROL"]
        ax.boxplot([ctrl_vals, asd_vals], labels=["Control", "ASD"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#aec7e8"),
                   medianprops=dict(color="black", lw=2))
        ax.set_title(
            f"{row['edge_id']}\n{row['roi_i_name'].split('_')[-1]}—{row['roi_j_name'].split('_')[-1]}",
            fontsize=7
        )
        ax.set_ylabel(feature_name.split("_")[0].upper(), fontsize=7)
        # add d and p annotations
        ax.text(0.97, 0.97,
                f"d={row['cohen_d']:.2f}\np={row['p_B']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)

    for k in range(len(top10), len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Top 10 Candidate Edges — {feature_name.replace('_', ' ').title()}",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_manhattan(stats_df: pd.DataFrame, feature_name: str, out_path: Path) -> None:
    """Manhattan-style plot: edge index vs -log10(p_B), colored by significance."""
    df = stats_df.copy().reset_index(drop=True)
    df["neg_log_p"] = -np.log10(df["p_B"].clip(lower=1e-300))
    df["edge_num"] = df["edge_id"].apply(lambda x: int(x.split("_")[1]))

    colors = []
    for _, row in df.iterrows():
        if row["q_B_strict"]:
            colors.append("#d62728")
        elif row["q_B_exploratory"]:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df["edge_num"], df["neg_log_p"], c=colors, s=4, alpha=0.6, linewidths=0)

    # Genome-wide significance line analogue
    p_bh_line = df.loc[df["q_B_exploratory"], "p_B"].max() if df["q_B_exploratory"].any() else None
    if p_bh_line is not None:
        ax.axhline(-np.log10(p_bh_line), color="#ff7f0e", lw=1, linestyle="--",
                   label=f"FDR q≤0.10")
    p05_line = df.loc[df["q_B_strict"], "p_B"].max() if df["q_B_strict"].any() else None
    if p05_line is not None:
        ax.axhline(-np.log10(p05_line), color="#d62728", lw=1, linestyle="--",
                   label=f"FDR q≤0.05")

    ax.set_xlabel("Edge index", fontsize=11)
    ax.set_ylabel("−log₁₀(p_B)", fontsize=11)
    ax.set_title(f"Manhattan Plot — {feature_name.replace('_', ' ').title()} (Model B)", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


# ── summary JSON ───────────────────────────────────────────────────────────────

def build_summary_json(results: dict, datasets: dict) -> dict:
    summary: dict = {
        "phase": "4.1",
        "analysis": "ASD Biomarker Discovery",
        "note": "Statistical association only. Not a diagnostic model.",
        "features": {},
    }
    for fn, res in results.items():
        ds = datasets[fn]
        perm_empirical_p = None
        if res.permutation_null_min_p is not None and not np.isnan(res.permutation_observed_min_p):
            perm_empirical_p = float(
                (res.permutation_null_min_p <= res.permutation_observed_min_p).mean()
            )

        net_df = network_summary(res.edge_stats)
        top_network_pair = None
        if not net_df.empty:
            row = net_df.iloc[0]
            top_network_pair = f"{row['network_i']}—{row['network_j']} (n={row['n_edges']})"

        top10_list = []
        for _, r in res.top10.iterrows():
            top10_list.append({
                "edge_id":     r["edge_id"],
                "roi_pair":    f"{r['roi_i_name']}—{r['roi_j_name']}",
                "networks":    f"{r['network_i']}—{r['network_j']}",
                "cohen_d":     round(float(r["cohen_d"]), 4),
                "p_B":         float(r["p_B"]),
                "q_B":         float(r["q_B"]),
                "direction":   r["direction"],
                "stability_score": float(r["stability_score"]),
                "cross_site_active": bool(r["is_cross_site_active"]),
            })

        summary["features"][fn] = {
            "n_subjects":         int(len(ds.meta)),
            "n_cross_site_active": int(ds.active_mask.sum()),
            "n_tested_edges":      res.n_tested,
            "n_significant_q05":   res.n_q05,
            "n_significant_q10":   res.n_q10,
            "n_candidates":        int(len(res.candidate_biomarkers)),
            "permutation_n_perms": int(len(res.permutation_null_min_p))
                                   if res.permutation_null_min_p is not None else 0,
            "permutation_observed_min_p": float(res.permutation_observed_min_p)
                                          if not np.isnan(res.permutation_observed_min_p) else None,
            "permutation_empirical_p":    perm_empirical_p,
            "top_network_pair":    top_network_pair,
            "top10":               top10_list,
        }
    return summary


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.1 — ASD Biomarker Discovery")
    p.add_argument(
        "--feature", nargs="+",
        default=["md_edges", "fa_edges", "length_edges"],
        choices=["md_edges", "fa_edges", "length_edges"],
        help="Feature types to analyze (default: all three)",
    )
    p.add_argument("--perms", type=int, default=100,
                   help="Number of permutations for null distribution (default: 100)")
    p.add_argument("--no-combat", action="store_true",
                   help="Use unharmonized features (not recommended)")
    p.add_argument("--min-valid", type=int, default=15,
                   help="Minimum non-NaN subjects to test an edge (default: 15)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    use_combat = not args.no_combat
    log.info("Phase 4.1 — ASD Biomarker Discovery")
    log.info("Features:     %s", args.feature)
    log.info("Permutations: %d", args.perms)
    log.info("Use ComBat:   %s", use_combat)
    log.info("Min valid:    %d", args.min_valid)
    log.info("")

    edge_index_table = build_edge_index_table()
    results: dict = {}
    datasets: dict = {}

    for fn in args.feature:
        log.info("=" * 60)
        log.info("Feature: %s", fn)
        log.info("=" * 60)

        ds = load_dataset(REPO_ROOT, fn)
        datasets[fn] = ds
        n_subjects = len(ds.meta)

        log.info("Computing per-edge statistics …")
        stats_list = compute_edge_stats(
            ds, edge_index_table,
            use_combat=use_combat,
            min_n_valid=args.min_valid,
        )
        log.info("  Tested: %d edges", len(stats_list))
        stats_list = apply_fdr(stats_list)

        n_q05 = sum(1 for s in stats_list if s.q_B_strict)
        n_q10 = sum(1 for s in stats_list if s.q_B_exploratory)
        log.info("  q≤0.05: %d   q≤0.10: %d", n_q05, n_q10)

        log.info("Computing site robustness …")
        robustness_list = compute_site_robustness(
            ds, stats_list, edge_index_table, use_combat=use_combat
        )

        rob_map = {r.edge_id: r for r in robustness_list}
        scores = [
            stability_score(es, rob_map[es.edge_id], n_subjects)
            if es.edge_id in rob_map else 0.0
            for es in stats_list
        ]

        stats_df = _stats_to_df(stats_list, scores)
        rob_df   = _robustness_to_df(robustness_list)
        loso_df  = _loso_to_df(robustness_list)

        candidate_mask = stats_df["q_B_exploratory"] | (stats_df["stability_score"] >= 0.50)
        candidates = stats_df[candidate_mask].copy().sort_values("stability_score", ascending=False)
        candidates["nyu2_stable"] = candidates["edge_id"].map(
            lambda eid: rob_map[eid].nyu2_removed_same_dir if eid in rob_map else True
        )

        log.info("Running permutation test (%d perms) …", args.perms)
        null_min_p, obs_min_p = permutation_test(
            ds, n_perms=args.perms, use_combat=use_combat, min_n_valid=args.min_valid
        )
        if not np.isnan(obs_min_p):
            emp_p = float((null_min_p <= obs_min_p).mean())
            log.info("  obs min-p=%.2e  empirical p=%.3f", obs_min_p, emp_p)

        top10 = stats_df.nlargest(10, "stability_score")[
            ["edge_id", "roi_i_name", "roi_j_name", "network_i", "network_j",
             "cohen_d", "p_B", "q_B", "direction", "n_valid",
             "stability_score", "is_cross_site_active"]
        ]

        results[fn] = DiscoveryResult(
            feature_name=fn,
            edge_stats=stats_df,
            site_robustness=rob_df,
            loso_table=loso_df,
            candidate_biomarkers=candidates,
            permutation_null_min_p=null_min_p,
            permutation_observed_min_p=obs_min_p,
            n_tested=len(stats_list),
            n_q05=n_q05,
            n_q10=n_q10,
            top10=top10,
        )

        # Save per-feature CSVs
        short = fn.replace("_edges", "")
        stats_df.to_csv(OUT_DIR / f"{short}_edge_statistics.csv", index=False)
        rob_df.to_csv(OUT_DIR / f"{short}_site_robustness.csv", index=False)
        loso_df.to_csv(OUT_DIR / f"{short}_loso.csv", index=False)
        log.info("  Saved CSVs to %s", OUT_DIR.relative_to(REPO_ROOT))

        # Figures
        log.info("Generating figures …")
        plot_volcano(stats_df, fn, FIG_DIR / f"volcano_{short}.png")
        plot_top20_effects(stats_df, fn, FIG_DIR / f"top20_{short}_edges.png")
        plot_manhattan(stats_df, fn, FIG_DIR / f"manhattan_{short}.png")
        plot_site_consistency_heatmap(rob_df, stats_df, fn,
                                      FIG_DIR / f"site_consistency_{short}.png")
        plot_top10_boxplots(ds, stats_df, fn, FIG_DIR / f"top10_boxplots_{short}.png")

    # Combined effect size distribution
    if len(results) > 1:
        plot_effect_distribution(results, FIG_DIR / "effect_size_distribution.png")

    # Save combined candidate list
    all_candidates = pd.concat([r.candidate_biomarkers.assign(feature=fn)
                                 for fn, r in results.items()], ignore_index=True)
    all_candidates.to_csv(OUT_DIR / "candidate_biomarkers.csv", index=False)

    # Merge all site robustness into one file (primary feature = md_edges)
    if "md_edges" in results:
        results["md_edges"].site_robustness.to_csv(OUT_DIR / "site_robustness.csv", index=False)
        results["md_edges"].loso_table.to_csv(OUT_DIR / "leave_one_site_out.csv", index=False)

    # Summary JSON
    summary = build_summary_json(results, datasets)
    with open(OUT_DIR / "biomarker_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: biomarker_summary.json")

    # Network summary
    if "md_edges" in results:
        net_df = network_summary(results["md_edges"].edge_stats)
        if not net_df.empty:
            net_df.to_csv(OUT_DIR / "network_summary_md.csv", index=False)

    # ── Final console report ─────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("Phase 4.1 — ASD Biomarker Discovery Results")
    print("=" * 62)
    print()

    for fn, res in results.items():
        label = fn.replace("_edges", "").upper()
        ds = datasets[fn]
        active_n = int(ds.active_mask.sum())
        print(f"  [{label}]")
        print(f"    Cross-site active edges:  {active_n}")
        print(f"    Tested edges:             {res.n_tested}")
        print(f"    Significant  q ≤ 0.05:   {res.n_q05}")
        print(f"    Exploratory  q ≤ 0.10:   {res.n_q10}")
        print(f"    Candidates (q≤0.10 or stability≥0.50): {len(res.candidate_biomarkers)}")
        if not np.isnan(res.permutation_observed_min_p):
            emp = float((res.permutation_null_min_p <= res.permutation_observed_min_p).mean())
            print(f"    Permutation empirical p:  {emp:.3f}  (obs min-p={res.permutation_observed_min_p:.2e})")
        else:
            print("    Permutation: no testable active edges")

        any_loso_stable = res.candidate_biomarkers["edge_id"].map(
            lambda eid: results[fn].site_robustness.loc[
                results[fn].site_robustness["edge_id"] == eid, "loso_stable"
            ].any()
        ).any() if not res.candidate_biomarkers.empty else False
        print(f"    Any candidate survives LOSO: {any_loso_stable}")
        print()

    print("  Top 10 MD edges by stability score:")
    print()
    if "md_edges" in results:
        top10 = results["md_edges"].top10
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            q_tag = "*" if row["q_B"] <= FDR_STRICT else ("~" if row["q_B"] <= FDR_EXPLORATORY else " ")
            active_tag = "[active]" if row["is_cross_site_active"] else "[sparse]"
            print(
                f"  {i:2d}. {row['edge_id']}  d={row['cohen_d']:.3f}  "
                f"p={row['p_B']:.3f}  q={row['q_B']:.3f}{q_tag}  "
                f"{row['direction']}  stab={row['stability_score']:.3f}  "
                f"{active_tag}"
            )
    print()
    print("  Legend: * q≤0.05   ~ q≤0.10   [active] cross-site ComBat-corrected")
    print()
    print(f"  Outputs: {OUT_DIR.relative_to(REPO_ROOT)}")
    print("=" * 62)
    print()
    print("  IMPORTANT: These are candidate research findings.")
    print("  They require external validation before any clinical interpretation.")
    print("=" * 62)


if __name__ == "__main__":
    main()
