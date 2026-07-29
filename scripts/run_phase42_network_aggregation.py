"""
NeuroFiber Phase 4.2 — Network-Pair Aggregation Runner

Aggregates ComBat-corrected MD/FA/length edges to 28 Schaefer 7-network pairs,
then performs ASD vs Control regression (Model B) across pairs with FDR correction
at 28 tests instead of 1009.

IMPORTANT: Statistical association analysis only.
Not a clinical diagnostic model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_AI_PIPELINE  = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_AI_PIPELINE))

from neurofiber.biomarkers.network_aggregation import (
    NETWORK_ORDER,
    NetworkAggResult,
    aggregate_to_network_pairs,
    run_network_aggregation,
    site_means_per_pair,
)

REPO_ROOT = _AI_PIPELINE.parent
OUT_DIR   = REPO_ROOT / "data" / "canonical_v1" / "biomarkers" / "network_pairs"
FEATURE_LABEL = {
    "md_edges":     "MD (mean diffusivity)",
    "fa_edges":     "FA (fractional anisotropy)",
    "length_edges": "Fiber length",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _pivot_to_grid(stats_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Pivot pair stats to a 7×7 symmetric grid (NETWORK_ORDER rows/columns).
    Missing pairs filled with NaN.
    """
    grid = pd.DataFrame(np.nan, index=NETWORK_ORDER, columns=NETWORK_ORDER)
    for _, row in stats_df.iterrows():
        ni, nj = row["net_i"], row["net_j"]
        v = row[value_col]
        grid.loc[ni, nj] = v
        grid.loc[nj, ni] = v     # symmetric
    return grid


# ── figure functions ───────────────────────────────────────────────────────────

def fig_cohen_d_heatmap(
    stats_df: pd.DataFrame,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
) -> None:
    """7×7 Cohen's d heatmap with significance markers."""
    grid_d = _pivot_to_grid(stats_df, "cohen_d")
    grid_q = _pivot_to_grid(stats_df, "q_B")

    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = np.nanmax(np.abs(grid_d.values))
    im = ax.imshow(grid_d.values.astype(float), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(7)); ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(7)); ax.set_yticklabels(NETWORK_ORDER, fontsize=9)

    for i in range(7):
        for j in range(7):
            d = grid_d.values[i, j]
            q = grid_q.values[i, j]
            if np.isnan(d):
                continue
            marker = ""
            if not np.isnan(q):
                if q <= 0.05:
                    marker = "**"
                elif q <= 0.10:
                    marker = "*"
            ax.text(j, i, f"{d:+.2f}{marker}", ha="center", va="center",
                    fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cohen's d  (ASD − Control)", fontsize=9)
    ax.set_title(f"Network-pair Cohen's d — {feature_label}\n(* q≤0.10  ** q≤0.05)", fontsize=10)
    plt.tight_layout()
    path = fig_dir / f"heatmap_cohen_d_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_qvalue_heatmap(
    stats_df: pd.DataFrame,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
) -> None:
    """7×7 FDR q-value heatmap (log10 scale)."""
    grid_q = _pivot_to_grid(stats_df, "q_B")
    log_q  = np.where(~np.isnan(grid_q.values), -np.log10(np.clip(grid_q.values, 1e-10, 1)), np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(log_q.astype(float), cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(7)); ax.set_yticklabels(NETWORK_ORDER, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("−log₁₀(q)", fontsize=9)
    ax.axhline(-0.5, color="none")  # layout spacer
    ax.set_title(f"Network-pair FDR q-value (−log₁₀) — {feature_label}", fontsize=10)
    plt.tight_layout()
    path = fig_dir / f"heatmap_qvalue_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_top_pairs_barplot(
    stats_df: pd.DataFrame,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
    top_n: int = 15,
) -> None:
    """Horizontal barplot of top network pairs sorted by |Cohen's d|."""
    df = stats_df.copy()
    df["abs_d"] = df["cohen_d"].abs()
    df_top = df.nlargest(top_n, "abs_d")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d73027" if d >= 0 else "#4575b4" for d in df_top["cohen_d"]]
    bars = ax.barh(range(len(df_top)), df_top["cohen_d"], color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)

    yticks = df_top["pair_label"].tolist()
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(yticks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d  (ASD − Control)", fontsize=9)
    ax.set_title(f"Top {top_n} network pairs — {feature_label}", fontsize=10)

    for idx, (bar, row) in enumerate(zip(bars, df_top.itertuples())):
        q = row.q_B
        if not np.isnan(q):
            marker = "**" if q <= 0.05 else ("*" if q <= 0.10 else "")
            if marker:
                x = bar.get_width() + (0.02 if bar.get_width() >= 0 else -0.02)
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        marker, va="center", fontsize=9, color="darkred")

    ax.text(0.99, 0.02, "* q≤0.10  ** q≤0.05",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="darkred")
    plt.tight_layout()
    path = fig_dir / f"top_pairs_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_top_pair_boxplots(
    result: NetworkAggResult,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
    top_n: int = 6,
) -> None:
    """ASD vs Control boxplots for the top-N pairs by |Cohen's d|."""
    stats_df = result.stats.copy()
    stats_df["abs_d"] = stats_df["cohen_d"].abs()
    top_pairs = stats_df.nlargest(top_n, "abs_d")["pair_label"].tolist()

    feat_df = result.feat_df
    dx = result.meta["diagnosis"].values

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, pair_label in zip(axes, top_pairs):
        y = feat_df[pair_label].values
        asd_vals  = y[dx == "ASD"]
        ctrl_vals = y[dx == "CONTROL"]
        asd_clean  = asd_vals[~np.isnan(asd_vals)]
        ctrl_clean = ctrl_vals[~np.isnan(ctrl_vals)]

        bp = ax.boxplot(
            [ctrl_clean, asd_clean],
            tick_labels=["Control", "ASD"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor("#4575b4")
        bp["boxes"][1].set_facecolor("#d73027")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_alpha(0.6)

        row = stats_df[stats_df["pair_label"] == pair_label].iloc[0]
        q_label = f"q={row['q_B']:.3f}" if not np.isnan(row["q_B"]) else ""
        ax.set_title(f"{pair_label}\nd={row['cohen_d']:+.3f}  {q_label}", fontsize=9)
        ax.set_ylabel(feature_label, fontsize=8)

    for ax in axes[len(top_pairs):]:
        ax.set_visible(False)

    fig.suptitle(f"Top {top_n} network pairs — {feature_label}", fontsize=11)
    plt.tight_layout()
    path = fig_dir / f"top_pairs_boxplots_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_loso_stability(
    result: NetworkAggResult,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
    top_n: int = 10,
) -> None:
    """
    LOSO stability plot: for top-N pairs by |Cohen's d|, show beta sign
    consistency across the 5 LOSO runs.
    """
    stats_df = result.stats.copy()
    stats_df["abs_d"] = stats_df["cohen_d"].abs()
    top_pairs = stats_df.nlargest(top_n, "abs_d")["pair_label"].tolist()

    loso_df = result.loso_df
    sites   = sorted(loso_df["left_out"].unique())

    data = []
    for pair in top_pairs:
        row_full = stats_df[stats_df["pair_label"] == pair].iloc[0]
        global_sign = np.sign(row_full["beta_B"]) if not np.isnan(row_full["beta_B"]) else 0
        sub = loso_df[loso_df["pair_label"] == pair]
        row_vals = []
        for site in sites:
            r = sub[sub["left_out"] == site]
            if len(r) == 0 or np.isnan(r["beta"].values[0]):
                row_vals.append(0)
            else:
                sign = np.sign(r["beta"].values[0])
                row_vals.append(1 if sign == global_sign else -1)
        data.append(row_vals)

    mat = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = matplotlib.colormaps["RdYlGn"].resampled(3)
    im = ax.imshow(mat, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")

    ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_pairs))); ax.set_yticklabels(top_pairs, fontsize=9)

    for i in range(len(top_pairs)):
        for j in range(len(sites)):
            v = mat[i, j]
            txt = "✓" if v == 1 else ("✗" if v == -1 else "–")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color="black")

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.03, pad=0.02)
    cbar.ax.set_yticklabels(["Flipped", "Missing", "Same sign"], fontsize=8)
    ax.set_title(f"LOSO stability (top {top_n} pairs by |d|) — {feature_label}", fontsize=10)
    plt.tight_layout()
    path = fig_dir / f"loso_stability_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_coverage_heatmap(
    result: NetworkAggResult,
    feature_label: str,
    fig_dir: Path,
    feature_key: str,
) -> None:
    """7×7 mean edge-coverage heatmap (fraction of active edges non-NaN per subject)."""
    cover_mean = result.cover_df.mean(axis=0)
    pair_labels = result.cover_df.columns.tolist()

    grid = pd.DataFrame(np.nan, index=NETWORK_ORDER, columns=NETWORK_ORDER)
    pair_info_by_label = {p.label: p for p in result.pair_index}
    for label in pair_labels:
        p = pair_info_by_label[label]
        grid.loc[p.net_i, p.net_j] = cover_mean[label]
        grid.loc[p.net_j, p.net_i] = cover_mean[label]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid.values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(7)); ax.set_yticklabels(NETWORK_ORDER, fontsize=9)

    for i in range(7):
        for j in range(7):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean edge coverage (fraction non-NaN)", fontsize=9)
    ax.set_title(f"Edge coverage per network pair — {feature_label}", fontsize=10)
    plt.tight_layout()
    path = fig_dir / f"coverage_{feature_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── summary printing ───────────────────────────────────────────────────────────

def _print_console_summary(results: dict[str, NetworkAggResult]) -> None:
    print("\n" + "=" * 65)
    print("PHASE 4.2  NETWORK-PAIR AGGREGATION — SUMMARY")
    print("=" * 65)
    print(f"  NOTE: Statistical association analysis. Not a diagnostic model.")
    print()

    for fn, r in results.items():
        flabel = FEATURE_LABEL.get(fn, fn)
        print(f"  {flabel}")
        print(f"    Pairs tested : {r.n_pairs_tested}")
        print(f"    q ≤ 0.05     : {r.n_q05}")
        print(f"    q ≤ 0.10     : {r.n_q10}")

        df = r.stats.copy()
        df["abs_d"] = df["cohen_d"].abs()
        top = df.nlargest(5, "abs_d")
        print()
        print(f"    Top 5 pairs by |Cohen's d|:")
        print(f"    {'Pair':<22} {'d':>7} {'p_B':>8} {'q_B':>8} {'Direction':<14} {'LOSO?'}")
        print(f"    {'-'*22} {'-'*7} {'-'*8} {'-'*8} {'-'*14} {'-'*5}")
        for _, row in top.iterrows():
            loso = row.get("loso_stable", np.nan)
            loso_s = "Yes" if loso is True else ("No" if loso is False else "—")
            d_str   = f"{row['cohen_d']:+.3f}"
            p_str   = f"{row['p_B']:.4f}" if not np.isnan(row["p_B"]) else "  nan"
            q_str   = f"{row['q_B']:.3f}"  if not np.isnan(row["q_B"]) else "  nan"
            print(f"    {row['pair_label']:<22} {d_str:>7} {p_str:>8} {q_str:>8} {row['direction']:<14} {loso_s}")
        print()

    print("=" * 65 + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4.2: Aggregate edges to 28 network pairs and test ASD associations."
    )
    p.add_argument(
        "--feature",
        nargs="+",
        default=["md_edges", "fa_edges", "length_edges"],
        choices=["md_edges", "fa_edges", "length_edges"],
        metavar="FEATURE",
        help="Feature type(s) to process (default: all three)",
    )
    p.add_argument(
        "--no-combat",
        action="store_true",
        help="Skip ComBat; use raw unharmonized edge values (for debugging only)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_phase42")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("Repo root : %s", REPO_ROOT)
    log.info("Output dir: %s", OUT_DIR)
    log.info("Features  : %s", args.feature)

    # ── run analysis ─────────────────────────────────────────────────────────
    results = run_network_aggregation(
        repo_root=REPO_ROOT,
        feature_names=args.feature,
        use_combat=not args.no_combat,
    )

    # ── save CSVs ─────────────────────────────────────────────────────────────
    summary_rows = []
    for fn, r in results.items():
        key = fn.replace("_edges", "")
        r.stats.to_csv(OUT_DIR / f"{key}_pair_stats.csv", index=False)
        r.loso_df.to_csv(OUT_DIR / f"{key}_pair_loso.csv", index=False)
        r.loso_summary.to_csv(OUT_DIR / f"{key}_pair_loso_summary.csv", index=False)
        r.feat_df.to_csv(OUT_DIR / f"{key}_pair_features_mean.csv")
        r.feat_df_median.to_csv(OUT_DIR / f"{key}_pair_features_median.csv")
        r.cover_df.to_csv(OUT_DIR / f"{key}_pair_coverage.csv")
        log.info("Saved CSVs for %s", fn)

        summary_rows.append({
            "feature": fn,
            "n_pairs_tested": r.n_pairs_tested,
            "n_q05": r.n_q05,
            "n_q10": r.n_q10,
        })

    # ── save JSON summary ─────────────────────────────────────────────────────
    summary = {"phase": "4.2", "features": {}}
    for fn, r in results.items():
        df = r.stats.copy()
        df["abs_d"] = df["cohen_d"].abs()
        top5 = df.nlargest(5, "abs_d")
        summary["features"][fn] = {
            "n_pairs_tested": r.n_pairs_tested,
            "n_q05":          r.n_q05,
            "n_q10":          r.n_q10,
            "top5_pairs":     top5["pair_label"].tolist(),
        }
    with open(OUT_DIR / "phase42_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── figures ───────────────────────────────────────────────────────────────
    for fn, r in results.items():
        key    = fn.replace("_edges", "")
        flabel = FEATURE_LABEL.get(fn, fn)
        df     = r.stats

        log.info("Generating figures for %s …", fn)

        fig_cohen_d_heatmap(df, flabel, fig_dir, key)
        fig_qvalue_heatmap(df, flabel, fig_dir, key)
        fig_top_pairs_barplot(df, flabel, fig_dir, key)
        fig_top_pair_boxplots(r, flabel, fig_dir, key)
        fig_loso_stability(r, flabel, fig_dir, key)
        fig_coverage_heatmap(r, flabel, fig_dir, key)

        log.info("  Saved 6 figures for %s", fn)

    _print_console_summary(results)

    log.info("Phase 4.2 complete. Outputs at: %s", OUT_DIR)


if __name__ == "__main__":
    main()
