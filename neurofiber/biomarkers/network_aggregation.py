"""
NeuroFiber Phase 4.2 — Network-Pair Aggregation

Aggregates ComBat-corrected edge features by Schaefer 7-network pairs,
producing 28 subject-level summary features per feature type (MD/FA/length).

Network aggregation reduces 688 sparse active edges to 28 dense network-pair
features, substantially improving statistical power after FDR correction
(28 tests vs 1009 in Phase 4.1).

Per subject per network pair:
  - mean of non-NaN active edge values
  - median of non-NaN active edge values
  - edge_coverage: fraction of pair's active edges with non-NaN value

Regression model (same as Phase 4.1 Model B):
  y_pair = β₀ + β₁·diagnosis + β₂·age + β₃·sex + Σβₖ·siteₖ + ε

Scientific context
──────────────────
Statistical association analysis only.
Not a clinical diagnostic model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .biomarker_discovery import (
    BiomarkerDataset,
    SCHAEFER_NETWORKS,
    BALANCED_SITES,
    FDR_STRICT,
    FDR_EXPLORATORY,
    MIN_N_PER_GROUP,
    build_edge_index_table,
    roi_to_network,
    cohen_d,
    _site_dummies,
    _fit_model,
    load_dataset,
)

log = logging.getLogger("network_aggregation")

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


# ── pair index ────────────────────────────────────────────────────────────────

@dataclass
class NetworkPairInfo:
    """Metadata for one network pair."""
    label: str             # e.g. "Cont_Default"
    net_i: str             # alphabetically first
    net_j: str             # alphabetically second
    within: bool           # net_i == net_j
    n_active_edges: int
    active_edge_positions: List[int]   # positions in ds.edge_cols


def build_pair_index(ds: BiomarkerDataset) -> List[NetworkPairInfo]:
    """
    Build the 28 network-pair records, each with the list of active-edge
    column positions that connect the two networks.
    """
    edge_index_table = build_edge_index_table()
    edge_to_col_pos = {ec: pos for pos, ec in enumerate(ds.edge_cols)}

    # Map each edge_idx → (net_i, net_j) using canonical sorted pair
    table = edge_index_table.copy()
    table["net_i"] = table["roi_i"].apply(roi_to_network)
    table["net_j"] = table["roi_j"].apply(roi_to_network)
    table["pair"] = table.apply(
        lambda r: "_".join(sorted([r.net_i, r.net_j])), axis=1
    )

    # Restrict to active edges only
    active_col_positions = np.where(ds.active_mask)[0]
    active_edge_names = {ds.edge_cols[p] for p in active_col_positions}
    active_edge_idx_set = {int(ec.split("_")[1]) - 1 for ec in active_edge_names}

    active_table = table[table["edge_idx"].isin(active_edge_idx_set)].copy()

    pairs: Dict[str, List[int]] = {}
    for _, row in active_table.iterrows():
        pair_label = row["pair"]
        col_pos = edge_to_col_pos.get(f"edge_{int(row['edge_idx'])+1:04d}")
        if col_pos is not None:
            pairs.setdefault(pair_label, []).append(col_pos)

    result: List[NetworkPairInfo] = []
    for label in sorted(pairs.keys()):
        nets = label.split("_")
        # canonical split: last network is net_j (they're sorted alphabetically)
        # Need to recover net_i, net_j from the sorted pair label
        # The pair label is '_'.join(sorted([net_a, net_b]))
        # Need to find which two networks produced this label
        net_i, net_j = _recover_networks_from_label(label)
        result.append(NetworkPairInfo(
            label=label,
            net_i=net_i,
            net_j=net_j,
            within=(net_i == net_j),
            n_active_edges=len(pairs[label]),
            active_edge_positions=sorted(pairs[label]),
        ))

    return result


def _recover_networks_from_label(label: str) -> Tuple[str, str]:
    """Recover (net_i, net_j) from a sorted pair label like 'Cont_Default'."""
    # Network names can contain underscores ('SalVentAttn')
    # Split on first matching pair from NETWORK_ORDER
    for net in NETWORK_ORDER:
        if label.startswith(net + "_"):
            rest = label[len(net) + 1:]
            if rest in NETWORK_ORDER:
                return net, rest
    # Fallback: try all pairs
    for n1 in NETWORK_ORDER:
        for n2 in NETWORK_ORDER:
            if "_".join(sorted([n1, n2])) == label:
                return n1, n2
    raise ValueError(f"Cannot parse pair label: {label!r}")


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate_to_network_pairs(
    ds: BiomarkerDataset,
    pair_index: List[NetworkPairInfo],
    use_combat: bool = True,
    agg: str = "mean",   # "mean" | "median"
    min_edges: int = 1,  # minimum non-NaN edges for a valid subject-pair value
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate edge features to network-pair level.

    Returns
    -------
    feat_df   : (n_subjects × n_pairs) float DataFrame — aggregated values
    cover_df  : (n_subjects × n_pairs) float DataFrame — edge coverage [0,1]
    """
    X = ds.X_combat if use_combat else ds.X_raw
    n_subjects = X.shape[0]
    pair_labels = [p.label for p in pair_index]

    feat_mat  = np.full((n_subjects, len(pair_index)), np.nan)
    cover_mat = np.full((n_subjects, len(pair_index)), np.nan)

    for k, pair in enumerate(pair_index):
        positions = pair.active_edge_positions
        if not positions:
            continue
        X_pair = X[:, positions]  # (n_subjects, n_pair_edges)
        valid_per_subj = (~np.isnan(X_pair)).sum(axis=1)   # (n_subjects,)
        cover_mat[:, k] = valid_per_subj / len(positions)

        for s in range(n_subjects):
            vals = X_pair[s, :]
            good = vals[~np.isnan(vals)]
            if len(good) < min_edges:
                continue
            feat_mat[s, k] = float(good.mean() if agg == "mean" else np.median(good))

    feat_df  = pd.DataFrame(feat_mat,  columns=pair_labels, index=ds.meta.index)
    cover_df = pd.DataFrame(cover_mat, columns=pair_labels, index=ds.meta.index)
    return feat_df, cover_df


# ── per-pair statistics ───────────────────────────────────────────────────────

@dataclass
class PairStats:
    label: str
    net_i: str
    net_j: str
    within: bool
    n_active_edges: int
    n_valid_subjects: int
    n_asd: int
    n_control: int
    mean_asd: float
    mean_control: float
    std_asd: float
    std_control: float
    mean_coverage: float
    cohen_d: float
    direction: str
    beta_B: float
    p_B: float
    se_B: float
    q_B: float = np.nan
    q_B_strict: bool = False
    q_B_exploratory: bool = False


def compute_pair_stats(
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    pair_index: List[NetworkPairInfo],
    min_n_per_group: int = MIN_N_PER_GROUP,
) -> List[PairStats]:
    """Fit Model B for each of the 28 network pairs."""
    meta_arr = meta.copy()
    sites  = meta_arr["site"].values
    ages   = meta_arr["age"].values.astype(float)
    sexes  = (meta_arr["sex"].values == "M").astype(float)
    dx     = (meta_arr["diagnosis"].values == "ASD").astype(float)
    age_c  = (ages - np.nanmean(ages)) / (np.nanstd(ages) + 1e-9)
    site_dum = _site_dummies(sites, reference="BNI")

    results: List[PairStats] = []
    for pair in pair_index:
        y_full = feat_df[pair.label].values

        valid = ~np.isnan(y_full)
        n_valid = int(valid.sum())

        if n_valid < 10:
            continue

        y   = y_full[valid]
        dx_v = dx[valid]; age_v = age_c[valid]; sex_v = sexes[valid]
        sd_v = site_dum[valid]

        asd_m  = dx_v == 1
        ctrl_m = dx_v == 0
        n_asd  = int(asd_m.sum())
        n_ctrl = int(ctrl_m.sum())
        if n_asd < min_n_per_group or n_ctrl < min_n_per_group:
            continue

        y_asd  = y[asd_m];  y_ctrl = y[ctrl_m]
        m_asd  = float(y_asd.mean()); m_ctrl = float(y_ctrl.mean())
        s_asd  = float(y_asd.std(ddof=1)) if len(y_asd) > 1 else np.nan
        s_ctrl = float(y_ctrl.std(ddof=1)) if len(y_ctrl) > 1 else np.nan
        d      = cohen_d(y_asd, y_ctrl)
        direction = "ASD_higher" if m_asd >= m_ctrl else "ASD_lower"

        X_design = sm.add_constant(np.column_stack([dx_v, age_v, sex_v, sd_v]))
        beta, p, se = _fit_model(y, X_design)

        results.append(PairStats(
            label=pair.label,
            net_i=pair.net_i, net_j=pair.net_j,
            within=pair.within,
            n_active_edges=pair.n_active_edges,
            n_valid_subjects=n_valid,
            n_asd=n_asd, n_control=n_ctrl,
            mean_asd=m_asd, mean_control=m_ctrl,
            std_asd=s_asd, std_control=s_ctrl,
            mean_coverage=float(feat_df[pair.label].notna().mean()),
            cohen_d=d, direction=direction,
            beta_B=beta, p_B=p, se_B=se,
        ))

    return results


def apply_fdr_pairs(stats: List[PairStats]) -> List[PairStats]:
    """BH FDR across all 28 network-pair tests."""
    pvals = np.array([s.p_B for s in stats])
    valid = ~np.isnan(pvals)
    q = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        _, q_vals, _, _ = multipletests(pvals[valid], alpha=FDR_STRICT, method="fdr_bh")
        q[valid] = q_vals
    for i, s in enumerate(stats):
        s.q_B = float(q[i])
        s.q_B_strict      = bool(q[i] <= FDR_STRICT)
        s.q_B_exploratory = bool(q[i] <= FDR_EXPLORATORY)
    return stats


# ── LOSO ─────────────────────────────────────────────────────────────────────

def loso_network_pairs(
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    pair_index: List[NetworkPairInfo],
) -> pd.DataFrame:
    """
    Leave-one-site-out regression for each network pair.

    Returns DataFrame with columns: pair_label, left_out, beta, p_value
    """
    sites_full = meta["site"].values
    ages       = meta["age"].values.astype(float)
    sexes      = (meta["sex"].values == "M").astype(float)
    dx         = (meta["diagnosis"].values == "ASD").astype(float)
    age_c      = (ages - np.nanmean(ages)) / (np.nanstd(ages) + 1e-9)

    all_sites = sorted(set(sites_full))
    rows: List[dict] = []

    for pair in pair_index:
        y_full = feat_df[pair.label].values

        for leave_out in all_sites:
            keep = (sites_full != leave_out) & ~np.isnan(y_full)
            n_keep = int(keep.sum())
            if n_keep < 10:
                rows.append({"pair_label": pair.label, "left_out": leave_out,
                             "beta": np.nan, "p_value": np.nan})
                continue

            y_s  = y_full[keep]
            dx_s = dx[keep]
            if (dx_s == 1).sum() < MIN_N_PER_GROUP or (dx_s == 0).sum() < MIN_N_PER_GROUP:
                rows.append({"pair_label": pair.label, "left_out": leave_out,
                             "beta": np.nan, "p_value": np.nan})
                continue

            sd_s = _site_dummies(sites_full[keep], reference="BNI")
            X_s  = sm.add_constant(np.column_stack(
                [dx_s, age_c[keep], sexes[keep], sd_s]
            ))
            beta, p, _ = _fit_model(y_s, X_s)
            rows.append({"pair_label": pair.label, "left_out": leave_out,
                         "beta": beta, "p_value": p})

    return pd.DataFrame(rows)


def loso_stability_summary(
    stats: List[PairStats],
    loso_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize LOSO stability per pair.

    A pair is LOSO-stable if ≥ 4/5 LOSO betas have the same sign as the full beta.
    """
    rows: List[dict] = []
    for s in stats:
        sub = loso_df[loso_df["pair_label"] == s.label]
        global_sign = np.sign(s.beta_B) if not np.isnan(s.beta_B) else 0
        n_same = 0
        n_run  = 0
        for _, row in sub.iterrows():
            if not np.isnan(row["beta"]):
                n_run += 1
                if np.sign(row["beta"]) == global_sign and global_sign != 0:
                    n_same += 1
        nyu2_row = sub[sub["left_out"] == "NYU_2"]
        nyu2_p   = float(nyu2_row["p_value"].values[0]) if len(nyu2_row) > 0 else np.nan
        nyu2_ok  = (
            len(nyu2_row) > 0 and
            not np.isnan(nyu2_row["beta"].values[0]) and
            global_sign != 0 and
            np.sign(nyu2_row["beta"].values[0]) == global_sign
        )
        rows.append({
            "pair_label":          s.label,
            "loso_n_same_sign":    n_same,
            "loso_n_run":          n_run,
            "loso_stable":         n_same >= 4 and n_run >= 4,
            "nyu2_removed_p":      nyu2_p,
            "nyu2_removed_same_dir": nyu2_ok,
        })
    return pd.DataFrame(rows)


# ── DataFrames ────────────────────────────────────────────────────────────────

def stats_to_df(
    stats: List[PairStats],
    loso_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for s in stats:
        row = {
            "pair_label":       s.label,
            "net_i":            s.net_i,
            "net_j":            s.net_j,
            "within_network":   s.within,
            "n_active_edges":   s.n_active_edges,
            "n_valid_subjects": s.n_valid_subjects,
            "n_asd":            s.n_asd,
            "n_control":        s.n_control,
            "mean_asd":         s.mean_asd,
            "mean_control":     s.mean_control,
            "std_asd":          s.std_asd,
            "std_control":      s.std_control,
            "mean_coverage":    s.mean_coverage,
            "cohen_d":          s.cohen_d,
            "direction":        s.direction,
            "beta_B":           s.beta_B,
            "p_B":              s.p_B,
            "se_B":             s.se_B,
            "q_B":              s.q_B,
            "q_B_strict":       s.q_B_strict,
            "q_B_exploratory":  s.q_B_exploratory,
        }
        if loso_summary is not None:
            sub = loso_summary[loso_summary["pair_label"] == s.label]
            if len(sub) > 0:
                row["loso_stable"]          = bool(sub["loso_stable"].values[0])
                row["loso_n_same_sign"]     = int(sub["loso_n_same_sign"].values[0])
                row["nyu2_removed_p"]       = float(sub["nyu2_removed_p"].values[0])
                row["nyu2_removed_same_dir"] = bool(sub["nyu2_removed_same_dir"].values[0])
        rows.append(row)
    return pd.DataFrame(rows)


# ── site-level mean per pair ──────────────────────────────────────────────────

def site_means_per_pair(
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    pair_labels: List[str],
) -> pd.DataFrame:
    """
    Per-site mean aggregate value for each network pair.
    Returns DataFrame: (site × pair_label).
    """
    sites = meta["site"].values
    rows: List[dict] = []
    for site in sorted(set(sites)):
        mask = sites == site
        row = {"site": site}
        for label in pair_labels:
            vals = feat_df.loc[mask, label].dropna()
            row[label] = float(vals.mean()) if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("site")


# ── main entry point ──────────────────────────────────────────────────────────

@dataclass
class NetworkAggResult:
    feature_name: str
    pair_index: List[NetworkPairInfo]
    meta: pd.DataFrame           # subject metadata (aligned to feat_df rows)
    feat_df: pd.DataFrame        # subjects × pairs (mean aggregate)
    feat_df_median: pd.DataFrame # subjects × pairs (median aggregate)
    cover_df: pd.DataFrame       # subjects × pairs (coverage)
    stats: pd.DataFrame          # pair-level statistics
    loso_df: pd.DataFrame        # LOSO raw results
    loso_summary: pd.DataFrame   # LOSO stability per pair
    n_pairs_tested: int
    n_q05: int
    n_q10: int


def run_network_aggregation(
    repo_root: Path,
    feature_names: Optional[List[str]] = None,
    use_combat: bool = True,
) -> Dict[str, NetworkAggResult]:
    """
    Full Phase 4.2 pipeline for all feature types.
    Returns dict: feature_name → NetworkAggResult.
    """
    if feature_names is None:
        feature_names = ["md_edges", "fa_edges", "length_edges"]

    results: Dict[str, NetworkAggResult] = {}

    for fn in feature_names:
        log.info("=" * 60)
        log.info("Feature: %s", fn)

        ds = load_dataset(repo_root, fn)
        pair_index = build_pair_index(ds)
        log.info("  Network pairs: %d  Active edges: %d", len(pair_index), ds.active_mask.sum())

        feat_df, cover_df = aggregate_to_network_pairs(
            ds, pair_index, use_combat=use_combat, agg="mean"
        )
        feat_df_med, _ = aggregate_to_network_pairs(
            ds, pair_index, use_combat=use_combat, agg="median"
        )

        meta = ds.meta.copy()
        pair_stats = compute_pair_stats(feat_df, meta, pair_index)
        pair_stats = apply_fdr_pairs(pair_stats)

        loso_df      = loso_network_pairs(feat_df, meta, pair_index)
        loso_summary = loso_stability_summary(pair_stats, loso_df)
        stats_df     = stats_to_df(pair_stats, loso_summary)

        n_q05 = int(stats_df["q_B_strict"].sum())
        n_q10 = int(stats_df["q_B_exploratory"].sum())
        log.info("  Tested: %d pairs  q≤0.05: %d  q≤0.10: %d", len(pair_stats), n_q05, n_q10)

        results[fn] = NetworkAggResult(
            feature_name=fn,
            pair_index=pair_index,
            meta=meta,
            feat_df=feat_df,
            feat_df_median=feat_df_med,
            cover_df=cover_df,
            stats=stats_df,
            loso_df=loso_df,
            loso_summary=loso_summary,
            n_pairs_tested=len(pair_stats),
            n_q05=n_q05,
            n_q10=n_q10,
        )

    return results
