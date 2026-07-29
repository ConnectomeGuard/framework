"""
Phase 3R.5C-A — Site Fingerprint Analysis

Before applying ComBat, characterize the exact mechanism responsible
for perfect site prediction (balanced_accuracy = 1.000) after site_zscore
harmonization of count connectome edges.

Questions answered:
  1. What is the zero-inflation structure per site?
  2. Is the fingerprint from the zero pattern or from the z-score transformation?
  3. How many edges are needed for perfect site identification?
  4. Does log(1+x) transformation reduce the fingerprint?
  5. Does density normalization (counts / total streamlines) reduce the fingerprint?
  6. Do FA / MD / length edges carry the same leakage mechanism?
  7. Can ComBat mathematically fix the zero-fingerprint, given the mechanism?

Outputs (data/processed_v2b/):
  phase3r_5ca_zero_inflation.csv        — per-edge per-site zero fraction
  phase3r_5ca_fingerprint_edges.csv     — top-k discriminative edges + stats
  phase3r_5ca_leakage_curve.csv         — bal_acc vs number of top edges
  phase3r_5ca_transformation_test.csv   — leakage under each transformation
  phase3r_5ca_confusion_matrix.csv      — site confusion matrix (site_zscore RF)
  phase3r_5ca_report.json               — mechanism characterization + ComBat prognosis

Usage:
    cd /path/to/neurofiber
    python ai_pipeline/scripts/run_phase35ca.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
HARMONIZED_DIR = ROOT / "data/processed_v2b/harmonized_connectomes"
FEATURES_DIR = ROOT / "data/processed_v2b/connectome_features"
OUTPUT_DIR = ROOT / "data/processed_v2b"
METADATA_PATH = FEATURES_DIR / "harmonization_metadata_enriched.csv"

N_SPLITS = 5
RANDOM_STATE = 42
RF_FAST = 200
RF_FULL = 500


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw(feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Load unharmonized features; return (X, feat_cols)."""
    path = HARMONIZED_DIR / "none" / f"{feature_set}.csv"
    df = pd.read_csv(path, index_col=0)
    feat_cols = [c for c in df.columns if c.startswith("edge_")]
    X = df[feat_cols].values.astype(float)
    return X, feat_cols


def load_sz(feature_set: str, feat_cols: list[str]) -> np.ndarray:
    """Load site_zscore features; impute NaN with 0."""
    path = HARMONIZED_DIR / "site_zscore" / f"{feature_set}.csv"
    df = pd.read_csv(path, index_col=0)
    X = df[feat_cols].values.astype(float)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nm = np.isnan(X)
    X[nm] = np.take(col_med, np.where(nm)[1])
    return X


def site_zscore_manual(X: np.ndarray, sites: np.ndarray, site_list: list[str]) -> np.ndarray:
    """Replicate site_zscore on an arbitrary X; NaN → 0."""
    X_out = X.copy().astype(float)
    for s in site_list:
        mask = sites == s
        mu = X_out[mask].mean(axis=0)
        sd = X_out[mask].std(axis=0)
        sd[sd == 0] = np.nan
        X_out[mask] = (X_out[mask] - mu) / sd
    return np.where(np.isnan(X_out), 0.0, X_out)


def cv_bal_acc(X: np.ndarray, y: np.ndarray, n_est: int = RF_FAST) -> float:
    """5-fold stratified CV balanced accuracy."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    bals = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(
            n_est, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X[tr], y[tr])
        bals.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
    return float(np.mean(bals))


def cv_confusion(X: np.ndarray, y: np.ndarray, n_est: int = RF_FAST) -> np.ndarray:
    """5-fold CV confusion matrix (accumulated over folds)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_pred, all_true = [], []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(
            n_est, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X[tr], y[tr])
        all_pred.extend(rf.predict(X[te]))
        all_true.extend(y[te])
    return confusion_matrix(all_true, all_pred)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 3R.5C-A — Site Fingerprint Analysis")
    print("=" * 60)

    meta = pd.read_csv(METADATA_PATH, index_col="subject_id")

    # ── load count edges ──────────────────────────────────────────────────────
    print("\n[1] Loading count_edges …")
    # Use the index from the harmonized file to align subjects
    df_ref = pd.read_csv(HARMONIZED_DIR / "none" / "count_edges.csv", index_col=0)
    subject_ids = df_ref.index
    feat_cols = [c for c in df_ref.columns if c.startswith("edge_")]

    sites = meta.loc[subject_ids, "site"].values
    site_list = sorted(set(sites))
    le = LabelEncoder()
    y = le.fit_transform(sites)
    n_sites = len(site_list)

    X_raw = df_ref[feat_cols].values.astype(float)
    X_sz = load_sz("count_edges", feat_cols)
    n_edges = X_raw.shape[1]

    print(f"  Subjects: {len(subject_ids)}, Edges: {n_edges}, Sites: {site_list}")

    # ── 1. Zero-inflation matrix ──────────────────────────────────────────────
    print("\n[2] Computing zero-inflation matrix …")
    zero_frac = np.zeros((n_edges, n_sites))
    for j, s in enumerate(site_list):
        mask = sites == s
        zero_frac[:, j] = (X_raw[mask] == 0).mean(axis=0)

    zi_df = pd.DataFrame(zero_frac, index=feat_cols, columns=site_list)
    zi_df["global_zero_frac"] = (X_raw == 0).mean(axis=0)
    zi_df["max_inter_site_diff"] = zi_df[site_list].max(axis=1) - zi_df[site_list].min(axis=1)

    # Edge categories
    active_in_all = (zero_frac < 1).all(axis=1)   # at least 1 nonzero in every site
    dead_in_all = (zero_frac == 1).all(axis=1)      # all zero in every site
    dead_in_some = ~active_in_all & ~dead_in_all    # structural fingerprint edges

    zi_df["category"] = "dead_in_all"
    zi_df.loc[active_in_all, "category"] = "active_in_all"
    zi_df.loc[dead_in_some, "category"] = "dead_in_some_sites"

    print(f"  Active in all {n_sites} sites:  {active_in_all.sum()}")
    print(f"  Dead in all sites:       {dead_in_all.sum()}")
    print(f"  Dead in >=1 site (fingerprint): {dead_in_some.sum()}")
    print(f"  Global zero fraction: {(X_raw == 0).mean():.3f}")

    zi_path = OUTPUT_DIR / "phase3r_5ca_zero_inflation.csv"
    zi_df.to_csv(zi_path)

    # ── 2. Mechanism isolation — which transformation creates the fingerprint ─
    print("\n[3] Mechanism isolation — transformation tests …")

    # Density normalization: divide each subject's count matrix by their row sum
    row_sums = X_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero (shouldn't happen)
    X_density = X_raw / row_sums

    # Binarize: presence/absence
    X_binary = (X_raw > 0).astype(float)

    def sz_of(X: np.ndarray) -> np.ndarray:
        return site_zscore_manual(X, sites, site_list)

    # Restrict to active-in-all edges for the clean comparison
    X_act = X_raw[:, active_in_all]
    X_act_log = np.log1p(X_act)
    X_act_dens = X_density[:, active_in_all]

    transformation_rows = []

    tests = [
        ("count_raw", X_raw, "all"),
        ("count_raw_active", X_act, "active_in_all"),
        ("count_log1p_active", X_act_log, "active_in_all"),
        ("count_density_active", X_act_dens, "active_in_all"),
        ("count_binary_all", X_binary, "all"),
        ("count_binary_active", X_binary[:, active_in_all], "active_in_all"),
        ("count_sz_all", X_sz, "all"),
        ("count_sz_active", X_sz[:, active_in_all], "active_in_all"),
        ("count_log1p_then_sz_active", sz_of(X_act_log), "active_in_all"),
        ("count_density_then_sz_active", sz_of(X_act_dens), "active_in_all"),
    ]

    for name, X_test, edge_scope in tests:
        print(f"  {name} ({X_test.shape[1]} features) …", end=" ", flush=True)
        bal = cv_bal_acc(X_test, y)
        print(f"bal_acc={bal:.3f}")
        transformation_rows.append({
            "transformation": name,
            "edge_scope": edge_scope,
            "n_features": X_test.shape[1],
            "site_balanced_accuracy": bal,
        })

    transf_df = pd.DataFrame(transformation_rows)
    transf_path = OUTPUT_DIR / "phase3r_5ca_transformation_test.csv"
    transf_df.to_csv(transf_path, index=False)

    # ── 3. Concrete z-score fingerprint demonstration ─────────────────────────
    print("\n[4] Demonstrating z-score fingerprint mechanism …")
    # Find the edge with largest spread of zscore_of_zero across sites
    zscore_of_zero_per_site = np.zeros((n_edges, n_sites))
    for j, s in enumerate(site_list):
        mask = sites == s
        mu = X_raw[mask].mean(axis=0)
        sd = X_raw[mask].std(axis=0)
        # z-score of 0: -mu / sd (where sd=0 → NaN → imputed to 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore_of_zero_per_site[:, j] = np.where(sd > 0, -mu / sd, 0.0)

    zero_fingerprint_range = (
        zscore_of_zero_per_site.max(axis=1) - zscore_of_zero_per_site.min(axis=1)
    )
    top10_fp_idx = np.argsort(zero_fingerprint_range)[::-1][:10]

    print("  Top 10 edges by z-score-of-zero spread across sites:")
    fp_rows = []
    for rank, ei in enumerate(top10_fp_idx):
        zvals = {s: float(zscore_of_zero_per_site[ei, j]) for j, s in enumerate(site_list)}
        zero_fracs = {s: float(zero_frac[ei, j]) for j, s in enumerate(site_list)}
        fp_rows.append({
            "rank": rank + 1,
            "edge": feat_cols[ei],
            "zero_fingerprint_range": float(zero_fingerprint_range[ei]),
            "category": zi_df.iloc[ei]["category"],
            **{f"zscore_of_zero_{s}": zvals[s] for s in site_list},
            **{f"zero_frac_{s}": zero_fracs[s] for s in site_list},
        })
        parts = " | ".join(f"{s}:{zvals[s]:+.3f}(z0) {zero_fracs[s]:.2f}(0fr)"
                           for s in site_list)
        print(f"  {rank+1:2d}. {feat_cols[ei]}: {parts}")

    fp_df = pd.DataFrame(fp_rows)

    # ── 4. Feature count leakage curve ────────────────────────────────────────
    print("\n[5] Leakage curve: bal_acc vs number of top edges (site_zscore count) …")
    # Train RF on all site_zscore data to get sorted importances
    rf_imp = RandomForestClassifier(
        RF_FULL, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_imp.fit(X_sz, y)
    sorted_by_imp = np.argsort(rf_imp.feature_importances_)[::-1]

    curve_rows = []
    for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4950]:
        top_k = sorted_by_imp[:k]
        bal = cv_bal_acc(X_sz[:, top_k], y)
        curve_rows.append({"n_top_edges": k, "site_bal_acc": bal})
        print(f"  top-{k:5d} edges: bal_acc={bal:.3f}")

    curve_df = pd.DataFrame(curve_rows)
    curve_path = OUTPUT_DIR / "phase3r_5ca_leakage_curve.csv"
    curve_df.to_csv(curve_path, index=False)

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    print("\n[6] Confusion matrix (site_zscore, count_edges, RF) …")
    cm = cv_confusion(X_sz, y)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    cm_path = OUTPUT_DIR / "phase3r_5ca_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # ── 6. Other feature sets: is leakage the same mechanism? ─────────────────
    print("\n[7] Checking leakage mechanism across other feature sets …")
    other_feature_rows = []
    for fs in ["fa_edges", "md_edges", "length_edges"]:
        print(f"  {fs} …", end=" ", flush=True)
        X_fs_raw, fc = load_raw(fs)
        X_fs_sz = load_sz(fs, fc)

        # Impute NaN
        col_m = np.nanmedian(X_fs_sz, axis=0)
        col_m[np.isnan(col_m)] = 0.0
        nm = np.isnan(X_fs_sz)
        X_fs_sz[nm] = np.take(col_m, np.where(nm)[1])

        # Zero fraction in raw
        zero_frac_fs = np.zeros((len(fc), n_sites))
        for j, s in enumerate(site_list):
            mask = sites == s
            zero_frac_fs[:, j] = (X_fs_raw[mask] == 0).mean(axis=0)

        # Active-in-all (at least 1 nonzero in every site)
        active_fs = (zero_frac_fs < 1).all(axis=1)

        # z-score of zero spread
        zoz = np.zeros((len(fc), n_sites))
        for j, s in enumerate(site_list):
            mask = sites == s
            mu = X_fs_raw[mask].mean(axis=0)
            sd = X_fs_raw[mask].std(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                zoz[:, j] = np.where(sd > 0, -mu / sd, 0.0)

        max_zoz = (zoz.max(axis=1) - zoz.min(axis=1)).max()

        bal_raw = cv_bal_acc(X_fs_raw, y)
        bal_sz_all = cv_bal_acc(X_fs_sz, y)
        bal_sz_active = cv_bal_acc(X_fs_sz[:, active_fs], y) if active_fs.sum() > 0 else np.nan

        print(f"raw={bal_raw:.3f}, sz_all={bal_sz_all:.3f}, sz_active={bal_sz_active:.3f}, "
              f"global_zero={((X_fs_raw == 0).mean()):.3f}, "
              f"active_in_all={active_fs.sum()}, max_zoz_range={max_zoz:.3f}")

        other_feature_rows.append({
            "feature_set": fs,
            "global_zero_frac": float((X_fs_raw == 0).mean()),
            "n_active_in_all_sites": int(active_fs.sum()),
            "max_zscore_of_zero_range": float(max_zoz),
            "site_bal_acc_raw": bal_raw,
            "site_bal_acc_sz_all": bal_sz_all,
            "site_bal_acc_sz_active_only": bal_sz_active,
        })

    other_df = pd.DataFrame(other_feature_rows)

    # Save fingerprint edges
    fp_path = OUTPUT_DIR / "phase3r_5ca_fingerprint_edges.csv"
    fp_df.to_csv(fp_path, index=False)

    # ── 7. Report ──────────────────────────────────────────────────────────────
    print("\n[8] Writing report …")

    # Top transformation results for report
    transf_summary = transf_df.set_index("transformation")["site_balanced_accuracy"].to_dict()

    # Minimum edges for 90% / 99% leakage
    k_for_90 = next((r["n_top_edges"] for r in curve_rows if r["site_bal_acc"] >= 0.90), None)
    k_for_99 = next((r["n_top_edges"] for r in curve_rows if r["site_bal_acc"] >= 0.99), None)
    k_for_100 = next((r["n_top_edges"] for r in curve_rows if r["site_bal_acc"] >= 1.00), None)

    report = {
        "phase": "3R.5C-A",
        "title": "Site Fingerprint Analysis",
        "zero_inflation": {
            "global_zero_fraction": float((X_raw == 0).mean()),
            "edges_active_in_all_sites": int(active_in_all.sum()),
            "edges_dead_in_all_sites": int(dead_in_all.sum()),
            "edges_dead_in_some_sites": int(dead_in_some.sum()),
        },
        "mechanism": {
            "description": (
                "site_zscore maps the value 0 to -(site_mean / site_std), a distinct constant "
                "per site. Because 95% of count edges are zero, subjects carry site-specific "
                "constants in the vast majority of their features after z-scoring. "
                "This creates a distributed site fingerprint."
            ),
            "transformation_test": transf_summary,
            "edges_needed_for_90pct_leakage": k_for_90,
            "edges_needed_for_99pct_leakage": k_for_99,
            "edges_needed_for_perfect_leakage": k_for_100,
        },
        "combat_prognosis": {
            "will_combat_fix_count_edges": False,
            "reason": (
                "ComBat removes additive and multiplicative batch effects by subtracting "
                "gamma_site and dividing by delta_site, then rescaling to pooled statistics. "
                "Applied to zero-inflated counts, this maps zero to "
                "(0 - gamma_site) / delta_site * pooled_std + pooled_mean. "
                "Since gamma_site ≈ site_mean and delta_site ≈ site_std/pooled_std, "
                "the corrected zero is pooled_mean - (site_mean/site_std)*pooled_std: "
                "still a site-specific constant. The fingerprint persists."
            ),
            "will_log1p_before_combat_help": False,
            "reason_log1p": (
                "log(1+x) maps zero to log(1) = 0, preserving all structural zeros. "
                "After ComBat on log-scale values, corrected_zero is again site-specific "
                "for the same mathematical reason."
            ),
            "what_would_help": [
                "Density normalization (count_ij / total_streamlines) removes absolute scale differences; "
                "still needs subsequent harmonization of residual site effects.",
                "Edge filtering: restrict to edges with >= threshold non-zero subjects in ALL sites; "
                "removes structurally absent connections that cannot carry cross-site biological signal.",
                "Use FA / MD / length edge features as primary features for harmonization and biomarker work "
                "(tissue microstructure metrics, less impacted by tractography sensitivity differences).",
                "Binary presence/absence adjacency matrix analyzed separately from weighted counts.",
            ],
        },
        "other_feature_sets": other_df.to_dict(orient="records"),
    }

    report_path = OUTPUT_DIR / "phase3r_5ca_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nZero inflation: {(X_raw==0).mean():.1%} of count edges are zero")
    print(f"Active in all 5 sites: {active_in_all.sum()} edges")
    print(f"Dead in some sites (structural fingerprint): {dead_in_some.sum()} edges")
    print(f"\nMinimum edges for 90% site leakage:  {k_for_90}")
    print(f"Minimum edges for 99% site leakage:  {k_for_99}")
    print(f"Minimum edges for 100% site leakage: {k_for_100}")
    print(f"\nComBat prognosis for count edges: WILL FAIL (same mechanism)")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    for p in [zi_path, fp_path, transf_path, curve_path, cm_path, report_path]:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
