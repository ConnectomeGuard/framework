"""
Phase 3R.5C — ComBat Harmonization Evaluation

Applies neuroCombat to FA, MD, and length edge features, then validates
site leakage and biological signal preservation against the none and
site_zscore baselines.

Count edges are explicitly excluded (see Phase 3R.5C-A findings).

Outputs (data/processed_v2b/):
  harmonized_connectomes/combat/
    fa_edges.csv
    md_edges.csv
    length_edges.csv
    combat_report.json
  phase3r_5c_combat_comparison.csv

Usage:
    cd /path/to/neurofiber
    python ai_pipeline/scripts/run_combat_evaluation.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai_pipeline"))

from neurofiber.harmonization.combat_evaluation import apply_combat, variance_retained
from neurofiber.harmonization.connectome_harmonization import load_edge_table, save_edge_table

HARM_DIR  = ROOT / "data/processed_v2b/harmonized_connectomes"
FEAT_DIR  = ROOT / "data/processed_v2b/connectome_features"
OUTPUT_DIR = ROOT / "data/processed_v2b"
META_PATH = FEAT_DIR / "harmonization_metadata_enriched.csv"

FEATURE_SETS = ["fa_edges", "md_edges", "length_edges"]
STRATEGIES   = ["none", "site_zscore", "combat"]

N_SPLITS = 5
N_PCA    = 50   # reduced from 100 to fit smaller active-edge set
RF_N     = 200
RANDOM_STATE = 42


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_harmonized(strategy: str, feature_set: str) -> tuple[np.ndarray, list[dict], list[str]]:
    """Load from harmonized_connectomes/<strategy>/<feature_set>.csv."""
    path = HARM_DIR / strategy / f"{feature_set}.csv"
    return load_edge_table(path)


def _impute_median(X: np.ndarray) -> np.ndarray:
    """Global median imputation for NaN; columns still all-NaN → 0."""
    X_out = X.copy()
    col_med = np.nanmedian(X_out, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nm = np.isnan(X_out)
    X_out[nm] = np.take(col_med, np.where(nm)[1])
    return X_out


def _meta_to_arrays(meta_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sites, ages, sexes from meta_rows."""
    meta = pd.read_csv(META_PATH, index_col="subject_id")
    subject_ids = [int(r["subject_id"]) for r in meta_rows]
    sites = meta.loc[subject_ids, "site"].values
    ages  = meta.loc[subject_ids, "age"].values.astype(float)
    sexes = meta.loc[subject_ids, "sex"].values
    return sites, ages, sexes


# ── leakage / signal pipelines ────────────────────────────────────────────────

def _lr_pipe(n_pca: int = N_PCA) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.1, random_state=RANDOM_STATE)),
    ])


def _rf_clf() -> RandomForestClassifier:
    return RandomForestClassifier(RF_N, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)


def _ridge_pipe(n_pca: int = N_PCA) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("reg", Ridge(alpha=1.0)),
    ])


def _rf_reg() -> RandomForestRegressor:
    return RandomForestRegressor(RF_N, random_state=RANDOM_STATE, n_jobs=-1)


def _site_leakage(X: np.ndarray, y: np.ndarray, n_pca: int) -> dict:
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in [("lr", _lr_pipe(n_pca)), ("rf", _rf_clf())]:
        bals, f1s = [], []
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            yp = pipe.predict(X[te])
            bals.append(balanced_accuracy_score(y[te], yp))
            f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
        results[f"site_{name}_bal_acc"] = float(np.mean(bals))
        results[f"site_{name}_f1"]      = float(np.mean(f1s))
    return results


def _age_signal(X: np.ndarray, ages: np.ndarray, n_pca: int) -> dict:
    mask = ~np.isnan(ages)
    X_a, y_a = X[mask], ages[mask]
    kf = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in [("ridge", _ridge_pipe(n_pca)), ("rf", _rf_reg())]:
        rs, r2s, maes = [], [], []
        for tr, te in kf.split(X_a):
            pipe.fit(X_a[tr], y_a[tr])
            yp = pipe.predict(X_a[te])
            r, _ = pearsonr(y_a[te], yp)
            rs.append(r); r2s.append(r2_score(y_a[te], yp))
            maes.append(mean_absolute_error(y_a[te], yp))
        results[f"age_{name}_r"]   = float(np.mean(rs))
        results[f"age_{name}_r2"]  = float(np.mean(r2s))
        results[f"age_{name}_mae"] = float(np.mean(maes))
    return results


def _sex_signal(X: np.ndarray, sexes: np.ndarray, y_site: np.ndarray, n_pca: int) -> dict:
    le = LabelEncoder()
    y = le.fit_transform(sexes)
    if len(np.unique(y)) < 2:
        return {"sex_lr_bal_acc": np.nan, "sex_rf_bal_acc": np.nan}
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in [("lr", _lr_pipe(n_pca)), ("rf", _rf_clf())]:
        bals = []
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            bals.append(balanced_accuracy_score(y[te], pipe.predict(X[te])))
        results[f"sex_{name}_bal_acc"] = float(np.mean(bals))
    return results


def evaluate_strategy(
    X_raw: np.ndarray,
    X_harm: np.ndarray,
    meta_rows: list[dict],
    feature_set: str,
    strategy: str,
) -> dict:
    """Run all leakage + signal tests for a single (feature_set, strategy) pair."""
    sites, ages, sexes = _meta_to_arrays(meta_rows)
    le = LabelEncoder()
    y_site = le.fit_transform(sites)

    X = _impute_median(X_harm)
    # drop all-zero columns after imputation (degenerate)
    keep = np.any(X != 0, axis=0)
    X = X[:, keep]
    n_pca = min(N_PCA, X.shape[1] - 1, X.shape[0] - 1)

    row = {"feature_set": feature_set, "strategy": strategy, "n_features": int(keep.sum())}
    row.update(_site_leakage(X, y_site, n_pca))
    row.update(_age_signal(X, ages, n_pca))
    row.update(_sex_signal(X, sexes, y_site, n_pca))
    row["variance_retained"] = variance_retained(X_harm, X_raw)
    return row


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 3R.5C — ComBat Harmonization Evaluation")
    print("=" * 60)

    combat_out_dir = HARM_DIR / "combat"
    combat_out_dir.mkdir(parents=True, exist_ok=True)

    combat_infos: dict[str, dict] = {}
    all_rows: list[dict] = []

    # ── Step 1: Apply ComBat and save harmonized files ────────────────────────
    print("\n[1] Applying ComBat to continuous edge features …")
    for fs in FEATURE_SETS:
        print(f"\n  {fs} …")
        meta_rows, X, edge_cols = load_harmonized("none", fs)
        X_corrected, active_mask, info = apply_combat(X, meta_rows)
        combat_infos[fs] = info
        print(f"    Active edges (ComBat-corrected): {info['n_active_edges']}")
        print(f"    Inactive edges (unchanged NaN): {info['n_inactive_edges']}")

        out_path = combat_out_dir / f"{fs}.csv"
        save_edge_table(out_path, meta_rows, X_corrected, edge_cols)
        print(f"    Saved → {out_path.relative_to(ROOT)}")

    # Save combat summary report
    report_path = combat_out_dir / "combat_report.json"
    with open(report_path, "w") as f:
        json.dump(combat_infos, f, indent=2)

    # ── Step 2: Load none + site_zscore for comparison ───────────────────────
    print("\n[2] Running leakage and biological signal validation …")
    print("    (none vs site_zscore vs combat, for each feature set)")

    for fs in FEATURE_SETS:
        print(f"\n  [{fs}]")
        meta_none, X_none, _ = load_harmonized("none", fs)

        for strat in STRATEGIES:
            if strat == "combat":
                meta_rows, X_harm, _ = load_edge_table(combat_out_dir / f"{fs}.csv")
            elif strat == "none":
                meta_rows, X_harm = meta_none, X_none
            else:
                meta_rows, X_harm, _ = load_harmonized(strat, fs)

            print(f"    {strat} …", end=" ", flush=True)
            row = evaluate_strategy(X_none, X_harm, meta_rows, fs, strat)
            all_rows.append(row)
            print(
                f"site_rf={row['site_rf_bal_acc']:.3f}  "
                f"site_lr={row['site_lr_bal_acc']:.3f}  "
                f"age_rf_r={row['age_rf_r']:.3f}  "
                f"age_ridge_r={row['age_ridge_r']:.3f}"
            )

    # ── Step 3: Save comparison table ─────────────────────────────────────────
    comp_df = pd.DataFrame(all_rows)
    comp_path = OUTPUT_DIR / "phase3r_5c_combat_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    # ── Step 4: Print summary and apply decision rule ─────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (MD edges — primary metric)")
    print("=" * 60)
    md_rows = comp_df[comp_df.feature_set == "md_edges"][
        ["strategy", "site_rf_bal_acc", "site_lr_bal_acc", "age_rf_r", "age_ridge_r", "variance_retained"]
    ]
    print(md_rows.to_string(index=False))

    # Decision: ComBat wins if it improves age_rf_r over site_zscore while keeping site leakage low
    none_md_age_r = float(comp_df.loc[
        (comp_df.feature_set == "md_edges") & (comp_df.strategy == "none"), "age_rf_r"
    ].values[0])
    combat_md_age_r = float(comp_df.loc[
        (comp_df.feature_set == "md_edges") & (comp_df.strategy == "combat"), "age_rf_r"
    ].values[0])
    sz_md_age_r = float(comp_df.loc[
        (comp_df.feature_set == "md_edges") & (comp_df.strategy == "site_zscore"), "age_rf_r"
    ].values[0])
    combat_md_site_rf = float(comp_df.loc[
        (comp_df.feature_set == "md_edges") & (comp_df.strategy == "combat"), "site_rf_bal_acc"
    ].values[0])
    none_md_site_rf = float(comp_df.loc[
        (comp_df.feature_set == "md_edges") & (comp_df.strategy == "none"), "site_rf_bal_acc"
    ].values[0])

    threshold_age = none_md_age_r * 0.80
    threshold_site = none_md_site_rf * 0.85

    combat_wins = (
        combat_md_age_r >= threshold_age
        and combat_md_site_rf <= none_md_site_rf  # not worse than baseline
        and combat_md_age_r > sz_md_age_r         # better age than site_zscore
    )

    decision = "combat_selected" if combat_wins else "needs_investigation"
    canonical_strategy = "combat" if combat_wins else "site_zscore"

    print(f"\nDecision thresholds (MD edges):")
    print(f"  Age Pearson r ≥ {threshold_age:.3f} (80% of none={none_md_age_r:.3f})")
    print(f"  Site RF ≤ {none_md_site_rf:.3f} (none baseline)")
    print(f"  ComBat age r = {combat_md_age_r:.3f} | site_zscore age r = {sz_md_age_r:.3f}")
    print(f"\nDecision: {decision}")
    print(f"Canonical harmonization strategy: {canonical_strategy}")

    # Save full report
    full_report = {
        "phase": "3R.5C",
        "decision": decision,
        "canonical_strategy": canonical_strategy,
        "feature_sets_evaluated": FEATURE_SETS,
        "excluded_feature_set": "count_edges",
        "exclusion_reason": "Zero-fingerprint mechanism: z-score maps 0 to site-specific constant. See Phase 3R.5C-A.",
        "thresholds": {
            "age_pearson_r_min": threshold_age,
            "site_rf_bal_acc_max": none_md_site_rf,
        },
        "results": {
            "combat_md_age_rf_r": combat_md_age_r,
            "site_zscore_md_age_rf_r": sz_md_age_r,
            "none_md_age_rf_r": none_md_age_r,
            "combat_md_site_rf_bal_acc": combat_md_site_rf,
            "none_md_site_rf_bal_acc": none_md_site_rf,
        },
        "combat_info": combat_infos,
        "comparison_table": comp_df.to_dict(orient="records"),
    }

    full_report_path = OUTPUT_DIR / "phase3r_5c_report.json"
    with open(full_report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print(f"  harmonized_connectomes/combat/ (fa, md, length)")
    print(f"  {comp_path.name}")
    print(f"  {full_report_path.name}")


if __name__ == "__main__":
    main()
