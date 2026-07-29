"""
Phase 3R.5B — Harmonization Validation and Site Leakage Analysis

Determines whether site_zscore (and other strategies) truly remove site
information while preserving biological signal (age, sex).

Experiments per strategy × feature set:
  1. Site prediction    → Logistic Regression + Random Forest
  2. Age prediction     → Ridge Regression + Random Forest Regressor
  3. Sex prediction     → Logistic Regression + Random Forest

Outputs (written to data/processed_v2b/):
  phase3r_5b_leakage_results.csv   — full row-per-experiment table
  phase3r_5b_variance_results.csv  — variance preservation per strategy/feature
  phase3r_5b_summary.csv           — per-strategy aggregate
  phase3r_5b_report.json           — decision + justification

Usage:
    cd /path/to/neurofiber
    python ai_pipeline/scripts/run_phase35b.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
HARMONIZED_DIR = ROOT / "data/processed_v2b/harmonized_connectomes"
OUTPUT_DIR = ROOT / "data/processed_v2b"

STRATEGIES = ["none", "site_zscore", "global_zscore", "residualized"]
FEATURE_SETS = ["count_edges", "fa_edges", "md_edges", "length_edges"]
META_COLS = ["site", "dataset", "diagnosis", "age", "sex"]
METADATA_PATH = ROOT / "data/processed_v2b/connectome_features/harmonization_metadata_enriched.csv"

N_SPLITS = 5
N_PCA = 100
RANDOM_STATE = 42
RF_ESTIMATORS = 200


# ── helpers ──────────────────────────────────────────────────────────────────

_METADATA: pd.DataFrame | None = None


def _get_metadata() -> pd.DataFrame:
    global _METADATA
    if _METADATA is None:
        _METADATA = pd.read_csv(METADATA_PATH, index_col="subject_id")
    return _METADATA


def load_features(strategy: str, feature_set: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (X, meta_df) where X is float array with NaNs median-imputed."""
    path = HARMONIZED_DIR / strategy / f"{feature_set}.csv"
    df = pd.read_csv(path, index_col=0)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(float)
    # median imputation; columns that are all-NaN get filled with 0
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    # drop columns still all-zero after imputation to avoid rank deficiency
    nonzero_cols = np.any(X != 0, axis=0)
    X = X[:, nonzero_cols]

    # build meta from enriched metadata (age/sex are NaN in edge CSVs)
    enriched = _get_metadata()
    meta = enriched.loc[df.index, ["site", "age", "sex", "diagnosis"]].copy()
    return X, meta


def lr_pipeline(n_pca: int = N_PCA, class_weight: str | None = None) -> Pipeline:
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            C=0.1,
        )),
    ]
    return Pipeline(steps)


def rf_clf_pipeline(class_weight: str | None = None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCA, random_state=RANDOM_STATE)),
        ("reg", Ridge(alpha=1.0)),
    ])


def rf_reg_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# ── experiment runners ────────────────────────────────────────────────────────

def run_site_prediction(X: np.ndarray, meta: pd.DataFrame) -> dict:
    y = LabelEncoder().fit_transform(meta["site"])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for model_name, pipe in [("lr", lr_pipeline(class_weight="balanced")),
                              ("rf", rf_clf_pipeline(class_weight="balanced"))]:
        accs, bal_accs, f1s = [], [], []
        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            accs.append((y_pred == y[test_idx]).mean())
            bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
        results[model_name] = {
            "accuracy": float(np.mean(accs)),
            "balanced_accuracy": float(np.mean(bal_accs)),
            "macro_f1": float(np.mean(f1s)),
        }
    return results


def run_age_prediction(X: np.ndarray, meta: pd.DataFrame) -> dict:
    mask = meta["age"].notna()
    X_age = X[mask]
    y_age = meta["age"].values[mask].astype(float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for model_name, pipe in [("ridge", ridge_pipeline()),
                              ("rf", rf_reg_pipeline())]:
        rs, r2s, maes = [], [], []
        for train_idx, test_idx in kf.split(X_age):
            pipe.fit(X_age[train_idx], y_age[train_idx])
            y_pred = pipe.predict(X_age[test_idx])
            r, _ = pearsonr(y_age[test_idx], y_pred)
            rs.append(r)
            r2s.append(r2_score(y_age[test_idx], y_pred))
            maes.append(mean_absolute_error(y_age[test_idx], y_pred))
        results[model_name] = {
            "pearson_r": float(np.mean(rs)),
            "r2": float(np.mean(r2s)),
            "mae": float(np.mean(maes)),
        }
    return results


def run_sex_prediction(X: np.ndarray, meta: pd.DataFrame) -> dict:
    # encode: F=0, M=1 (or whatever LabelEncoder produces)
    le = LabelEncoder()
    y = le.fit_transform(meta["sex"])
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return {"lr": {"balanced_accuracy": np.nan, "macro_f1": np.nan},
                "rf": {"balanced_accuracy": np.nan, "macro_f1": np.nan}}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for model_name, pipe in [("lr", lr_pipeline(class_weight="balanced")),
                              ("rf", rf_clf_pipeline(class_weight="balanced"))]:
        bal_accs, f1s = [], []
        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
        results[model_name] = {
            "balanced_accuracy": float(np.mean(bal_accs)),
            "macro_f1": float(np.mean(f1s)),
        }
    return results


def compute_variance(X: np.ndarray) -> dict:
    """Per-feature variance statistics."""
    variances = np.var(X, axis=0)
    return {
        "total_variance": float(np.sum(variances)),
        "mean_variance": float(np.mean(variances)),
        "median_variance": float(np.median(variances)),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 3R.5B — Harmonization Validation")
    print("=" * 60)

    # baseline variance (none strategy)
    baseline_variance: dict[str, float] = {}

    leakage_rows = []
    variance_rows = []

    for strategy in STRATEGIES:
        print(f"\n[Strategy] {strategy}")
        for fs in FEATURE_SETS:
            print(f"  [Feature set] {fs} ...", end=" ", flush=True)
            X, meta = load_features(strategy, fs)

            # variance
            var_stats = compute_variance(X)
            if strategy == "none":
                baseline_variance[fs] = var_stats["total_variance"]

            variance_rows.append({
                "strategy": strategy,
                "feature_set": fs,
                **var_stats,
            })

            # experiments
            site_res = run_site_prediction(X, meta)
            age_res = run_age_prediction(X, meta)
            sex_res = run_sex_prediction(X, meta)

            for model, metrics in site_res.items():
                leakage_rows.append({
                    "strategy": strategy,
                    "feature_set": fs,
                    "experiment": "site_prediction",
                    "model": model,
                    **metrics,
                    "pearson_r": np.nan,
                    "r2": np.nan,
                    "mae": np.nan,
                })

            for model, metrics in age_res.items():
                leakage_rows.append({
                    "strategy": strategy,
                    "feature_set": fs,
                    "experiment": "age_prediction",
                    "model": model,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    **metrics,
                })

            for model, metrics in sex_res.items():
                leakage_rows.append({
                    "strategy": strategy,
                    "feature_set": fs,
                    "experiment": "sex_prediction",
                    "model": model,
                    "accuracy": np.nan,
                    **metrics,
                    "pearson_r": np.nan,
                    "r2": np.nan,
                    "mae": np.nan,
                })

            print("done")

    # ── build DataFrames ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(leakage_rows)
    variance_df = pd.DataFrame(variance_rows)

    # add variance preservation ratio
    variance_df["variance_ratio_vs_none"] = variance_df.apply(
        lambda r: r["total_variance"] / baseline_variance[r["feature_set"]]
        if baseline_variance.get(r["feature_set"], 0) > 0 else np.nan,
        axis=1,
    )

    # ── summary: per strategy, mean of best model per experiment ─────────────
    summary_rows = []
    for strategy in STRATEGIES:
        s_df = results_df[results_df["strategy"] == strategy]

        # site leakage: best balanced accuracy (worst-case = highest)
        site_df = s_df[s_df["experiment"] == "site_prediction"]
        site_bal_acc = site_df["balanced_accuracy"].max() if not site_df.empty else np.nan

        # age signal: best pearson r
        age_df = s_df[s_df["experiment"] == "age_prediction"]
        age_r = age_df["pearson_r"].max() if not age_df.empty else np.nan

        # sex signal: best balanced accuracy
        sex_df = s_df[s_df["experiment"] == "sex_prediction"]
        sex_bal_acc = sex_df["balanced_accuracy"].max() if not sex_df.empty else np.nan

        # variance ratio (mean across feature sets)
        var_ratio = variance_df[variance_df["strategy"] == strategy]["variance_ratio_vs_none"].mean()

        summary_rows.append({
            "strategy": strategy,
            "site_leakage_bal_acc": site_bal_acc,
            "age_signal_pearson_r": age_r,
            "sex_signal_bal_acc": sex_bal_acc,
            "mean_variance_ratio": float(var_ratio),
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── save outputs ──────────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "phase3r_5b_leakage_results.csv"
    variance_path = OUTPUT_DIR / "phase3r_5b_variance_results.csv"
    summary_path = OUTPUT_DIR / "phase3r_5b_summary.csv"

    results_df.to_csv(results_path, index=False)
    variance_df.to_csv(variance_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ── decision logic ────────────────────────────────────────────────────────
    # Chance for 5-site balanced accuracy ≈ 0.20
    # Good harmonization: site leakage << 1.0; age/sex signal preserved
    none_site = summary_df.loc[summary_df["strategy"] == "none", "site_leakage_bal_acc"].values[0]
    none_age_r = summary_df.loc[summary_df["strategy"] == "none", "age_signal_pearson_r"].values[0]

    decision = "undecided"
    selected_strategy = None
    justification_parts = []

    for _, row in summary_df.iterrows():
        if row["strategy"] == "none":
            continue
        site_reduction = none_site - row["site_leakage_bal_acc"]
        age_preserved = row["age_signal_pearson_r"] >= (none_age_r * 0.80)
        leakage_acceptable = row["site_leakage_bal_acc"] < (none_site * 0.70)

        justification_parts.append({
            "strategy": row["strategy"],
            "site_reduction": float(site_reduction),
            "site_leakage_acceptable": bool(leakage_acceptable),
            "age_signal_preserved": bool(age_preserved),
            "variance_ratio": float(row["mean_variance_ratio"]),
        })

        if leakage_acceptable and age_preserved:
            if selected_strategy is None:
                selected_strategy = row["strategy"]
                decision = "strategy_selected"

    if selected_strategy is None:
        decision = "no_strategy_passes"
        selected_strategy = "combat_needed"

    report = {
        "phase": "3R.5B",
        "decision": decision,
        "selected_strategy": selected_strategy,
        "none_site_leakage_balanced_acc": float(none_site),
        "none_age_pearson_r": float(none_age_r),
        "strategy_evaluations": justification_parts,
        "summary": summary_df.to_dict(orient="records"),
    }

    report_path = OUTPUT_DIR / "phase3r_5b_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDecision: {decision}")
    print(f"Selected strategy: {selected_strategy}")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print(f"  {results_path.name}")
    print(f"  {variance_path.name}")
    print(f"  {summary_path.name}")
    print(f"  {report_path.name}")


if __name__ == "__main__":
    main()
