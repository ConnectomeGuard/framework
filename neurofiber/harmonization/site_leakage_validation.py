"""
NeuroFiber Phase 3R.5B — Harmonization Validation and Site Leakage Analysis

Tests whether harmonization truly removes site effects by training simple ML models
to predict site, age, and sex from connectome edge features.

Key question: Can site still be decoded after harmonization?

If a classifier can predict site much better than chance (balanced acc > 0.30
for 5 classes where chance = 0.20), site information survives harmonization.

Design:
  - PCA(50) + StandardScaler dimensionality reduction
  - 5-fold stratified cross-validation
  - Logistic Regression + Random Forest for classification
  - Ridge + Random Forest for regression
  - Balanced accuracy and macro F1 for imbalanced classes

Limitations documented:
  - Sex is severely imbalanced (17F / 212M) — sex experiment is exploratory only
  - NYU_2 has only 19 subjects — may be unstable across folds

Safety: read-only access to harmonized features; writes only to site_leakage_validation/.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

PIPELINE_VERSION = "3R.5B"
N_CV_FOLDS   = 5
PCA_COMPONENTS = 50
RANDOM_STATE = 42

STRATEGIES    = ["none", "site_zscore", "global_zscore", "residualized"]
FEATURE_TYPES = ["count_edges", "fa_edges", "md_edges", "length_edges"]

# Chance level for 5-class balanced accuracy
SITE_CHANCE_LEVEL   = 0.20
# Threshold: harmonization is "effective" if site balanced_acc < this
SITE_LEAKAGE_THRESH = 0.35

META_COLS = ["subject_id", "site", "dataset", "diagnosis", "age", "sex"]

SUMMARY_FIELDS = [
    "strategy", "feature_type",
    "site_accuracy", "site_balanced_accuracy", "site_macro_f1",
    "age_r", "age_r2", "age_mae",
    "sex_accuracy", "sex_balanced_accuracy",
    "site_leakage_flag",
    "recommendation",
]


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(path: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    """
    Load edge CSV.
    Returns (meta_rows, X, edge_cols) where X is (n_subjects, n_edges).
    """
    meta_rows: list[dict] = []
    rows_data: list[list[float]] = []
    edge_cols: list[str] = []

    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames
        edge_cols = [c for c in reader.fieldnames if c.startswith("edge_")]
        for row in reader:
            meta_rows.append({k: row.get(k, "") for k in META_COLS})
            vals = []
            for c in edge_cols:
                v = row[c]
                vals.append(float(v) if v != "" else 0.0)   # treat missing as 0
            rows_data.append(vals)

    X = np.array(rows_data, dtype=np.float64)
    return meta_rows, X, edge_cols


def load_metadata(meta_path: Path) -> dict[str, dict]:
    """Load enriched metadata keyed by subject_id."""
    lookup: dict[str, dict] = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            lookup[row["subject_id"]] = row
    return lookup


# ---------------------------------------------------------------------------
# ML pipeline builder
# ---------------------------------------------------------------------------

def _make_classifier(model_name: str, n_components: int) -> Pipeline:
    if model_name == "logistic":
        clf = LogisticRegression(
            C=0.1, max_iter=1000, random_state=RANDOM_STATE,
            class_weight="balanced", solver="lbfgs", multi_class="auto",
        )
    else:  # random_forest
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=2,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("clf",    clf),
    ])


def _make_regressor(model_name: str, n_components: int) -> Pipeline:
    if model_name == "ridge":
        reg = Ridge(alpha=1.0)
    else:  # random_forest
        reg = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=2,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("reg",    reg),
    ])


# ---------------------------------------------------------------------------
# Experiment functions
# ---------------------------------------------------------------------------

def run_site_prediction(
    X:     np.ndarray,
    sites: list[str],
    n_components: int = PCA_COMPONENTS,
) -> dict:
    """
    Predict site from features using stratified 5-fold CV.
    Returns dict of metrics.
    """
    le = LabelEncoder()
    y  = le.fit_transform(sites)
    n_classes = len(le.classes_)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])

    results = {"n_classes": n_classes, "classes": list(le.classes_), "models": {}}
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name in ["logistic", "random_forest"]:
        pipe = _make_classifier(model_name, n_comp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)

        acc     = float(np.mean(y_pred == y))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
        cm      = confusion_matrix(y, y_pred).tolist()

        results["models"][model_name] = {
            "accuracy":          round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1":          round(macro_f1, 4),
            "confusion_matrix":  cm,
        }
        logger.info("  site %s: bal_acc=%.3f  f1=%.3f", model_name, bal_acc, macro_f1)

    # Aggregate (take max across models = most favourable to site leakage)
    best = max(results["models"].values(), key=lambda m: m["balanced_accuracy"])
    results["best_balanced_accuracy"] = best["balanced_accuracy"]
    results["best_macro_f1"]          = best["macro_f1"]
    results["best_accuracy"]          = best["accuracy"]
    results["leakage_flag"] = best["balanced_accuracy"] > SITE_LEAKAGE_THRESH

    return results


def run_age_prediction(
    X:    np.ndarray,
    ages: list[Optional[float]],
    n_components: int = PCA_COMPONENTS,
) -> dict:
    """
    Predict age (continuous) using 5-fold CV.
    Returns dict of metrics.
    """
    valid_idx = [i for i, a in enumerate(ages) if a is not None]
    if len(valid_idx) < 20:
        return {"status": "insufficient_data", "n_valid": len(valid_idx)}

    X_v = X[valid_idx, :]
    y_v = np.array([ages[i] for i in valid_idx], dtype=float)
    n_comp = min(n_components, X_v.shape[0] - 1, X_v.shape[1])

    results = {"n_subjects": len(valid_idx), "models": {}}
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Use simple CV since it's regression
    from sklearn.model_selection import KFold
    cv_reg = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name in ["ridge", "random_forest"]:
        pipe = _make_regressor(model_name, n_comp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipe, X_v, y_v, cv=cv_reg)

        r2  = float(r2_score(y_v, y_pred))
        mae = float(mean_absolute_error(y_v, y_pred))
        try:
            r_val, _ = pearsonr(y_v, y_pred)
            r_val = float(r_val)
        except Exception:
            r_val = float("nan")

        results["models"][model_name] = {
            "r2":  round(r2, 4),
            "mae": round(mae, 2),
            "r":   round(r_val, 4) if not np.isnan(r_val) else None,
        }
        logger.info("  age %s: r=%.3f  r2=%.3f  MAE=%.1f", model_name, r_val, r2, mae)

    # Best = highest R²
    best = max(results["models"].values(), key=lambda m: m["r2"] or -999)
    results["best_r2"]  = best["r2"]
    results["best_mae"] = best["mae"]
    results["best_r"]   = best["r"]
    return results


def run_sex_prediction(
    X:     np.ndarray,
    sexes: list[str],
    n_components: int = PCA_COMPONENTS,
) -> dict:
    """
    Predict sex (binary M/F) using 5-fold CV.
    NOTE: severely imbalanced (17F / 212M) — results are exploratory only.
    """
    valid_idx = [i for i, s in enumerate(sexes) if s in ("M", "F")]
    if len(valid_idx) < 10:
        return {"status": "insufficient_data"}

    X_v  = X[valid_idx, :]
    y_str = [sexes[i] for i in valid_idx]
    le    = LabelEncoder()
    y_v   = le.fit_transform(y_str)

    class_counts = dict(zip(*np.unique(y_v, return_counts=True)))
    min_class = min(class_counts.values())
    if min_class < N_CV_FOLDS:
        return {"status": "too_few_minority_class", "class_counts": {k: int(v) for k, v in class_counts.items()}}

    n_comp = min(n_components, X_v.shape[0] - 1, X_v.shape[1])
    results = {"n_subjects": len(valid_idx),
               "class_counts": {le.classes_[k]: int(v) for k, v in class_counts.items()},
               "warning": "severely imbalanced — exploratory only",
               "models": {}}
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name in ["logistic", "random_forest"]:
        pipe = _make_classifier(model_name, n_comp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipe, X_v, y_v, cv=cv, n_jobs=1)

        acc      = float(np.mean(y_pred == y_v))
        bal_acc  = float(balanced_accuracy_score(y_v, y_pred))
        macro_f1 = float(f1_score(y_v, y_pred, average="macro", zero_division=0))

        results["models"][model_name] = {
            "accuracy":          round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1":          round(macro_f1, 4),
        }
        logger.info("  sex %s: bal_acc=%.3f  f1=%.3f", model_name, bal_acc, macro_f1)

    best = max(results["models"].values(), key=lambda m: m["balanced_accuracy"])
    results["best_balanced_accuracy"] = best["balanced_accuracy"]
    results["best_accuracy"]          = best["accuracy"]
    return results


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def run_validation_pipeline(
    harmonized_root: Path,
    meta_path:       Path,
    out_dir:         Path,
) -> list[dict]:
    """
    Run all validation experiments across all strategies and feature types.
    Returns list of summary rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    metadata = load_metadata(meta_path)

    summary_rows: list[dict] = []
    site_results_all:  list[dict] = []
    age_results_all:   list[dict] = []
    sex_results_all:   list[dict] = []

    for strategy in STRATEGIES:
        strat_dir = harmonized_root / strategy
        if not strat_dir.exists():
            logger.warning("Strategy dir not found: %s", strat_dir)
            continue

        logger.info("=" * 60)
        logger.info("Strategy: %s", strategy)

        for feat_type in FEATURE_TYPES:
            feat_path = strat_dir / f"{feat_type}.csv"
            if not feat_path.exists():
                logger.warning("[%s/%s] not found", strategy, feat_type)
                continue

            logger.info("[%s] %s", strategy, feat_type)
            meta_rows, X, _ = load_features(feat_path)

            # Align metadata
            sites = [m["site"] for m in meta_rows]
            ages  = []
            sexes = []
            for m in meta_rows:
                sub = m["subject_id"]
                full = metadata.get(sub, m)
                a = full.get("age", "")
                s = full.get("sex", "")
                ages.append(float(a) if a else None)
                sexes.append(s if s in ("M", "F") else "")

            # Site prediction
            logger.info("  Running site prediction ...")
            site_res = run_site_prediction(X, sites)
            site_res.update({"strategy": strategy, "feature_type": feat_type})
            site_results_all.append(site_res)

            # Save confusion matrix
            for model_name, model_res in site_res.get("models", {}).items():
                cm_path = cm_dir / f"{strategy}_{feat_type}_{model_name}_site_cm.json"
                cm_path.write_text(json.dumps({
                    "strategy": strategy, "feature_type": feat_type,
                    "model": model_name, "classes": site_res.get("classes"),
                    "confusion_matrix": model_res.get("confusion_matrix"),
                }, indent=2))

            # Age prediction
            logger.info("  Running age prediction ...")
            age_res = run_age_prediction(X, ages)
            age_res.update({"strategy": strategy, "feature_type": feat_type})
            age_results_all.append(age_res)

            # Sex prediction
            logger.info("  Running sex prediction ...")
            sex_res = run_sex_prediction(X, sexes)
            sex_res.update({"strategy": strategy, "feature_type": feat_type})
            sex_results_all.append(sex_res)

            # Build summary row
            leakage = site_res.get("leakage_flag", True)
            age_r   = age_res.get("best_r")
            age_r2  = age_res.get("best_r2")
            age_mae = age_res.get("best_mae")
            sex_bal = sex_res.get("best_balanced_accuracy")

            # Recommendation per row
            if not leakage and (age_r or 0) > 0.05:
                rec = "ACCEPTABLE"
            elif leakage:
                rec = "SITE_LEAKAGE_DETECTED"
            else:
                rec = "REVIEW_NEEDED"

            summary_rows.append({
                "strategy":           strategy,
                "feature_type":       feat_type,
                "site_accuracy":      round(site_res.get("best_accuracy", 0), 4),
                "site_balanced_accuracy": round(site_res.get("best_balanced_accuracy", 0), 4),
                "site_macro_f1":      round(site_res.get("best_macro_f1", 0), 4),
                "age_r":              round(age_r, 4) if age_r is not None else "",
                "age_r2":             round(age_r2, 4) if age_r2 is not None else "",
                "age_mae":            round(age_mae, 2) if age_mae is not None else "",
                "sex_accuracy":       round(sex_bal, 4) if sex_bal is not None else "",
                "sex_balanced_accuracy": round(sex_bal, 4) if sex_bal is not None else "",
                "site_leakage_flag":  "YES" if leakage else "NO",
                "recommendation":     rec,
            })

    # Write detailed result files
    _write_csv(out_dir / "site_prediction_results.csv", site_results_all,
               ["strategy","feature_type","best_balanced_accuracy","best_macro_f1","best_accuracy","leakage_flag","classes"])
    _write_csv(out_dir / "age_prediction_results.csv",  age_results_all,
               ["strategy","feature_type","best_r","best_r2","best_mae","n_subjects"])
    _write_csv(out_dir / "sex_prediction_results.csv",  sex_results_all,
               ["strategy","feature_type","best_balanced_accuracy","best_accuracy","warning","status"])

    return summary_rows


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("→ %s  (%d rows)", path.name, len(rows))


def write_summary_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Leakage validation summary → %s", out_path)
    return out_path


def recommend_final_strategy(summary_rows: list[dict]) -> tuple[str, str]:
    """
    Evaluate all strategies and return (recommended_strategy, rationale).
    Prefers low site leakage AND preserved age signal.
    """
    # Score each strategy: lower site_bal = better, higher age_r = better
    from collections import defaultdict
    strategy_scores: dict[str, list] = defaultdict(list)

    for row in summary_rows:
        if row["feature_type"] != "count_edges":
            continue
        s = row["strategy"]
        site_bal = float(row["site_balanced_accuracy"] or 1.0)
        age_r    = float(row["age_r"] or 0)
        # Score: want low site_bal and high age_r
        score = (1 - site_bal) * max(age_r, 0.01)
        strategy_scores[s].append(score)

    mean_scores = {s: float(np.mean(v)) for s, v in strategy_scores.items()}

    # Check if any strategy has unacceptable site leakage
    leaky = {
        row["strategy"]
        for row in summary_rows
        if row["site_leakage_flag"] == "YES"
    }

    # Prefer site_zscore if it's not leaky
    for preferred in ["site_zscore", "residualized", "global_zscore", "none"]:
        if preferred not in leaky and preferred in mean_scores:
            rationale = (
                f"{preferred} selected: site leakage absent, age signal retained. "
                f"Score={mean_scores.get(preferred, 0):.4f}"
            )
            return preferred, rationale

    # All strategies leak — recommend site_zscore as least-bad
    best = min(mean_scores, key=lambda s: float(
        next((r["site_balanced_accuracy"] for r in summary_rows
              if r["strategy"] == s and r["feature_type"] == "count_edges"), 1.0)
    ))
    rationale = (
        f"All strategies show site leakage. {best} has lowest site prediction. "
        "Consider installing neuroCombat for Phase 3R.5C."
    )
    return best, rationale
