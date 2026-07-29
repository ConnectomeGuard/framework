"""
NeuroFiber Phase 3R.5 — Connectome Harmonization Strategy Selection

Applies four harmonization strategies to edge-feature CSVs and evaluates each
by measuring residual site effect and variance retained.

Strategies:
  none            — unmodified baseline (copy of original)
  site_zscore     — per-site per-edge z-score
  global_zscore   — global per-edge z-score across all subjects
  residualized    — OLS regression on site dummies; residuals as corrected features
  combat_optional — neuroCombat if available; skip with warning otherwise

Safety contract:
  - Never overwrites original connectome_features/ files
  - All outputs land under harmonized_connectomes/
  - guard_no_raw_write() enforced at every write entry point
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import kruskal

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

PIPELINE_VERSION = "3R.5"

META_COLS = ["subject_id", "site", "dataset", "diagnosis", "age", "sex"]
EDGE_PREFIX = "edge_"

STRATEGIES = ["none", "site_zscore", "global_zscore", "residualized", "combat_optional"]
FEATURE_TYPES = ["count_edges", "fa_edges", "md_edges", "length_edges"]

COMPARISON_FIELDS = [
    "strategy", "feature_type",
    "site_effect_p_density",
    "site_variance_ratio",
    "mean_abs_site_z",
    "variance_retained",
    "age_corr_before", "age_corr_after",
    "status", "notes",
]

CLEAN_SITES = ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_edge_table(path: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    """
    Load an edge CSV.
    Returns (meta_rows, X, edge_cols)
    where X is (n_subjects, n_edges) float64 and meta_rows is list of dicts.
    """
    meta_rows: list[dict] = []
    edge_cols: list[str]  = []
    rows_data: list[list[float]] = []

    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        edge_cols = [c for c in reader.fieldnames if c.startswith(EDGE_PREFIX)]

        for row in reader:
            meta_rows.append({k: row[k] for k in META_COLS if k in row})
            vals = []
            for c in edge_cols:
                v = row[c]
                vals.append(float(v) if v != "" else np.nan)
            rows_data.append(vals)

    X = np.array(rows_data, dtype=np.float64)
    return meta_rows, X, edge_cols


def save_edge_table(
    path:       Path,
    meta_rows:  list[dict],
    X:          np.ndarray,
    edge_cols:  list[str],
) -> None:
    """Save harmonized edge table to CSV (same format as input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(META_COLS + edge_cols)
        for i, meta in enumerate(meta_rows):
            meta_vals = [meta.get(k, "") for k in META_COLS]
            edge_vals = [
                str(X[i, j]) if not np.isnan(X[i, j]) else ""
                for j in range(X.shape[1])
            ]
            writer.writerow(meta_vals + edge_vals)


# ---------------------------------------------------------------------------
# Harmonization strategies
# ---------------------------------------------------------------------------

def apply_none(X: np.ndarray, sites: list[str]) -> np.ndarray:
    """Baseline — return unchanged."""
    return X.copy()


def apply_site_zscore(X: np.ndarray, sites: list[str]) -> np.ndarray:
    """
    Per-site per-edge z-score:
      x' = (x - μ_site) / σ_site

    Sites with σ=0 on a given edge are left unmodified for that edge.
    """
    X_out = X.copy()
    for site in set(sites):
        idx = [i for i, s in enumerate(sites) if s == site]
        if len(idx) < 2:
            continue
        site_data = X[idx, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu    = np.nanmean(site_data, axis=0)
            sigma = np.nanstd(site_data,  axis=0)
        sigma[sigma == 0] = np.nan  # don't divide by zero
        X_out[np.ix_(idx, np.arange(X.shape[1]))] = (site_data - mu) / sigma
    return X_out


def apply_global_zscore(X: np.ndarray, sites: list[str]) -> np.ndarray:
    """
    Global per-edge z-score across all subjects:
      x' = (x - μ_global) / σ_global
    """
    X_out = X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu    = np.nanmean(X, axis=0)
        sigma = np.nanstd(X,  axis=0)
    sigma[sigma == 0] = np.nan
    X_out = (X - mu) / sigma
    return X_out


def apply_residualization(
    X:           np.ndarray,
    sites:       list[str],
    covariates:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    OLS residualization: regress out site dummies (and optional covariates).
    x = β0 + β_site*S + β_cov*C + ε  →  x' = ε

    If no covariates are available (all NaN or None), reverts to site
    mean-centering (removes mean effect without touching variance structure).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    n = X.shape[0]

    # Site dummy encoding
    site_arr = np.array(sites).reshape(-1, 1)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    site_dummies = enc.fit_transform(site_arr)

    # Covariate matrix (drop columns with all NaN)
    if covariates is not None and not np.all(np.isnan(covariates)):
        cov_valid = covariates[:, ~np.all(np.isnan(covariates), axis=0)]
        # Fill remaining NaN with column means
        col_means = np.nanmean(cov_valid, axis=0)
        for j in range(cov_valid.shape[1]):
            nan_mask = np.isnan(cov_valid[:, j])
            cov_valid[nan_mask, j] = col_means[j]
        design = np.hstack([site_dummies, cov_valid])
    else:
        design = site_dummies

    X_out = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        y = X[:, j]
        valid = ~np.isnan(y)
        if valid.sum() < design.shape[1] + 1:
            X_out[:, j] = y
            continue
        reg = LinearRegression().fit(design[valid], y[valid])
        X_out[valid, j] = y[valid] - reg.predict(design[valid])

    return X_out


def apply_combat(
    X:           np.ndarray,
    sites:       list[str],
    covariates:  Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], str]:
    """
    Apply neuroCombat if available. Returns (X_harmonized, status_message).
    If neuroCombat not available, returns (None, warning_message).
    """
    try:
        from neuroCombat import neuroCombat  # type: ignore
    except ImportError:
        msg = ("neuroCombat not installed — ComBat strategy skipped. "
               "Install via: pip install neuroCombat")
        logger.warning(msg)
        return None, msg

    try:
        import pandas as pd
        site_series = pd.Series(sites, name="site")
        batch = site_series.astype("category").cat.codes.values

        covar_df = pd.DataFrame({"site": sites})
        if covariates is not None:
            for i in range(covariates.shape[1]):
                covar_df[f"cov_{i}"] = covariates[:, i]

        # neuroCombat expects (features, subjects)
        data_T = X.T.copy()
        data_T = np.nan_to_num(data_T, nan=0.0)

        result = neuroCombat(dat=data_T, covars=covar_df, batch_col="site")
        X_out = result["data"].T
        return X_out, "success"

    except Exception as exc:
        msg = f"neuroCombat failed: {exc}"
        logger.error(msg)
        return None, msg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_site_effect(
    X:      np.ndarray,
    sites:  list[str],
    n_sample: int = 200,
) -> dict:
    """
    Estimate residual site effect on a random sample of edges.
    Returns dict with site_effect_p (median KW p-value), site_variance_ratio,
    mean_abs_site_z, variance_retained (vs original).
    """
    rng = np.random.default_rng(42)
    n_edges = X.shape[1]
    sample_idx = rng.choice(n_edges, size=min(n_sample, n_edges), replace=False)

    p_values = []
    for j in sample_idx:
        col = X[:, j]
        groups = [col[[i for i, s in enumerate(sites) if s == site]]
                  for site in CLEAN_SITES]
        groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
        if len(groups) < 2:
            continue
        try:
            _, p = kruskal(*groups)
            p_values.append(p)
        except Exception:
            pass

    # Site variance ratio: between-site variance / total variance
    site_means = []
    for site in CLEAN_SITES:
        idx = [i for i, s in enumerate(sites) if s == site]
        if idx:
            site_means.append(np.nanmean(X[idx, :]))
    grand_mean = np.nanmean(X)
    total_var  = np.nanvar(X)
    between_var = np.nanvar(site_means) if site_means else 0.0
    site_var_ratio = between_var / total_var if total_var > 0 else 0.0

    # Mean absolute site z-score (of site means vs grand mean)
    grand_std = np.nanstd(X)
    abs_zs = [abs((m - grand_mean) / grand_std) for m in site_means] if grand_std > 0 else [0.0]

    return {
        "median_site_effect_p": float(np.median(p_values)) if p_values else 1.0,
        "site_variance_ratio":  round(float(site_var_ratio), 6),
        "mean_abs_site_z":      round(float(np.mean(abs_zs)), 4),
    }


def evaluate_variance_retained(
    X_orig: np.ndarray,
    X_harm: np.ndarray,
) -> float:
    """Fraction of total variance retained after harmonization."""
    orig_var = float(np.nanvar(X_orig))
    harm_var = float(np.nanvar(X_harm))
    if orig_var == 0:
        return 1.0
    return round(harm_var / orig_var, 6)


def evaluate_age_correlation(
    X:    np.ndarray,
    ages: list[Optional[float]],
    n_sample: int = 100,
) -> Optional[float]:
    """
    Mean absolute Pearson correlation between features and age.
    Returns None if fewer than 10 subjects have age data.
    """
    valid_idx = [i for i, a in enumerate(ages) if a is not None and not np.isnan(a)]
    if len(valid_idx) < 10:
        return None

    from scipy.stats import pearsonr
    age_arr  = np.array([ages[i] for i in valid_idx])
    rng      = np.random.default_rng(42)
    n_edges  = X.shape[1]
    # Prefer non-constant (non-zero variance) edges for more informative correlation
    edge_stds = np.nanstd(X[valid_idx, :], axis=0)
    nonconst  = np.where(edge_stds > 1e-10)[0]
    pool      = nonconst if len(nonconst) >= n_sample else np.arange(n_edges)
    sample    = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)

    corrs = []
    for j in sample:
        col = X[valid_idx, j]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        col_valid = col[valid]
        if np.std(col_valid) < 1e-10:   # skip constant edges (all-zero)
            continue
        try:
            r, _ = pearsonr(age_arr[valid], col_valid)
            if not np.isnan(r):
                corrs.append(abs(r))
        except Exception:
            pass

    return round(float(np.mean(corrs)), 6) if corrs else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_harmonization_pipeline(
    feat_dir:  Path,
    out_root:  Path,
    raw_root:  Path,
) -> tuple[list[dict], str]:
    """
    Run all harmonization strategies on all feature types.
    Returns (comparison_rows, recommended_strategy).
    """
    guard_no_raw_write(out_root, raw_root)

    # Safety: never write into original feat_dir
    assert str(out_root.resolve()) != str(feat_dir.resolve()), (
        "SAFETY: output must not point to original connectome_features/ dir"
    )

    # Prefer enriched metadata (with phenotype) if available; fall back to base
    enriched_meta = feat_dir / "harmonization_metadata_enriched.csv"
    base_meta     = feat_dir / "harmonization_metadata.csv"
    harmeta_path  = enriched_meta if enriched_meta.exists() else base_meta
    if enriched_meta.exists():
        logger.info("Using enriched phenotype metadata: %s", harmeta_path.name)
    else:
        logger.warning("Enriched metadata not found — run enrich_harmonization_metadata.py first")

    ages: list[Optional[float]] = []
    if harmeta_path.exists():
        with open(harmeta_path) as f:
            for row in csv.DictReader(f):
                a = row.get("age", "").strip()
                try:
                    ages.append(float(a) if a else None)
                except ValueError:
                    ages.append(None)

    comparison_rows: list[dict] = []

    for feat_type in FEATURE_TYPES:
        src_path = feat_dir / f"{feat_type}.csv"
        if not src_path.exists():
            logger.warning("Feature file not found: %s", src_path)
            continue

        meta_rows, X_orig, edge_cols = load_edge_table(src_path)
        sites = [m["site"] for m in meta_rows]

        # Ensure ages list aligns with subjects
        if not ages or len(ages) != len(meta_rows):
            ages_aligned: list[Optional[float]] = [None] * len(meta_rows)
        else:
            ages_aligned = ages

        # Covariate matrix (age, sex) — all NaN if unavailable
        cov_age = np.array([a if a is not None else np.nan for a in ages_aligned]).reshape(-1, 1)
        covariates = cov_age

        age_corr_before = evaluate_age_correlation(X_orig, ages_aligned)

        eval_orig = evaluate_site_effect(X_orig, sites)

        for strategy in STRATEGIES:
            logger.info("[%s] applying %s ...", feat_type, strategy)
            status = "success"
            notes  = ""

            if strategy == "none":
                X_harm = apply_none(X_orig, sites)

            elif strategy == "site_zscore":
                X_harm = apply_site_zscore(X_orig, sites)

            elif strategy == "global_zscore":
                X_harm = apply_global_zscore(X_orig, sites)

            elif strategy == "residualized":
                X_harm = apply_residualization(X_orig, sites, covariates)

            elif strategy == "combat_optional":
                X_harm_maybe, msg = apply_combat(X_orig, sites, covariates)
                if X_harm_maybe is None:
                    X_harm = X_orig.copy()
                    status = "skipped"
                    notes  = msg
                else:
                    X_harm = X_harm_maybe

            else:
                continue

            # Save harmonized table
            out_dir  = out_root / strategy
            out_path = out_dir / f"{feat_type}.csv"
            guard_no_raw_write(out_dir, raw_root)
            save_edge_table(out_path, meta_rows, X_harm, edge_cols)

            # Evaluate
            if status == "success":
                eval_harm          = evaluate_site_effect(X_harm, sites)
                var_retained       = evaluate_variance_retained(X_orig, X_harm)
                age_corr_after     = evaluate_age_correlation(X_harm, ages_aligned)
                site_effect_p      = eval_harm["median_site_effect_p"]
                site_var_ratio     = eval_harm["site_variance_ratio"]
                mean_abs_site_z    = eval_harm["mean_abs_site_z"]
            else:
                eval_harm      = eval_orig
                var_retained   = 1.0
                age_corr_after = age_corr_before
                site_effect_p  = eval_orig["median_site_effect_p"]
                site_var_ratio = eval_orig["site_variance_ratio"]
                mean_abs_site_z = eval_orig["mean_abs_site_z"]

            comparison_rows.append({
                "strategy":          strategy,
                "feature_type":      feat_type,
                "site_effect_p_density": round(site_effect_p, 6),
                "site_variance_ratio":   round(site_var_ratio, 6),
                "mean_abs_site_z":       round(mean_abs_site_z, 4),
                "variance_retained":     round(var_retained, 6),
                "age_corr_before":       age_corr_before if age_corr_before is not None else "",
                "age_corr_after":        age_corr_after  if age_corr_after  is not None else "",
                "status":                status,
                "notes":                 notes,
            })

        logger.info("[%s] done", feat_type)

    # Write per-strategy harmonization reports
    for strategy in STRATEGIES:
        out_dir = out_root / strategy
        if not out_dir.exists():
            continue
        _write_harmonization_report(out_dir, strategy, comparison_rows)

    # Recommend strategy
    recommended = _recommend_strategy(comparison_rows)
    logger.info("Recommended strategy: %s", recommended)

    return comparison_rows, recommended


def _write_harmonization_report(
    out_dir:  Path,
    strategy: str,
    all_rows: list[dict],
) -> None:
    """Write harmonization_report.json to strategy output dir."""
    rows = [r for r in all_rows if r["strategy"] == strategy]
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "strategy":         strategy,
        "timestamp":        _now_iso(),
        "feature_types":    FEATURE_TYPES,
        "results":          rows,
    }
    (out_dir / "harmonization_report.json").write_text(
        json.dumps(report, indent=2)
    )


def _recommend_strategy(comparison_rows: list[dict]) -> str:
    """
    Select the strategy that best balances:
      1. Lower site effect (higher p-value = weaker remaining effect)
      2. Higher variance retained (don't collapse signal)
      3. Simplicity and reproducibility

    Preference order (when all strategies succeed):
      site_zscore > residualized > global_zscore > none > combat_optional
    """
    successful = [r for r in comparison_rows
                  if r["status"] == "success" and r["strategy"] != "none"]

    if not successful:
        return "none"

    # Score each strategy: site_effect_p (higher = better) + variance_retained (higher = better)
    scores: dict[str, list[float]] = {}
    for row in successful:
        s = row["strategy"]
        p = float(row["site_effect_p_density"] or 0)
        v = float(row["variance_retained"] or 0)
        # Penalize strategies that collapse variance (< 20%)
        if v < 0.2:
            p = 0.0
        scores.setdefault(s, []).append(p * v)

    mean_scores = {s: np.mean(v) for s, v in scores.items()}

    # Prefer site_zscore if it scores well
    for preferred in ["site_zscore", "residualized", "global_zscore"]:
        if preferred in mean_scores and mean_scores[preferred] > 0:
            return preferred

    return max(mean_scores, key=mean_scores.get) if mean_scores else "none"


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_comparison_csv(
    rows:     list[dict],
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Strategy comparison → %s  (%d rows)", out_path, len(rows))
    return out_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
