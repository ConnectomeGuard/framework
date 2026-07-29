"""
NeuroFiber Phase 3R.5C — ComBat Harmonization for Continuous Connectome Features

Applies neuroCombat to FA, MD, and length edge features.

Design contract:
  - Count edges are NOT processed here; they cannot be linearly harmonized.
  - Only edges active in all sites (≥ min_per_site non-NaN subjects per site) are corrected.
  - NaN positions (structurally absent connections) are restored after ComBat.
  - Inactive edges are left as NaN throughout.
  - Biological covariates (age, sex, diagnosis) are explicitly protected.
  - Never modifies connectome_features/ originals.

ComBat model:
  x_ij = α_j + X_i β_j + γ_{batch(i),j} + δ_{batch(i),j} ε_ij
  where X_i contains [age, sex, diagnosis] and batch is site.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

META_COLS = ["subject_id", "site", "dataset", "diagnosis", "age", "sex"]
EDGE_PREFIX = "edge_"

COMBAT_MIN_PER_SITE = 1  # minimum non-NaN subjects per site for an edge to be included


# ── edge selection ────────────────────────────────────────────────────────────

def active_edge_mask(
    X: np.ndarray,
    sites: np.ndarray,
    min_per_site: int = COMBAT_MIN_PER_SITE,
) -> np.ndarray:
    """
    Boolean mask (n_edges,): True if edge has >= min_per_site non-NaN subjects
    in every site. Only these edges can have reliable ComBat estimates.
    """
    site_list = sorted(set(sites))
    counts = np.column_stack([
        (~np.isnan(X[sites == s])).sum(axis=0)
        for s in site_list
    ])  # shape (n_edges, n_sites)
    return counts.min(axis=1) >= min_per_site


# ── imputation ────────────────────────────────────────────────────────────────

def impute_site_mean(
    X: np.ndarray,
    sites: np.ndarray,
) -> np.ndarray:
    """
    Fill NaN positions with the site-specific mean for each edge.
    For sites where every subject has NaN on an edge (should not happen
    for active edges), fall back to the global mean.
    """
    X_out = X.copy()
    site_list = sorted(set(sites))
    global_means = np.nanmean(X, axis=0)
    global_means = np.where(np.isnan(global_means), 0.0, global_means)

    for s in site_list:
        smask = sites == s
        site_means = np.nanmean(X_out[smask], axis=0)
        # fall back to global mean where site mean is NaN (all-NaN site-edge)
        site_means = np.where(np.isnan(site_means), global_means, site_means)
        nan_in_site = smask[:, None] & np.isnan(X_out)
        X_out[nan_in_site] = np.broadcast_to(site_means, X_out.shape)[nan_in_site]

    return X_out


# ── ComBat application ────────────────────────────────────────────────────────

def apply_combat(
    X: np.ndarray,
    meta_rows: list[dict],
    min_per_site: int = COMBAT_MIN_PER_SITE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply neuroCombat to X (n_subjects, n_edges) using site as batch
    and age / sex / diagnosis as protected biological covariates.

    Returns:
      X_corrected  — (n_subjects, n_edges), NaN restored where original was NaN
      active_mask  — boolean (n_edges,), True for ComBat-corrected edges
      info         — dict with statistics about the correction
    """
    sites = np.array([r["site"] for r in meta_rows])
    nan_mask = np.isnan(X)  # original NaN positions

    # ── select active edges ───────────────────────────────────────────────────
    active = active_edge_mask(X, sites, min_per_site)
    n_active = active.sum()
    n_total = X.shape[1]

    if n_active == 0:
        warnings.warn("No edges pass the activity threshold; ComBat skipped.")
        return X.copy(), active, {"n_active": 0, "n_total": n_total}

    X_active = X[:, active]  # (n_subjects, n_active)

    # ── impute NaN with site mean so ComBat gets a complete matrix ────────────
    X_imputed = impute_site_mean(X_active, sites)

    # ── build covariates DataFrame ─────────────────────────────────────────────
    def _get(row, key, default=""):
        return row.get(key, default) or default

    ages, sexes, diagnoses = [], [], []
    for r in meta_rows:
        ages.append(float(_get(r, "age")) if _get(r, "age") else np.nan)
        sexes.append(1 if str(_get(r, "sex")).strip().upper() == "M" else 0)
        diagnoses.append(1 if str(_get(r, "diagnosis")).strip().upper() == "ASD" else 0)

    covars_df = pd.DataFrame({
        "site": sites,
        "age": ages,
        "sex": sexes,
        "diagnosis": diagnoses,
    })

    # replace NaN age with site mean age for covariate
    age_arr = np.array(ages)
    if np.isnan(age_arr).any():
        for s in sorted(set(sites)):
            smask = sites == s
            site_age_mean = np.nanmean(age_arr[smask])
            nan_age = smask & np.isnan(age_arr)
            covars_df.loc[nan_age, "age"] = site_age_mean

    # ── run ComBat (features × subjects) ─────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = neuroCombat(
            dat=X_imputed.T,                   # (n_active, n_subjects)
            covars=covars_df,
            batch_col="site",
            continuous_cols=["age"],
            categorical_cols=["sex", "diagnosis"],
            eb=True,
            parametric=True,
        )

    X_combat = result["data"].T  # back to (n_subjects, n_active)

    # ── restore NaN positions ─────────────────────────────────────────────────
    nan_active = nan_mask[:, active]
    X_combat[nan_active] = np.nan

    # ── assemble full output ───────────────────────────────────────────────────
    X_corrected = X.copy()
    X_corrected[:, active] = X_combat

    info = {
        "n_active_edges": int(n_active),
        "n_total_edges": int(n_total),
        "n_inactive_edges": int(n_total - n_active),
        "nan_fraction_before": float(np.isnan(X).mean()),
        "nan_fraction_after": float(np.isnan(X_corrected).mean()),
        "min_per_site_threshold": min_per_site,
    }
    return X_corrected, active, info


# ── variance metrics ──────────────────────────────────────────────────────────

def variance_retained(X_corrected: np.ndarray, X_baseline: np.ndarray) -> float:
    """Ratio of total (nanvar) variance vs baseline across all edges."""
    var_corr = float(np.nanvar(X_corrected))
    var_base = float(np.nanvar(X_baseline))
    return var_corr / var_base if var_base > 0 else np.nan
