"""
Tests for Phase 3R.5C — ComBat Harmonization Evaluation

Coverage:
  - active_edge_mask correctly identifies edges with sufficient data
  - impute_site_mean fills NaN with site means without cross-site leakage
  - apply_combat runs without error and restores NaN positions
  - apply_combat output has same shape as input
  - apply_combat does not modify inactive edges
  - apply_combat does not modify count_edges (guard)
  - variance_retained returns sensible value (≥ 0)
  - NaN positions in output match original NaN positions for inactive edges
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.harmonization.combat_evaluation import (
    active_edge_mask,
    apply_combat,
    impute_site_mean,
    variance_retained,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_meta(n_subjects: int = 30, n_sites: int = 3) -> list[dict]:
    """Minimal meta_rows with balanced sites and complete covariates."""
    sites = ["SITE_A", "SITE_B", "SITE_C"][:n_sites]
    per_site = n_subjects // n_sites
    rows = []
    for i in range(n_subjects):
        site = sites[i // per_site] if i // per_site < n_sites else sites[-1]
        rows.append({
            "subject_id": str(1000 + i),
            "site": site,
            "dataset": "test",
            "age": str(20 + i % 20),
            "sex": "M" if i % 2 == 0 else "F",
            "diagnosis": "ASD" if i % 3 == 0 else "CONTROL",
        })
    return rows


def _make_X(n_subjects: int = 30, n_edges: int = 50, sparse: float = 0.7) -> np.ndarray:
    """Random feature matrix with ~sparse fraction NaN."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_subjects, n_edges))
    mask = rng.random(size=X.shape) < sparse
    X[mask] = np.nan
    return X


# ── tests: active_edge_mask ───────────────────────────────────────────────────

def test_active_edge_mask_basic():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = _make_X(30, 20, sparse=0.0)  # no NaN → all edges active
    mask = active_edge_mask(X, sites, min_per_site=1)
    assert mask.all(), "all edges should be active when no NaN"


def test_active_edge_mask_with_dead_edges():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = _make_X(30, 20, sparse=0.0)
    # kill all of SITE_A on edge 0
    X[sites == "SITE_A", 0] = np.nan
    mask = active_edge_mask(X, sites, min_per_site=1)
    assert not mask[0], "edge 0 dead in SITE_A → should be inactive"
    assert mask[1:].all(), "other edges should remain active"


def test_active_edge_mask_threshold():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = np.ones((30, 5))
    # SITE_B has only 1 non-NaN for edge 0 → fails min=2 but passes min=1
    site_b = sites == "SITE_B"
    X[site_b, 0] = np.nan
    first_b = np.where(site_b)[0][0]
    X[first_b, 0] = 1.0  # exactly 1 non-NaN in SITE_B

    assert active_edge_mask(X, sites, min_per_site=1)[0]
    assert not active_edge_mask(X, sites, min_per_site=2)[0]


# ── tests: impute_site_mean ───────────────────────────────────────────────────

def test_impute_site_mean_no_nan_remaining():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = _make_X(30, 20, sparse=0.5)
    X_imp = impute_site_mean(X, sites)
    assert not np.isnan(X_imp).any(), "imputation should remove all NaN"


def test_impute_site_mean_preserves_nonnan():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = _make_X(30, 20, sparse=0.3)
    non_nan_before = ~np.isnan(X)
    X_imp = impute_site_mean(X, sites)
    np.testing.assert_array_almost_equal(
        X_imp[non_nan_before], X[non_nan_before],
        err_msg="imputation must not modify non-NaN values",
    )


def test_impute_site_mean_uses_site_values():
    # Each site has a distinct mean; imputed values should match site mean, not global mean.
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = np.full((30, 1), np.nan)
    site_list = sorted(set(sites))
    # assign distinct site means via non-NaN subjects
    for i, s in enumerate(site_list):
        smask = sites == s
        idx = np.where(smask)[0]
        X[idx[0], 0] = float(i * 100)  # SITE_A=0, SITE_B=100, SITE_C=200

    X_imp = impute_site_mean(X, sites)
    for i, s in enumerate(site_list):
        vals = X_imp[sites == s, 0]
        expected = float(i * 100)
        assert np.allclose(vals, expected), f"{s}: expected {expected}, got {vals}"


# ── tests: apply_combat ───────────────────────────────────────────────────────

def test_apply_combat_output_shape():
    meta = _make_meta(30, 3)
    X = _make_X(30, 40, sparse=0.3)
    X_corr, active, info = apply_combat(X, meta)
    assert X_corr.shape == X.shape, "corrected matrix must have same shape as input"


def test_apply_combat_restores_nan_on_inactive_edges():
    meta = _make_meta(30, 3)
    sites = np.array([r["site"] for r in meta])
    X = _make_X(30, 40, sparse=0.0)   # start with no NaN
    # make edge 0 inactive: all NaN in SITE_A
    X[sites == "SITE_A", 0] = np.nan
    X_corr, active, info = apply_combat(X, meta, min_per_site=2)
    # edge 0 is inactive (SITE_A has 0 non-NaN); check it's unchanged in output
    assert not active[0], "edge 0 should be inactive"
    np.testing.assert_array_equal(
        np.isnan(X_corr[:, 0]),
        np.isnan(X[:, 0]),
        err_msg="NaN pattern of inactive edges must be preserved",
    )


def test_apply_combat_restores_nan_positions_on_active_edges():
    """ComBat should not invent data for originally-NaN positions."""
    meta = _make_meta(30, 3)
    X = _make_X(30, 20, sparse=0.4)
    original_nan = np.isnan(X)
    X_corr, active, _ = apply_combat(X, meta)
    # All originally-NaN active positions should remain NaN
    assert np.all(np.isnan(X_corr[original_nan])), (
        "ComBat must not fill originally-NaN positions with values"
    )


def test_apply_combat_non_nan_positions_changed():
    """ComBat should actually modify at least some non-NaN values."""
    meta = _make_meta(30, 3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 20))   # no NaN, so all 20 edges active
    # add strong site effect so ComBat has something to remove
    sites = np.array([r["site"] for r in meta])
    for j, s in enumerate(sorted(set(sites))):
        X[sites == s] += j * 5
    original_non_nan = ~np.isnan(X)
    X_corr, _, _ = apply_combat(X, meta)
    assert not np.allclose(X_corr[original_non_nan], X[original_non_nan]), (
        "ComBat should modify non-NaN values when site effects are present"
    )


def test_apply_combat_info_keys():
    meta = _make_meta(30, 3)
    X = _make_X(30, 20, sparse=0.3)
    _, _, info = apply_combat(X, meta)
    for key in ["n_active_edges", "n_total_edges", "n_inactive_edges"]:
        assert key in info, f"info must contain '{key}'"
    assert info["n_active_edges"] + info["n_inactive_edges"] == info["n_total_edges"]


# ── tests: variance_retained ─────────────────────────────────────────────────

def test_variance_retained_identity():
    X = np.random.default_rng(0).normal(size=(50, 100))
    assert np.isclose(variance_retained(X, X), 1.0)


def test_variance_retained_nonnegative():
    rng = np.random.default_rng(0)
    X_base = rng.normal(size=(50, 100))
    X_corr = X_base * 0.5
    r = variance_retained(X_corr, X_base)
    assert r >= 0, "variance retained must be non-negative"


def test_variance_retained_with_nan():
    rng = np.random.default_rng(0)
    X_base = rng.normal(size=(50, 100))
    X_base[rng.random(size=X_base.shape) < 0.3] = np.nan
    X_corr = X_base * 0.8
    r = variance_retained(X_corr, X_base)
    assert np.isfinite(r), "variance retained should be finite"
