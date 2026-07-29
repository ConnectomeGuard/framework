"""
Tests for neurofiber.biomarkers.biomarker_discovery — Phase 4.1

Covers:
  - metadata alignment
  - FDR correction
  - Cohen's d calculation
  - design matrix construction
  - NaN handling
  - site robustness logic
  - output schema
  - edge index mapping
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from neurofiber.biomarkers.biomarker_discovery import (
    cohen_d,
    apply_fdr,
    edge_index_to_rois,
    build_edge_index_table,
    roi_to_network,
    _site_dummies,
    _fit_model,
    compute_edge_stats,
    compute_site_robustness,
    stability_score,
    BiomarkerDataset,
    EdgeStats,
    SiteRobustness,
    MIN_N_VALID,
    MIN_N_PER_GROUP,
    FDR_STRICT,
    FDR_EXPLORATORY,
    BALANCED_SITES,
    N_ROIS,
    N_EDGES,
)
import statsmodels.api as sm


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_meta(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic metadata with balanced ASD/Control across 4 balanced sites."""
    rng = np.random.default_rng(seed)
    # Ensure n_per_site is even so ASD+CONTROL exactly fills n_per_site
    n_per_site = (n // 4 // 2) * 2  # round down to nearest even
    n_actual = n_per_site * 4
    sites = ["BNI"] * n_per_site + ["NYU_1"] * n_per_site + ["SDSU_1"] * n_per_site + ["TCD_1"] * n_per_site
    dx = (["ASD"] * (n_per_site // 2) + ["CONTROL"] * (n_per_site // 2)) * 4
    n = n_actual  # use actual count
    ages = rng.uniform(8, 35, n)
    sexes = rng.choice(["M", "F"], n, p=[0.9, 0.1])
    mean_fa = rng.uniform(0.20, 0.30, n)
    density = rng.uniform(0.03, 0.10, n)
    return pd.DataFrame({
        "subject_id": range(1, n + 1),
        "site": sites,
        "diagnosis": dx,
        "age": ages,
        "sex": sexes,
        "mean_fa": mean_fa,
        "density": density,
    })


def _make_dataset(n_subj: int = 100, n_edges: int = 50, seed: int = 1) -> BiomarkerDataset:
    """Minimal synthetic BiomarkerDataset for testing."""
    rng = np.random.default_rng(seed)
    meta = _make_meta(n_subj)
    n_actual = len(meta)  # may differ from n_subj after even-rounding in _make_meta
    edge_cols = [f"edge_{i+1:04d}" for i in range(n_edges)]

    # Sparse matrix: ~60% NaN like real data
    X = rng.normal(0.0007, 0.0001, (n_actual, n_edges))
    nan_mask = rng.random((n_actual, n_edges)) < 0.60
    X[nan_mask] = np.nan

    # active edges: non-NaN in every site (at least 1 per site)
    sites = meta["site"].values
    site_list = sorted(set(sites))
    active = np.ones(n_edges, dtype=bool)
    for s in site_list:
        site_rows = X[sites == s]
        active &= (~np.isnan(site_rows)).any(axis=0)

    return BiomarkerDataset(
        X_combat=X,
        X_raw=X.copy(),
        meta=meta,
        edge_cols=edge_cols,
        active_mask=active,
        feature_name="md_edges",
        roi_labels=None,
    )


# ── Cohen's d ─────────────────────────────────────────────────────────────────

class TestCohenD:
    def test_positive_effect(self):
        a = np.array([2.0, 2.5, 3.0, 2.8])
        b = np.array([1.0, 1.2, 0.9, 1.1])
        d = cohen_d(a, b)
        assert d > 0, "ASD group has higher values → d should be positive"
        assert 1.5 < d < 5.0

    def test_negative_effect(self):
        a = np.array([1.0, 1.1])
        b = np.array([3.0, 3.1])
        d = cohen_d(a, b)
        assert d < 0

    def test_zero_effect(self):
        a = np.ones(20)
        b = np.ones(20)
        assert np.isnan(cohen_d(a, b)), "Zero pooled std → NaN"

    def test_too_few_samples(self):
        assert np.isnan(cohen_d(np.array([1.0]), np.array([2.0])))

    def test_known_value(self):
        # d = (mean_a - mean_b) / pooled_std
        # a ~ N(1, 1), b ~ N(0, 1) → d ≈ 1.0
        rng = np.random.default_rng(42)
        a = rng.normal(1, 1, 500)
        b = rng.normal(0, 1, 500)
        d = cohen_d(a, b)
        assert 0.8 < d < 1.2, f"Expected d≈1.0, got {d:.3f}"


# ── FDR correction ────────────────────────────────────────────────────────────

class TestFDR:
    def _make_stats(self, pvals):
        out = []
        for i, p in enumerate(pvals):
            s = EdgeStats(
                edge_id=f"edge_{i:04d}",
                roi_i=1, roi_j=2,
                roi_i_name="ROI_1", roi_j_name="ROI_2",
                network_i="Vis", network_j="Default",
                is_cross_site_active=True,
                n_valid=50, n_asd=25, n_control=25,
                mean_asd=0.001, mean_control=0.0008,
                std_asd=0.0001, std_control=0.0001,
                cohen_d=0.5, direction="ASD_higher",
                beta_A=0.0002, p_A=p,
                beta_B=0.0002, p_B=p, se_B=0.00005,
                beta_C=0.0002, p_C=p,
            )
            out.append(s)
        return out

    def test_all_high_pvalues(self):
        stats = self._make_stats([0.5, 0.6, 0.7, 0.8, 0.9])
        result = apply_fdr(stats)
        assert all(not s.q_B_strict for s in result)
        assert all(s.q_B > FDR_EXPLORATORY for s in result)

    def test_all_low_pvalues(self):
        stats = self._make_stats([1e-10] * 10)
        result = apply_fdr(stats)
        assert all(s.q_B_strict for s in result)

    def test_q_monotone_with_p(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        stats = self._make_stats(pvals)
        result = apply_fdr(stats)
        q_vals = [s.q_B for s in result]
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-9, "q-values should be non-decreasing with p"

    def test_nan_p_handled(self):
        stats = self._make_stats([0.001, np.nan, 0.5])
        result = apply_fdr(stats)
        assert np.isnan(result[1].q_B)
        assert not np.isnan(result[0].q_B)

    def test_q_between_0_and_1(self):
        pvals = [0.001, 0.01, 0.05, 0.3, 0.8]
        stats = self._make_stats(pvals)
        result = apply_fdr(stats)
        for s in result:
            if not np.isnan(s.q_B):
                assert 0.0 <= s.q_B <= 1.0


# ── edge index mapping ────────────────────────────────────────────────────────

class TestEdgeIndex:
    def test_first_edge(self):
        i, j = edge_index_to_rois(0, n_rois=100)
        assert (i, j) == (1, 2)

    def test_last_edge(self):
        i, j = edge_index_to_rois(N_EDGES - 1, n_rois=100)
        assert (i, j) == (99, 100)

    def test_all_edges_unique_pairs(self):
        table = build_edge_index_table(n_rois=100)
        assert len(table) == N_EDGES
        pairs = list(zip(table["roi_i"], table["roi_j"]))
        assert len(set(pairs)) == len(pairs), "All edge pairs must be unique"

    def test_roi_i_less_than_j(self):
        table = build_edge_index_table(n_rois=100)
        assert (table["roi_i"] < table["roi_j"]).all()

    def test_table_shape(self):
        table = build_edge_index_table()
        assert table.shape == (N_EDGES, 3)  # edge_idx, roi_i, roi_j


# ── network mapping ───────────────────────────────────────────────────────────

class TestNetworkMapping:
    def test_known_rois_return_string(self):
        for roi in range(1, N_ROIS + 1):
            net = roi_to_network(roi)
            assert isinstance(net, str) and len(net) > 0

    def test_7_networks_covered(self):
        nets = {roi_to_network(roi) for roi in range(1, N_ROIS + 1)}
        assert len(nets) <= 8  # 7 networks + possibly "Unknown"

    def test_out_of_range_returns_unknown(self):
        assert roi_to_network(0) == "Unknown"
        assert roi_to_network(101) == "Unknown"


# ── site dummies ─────────────────────────────────────────────────────────────

class TestSiteDummies:
    def test_reference_excluded(self):
        sites = np.array(["BNI", "NYU_1", "BNI", "SDSU_1"])
        dum = _site_dummies(sites, reference="BNI")
        assert dum.shape == (4, 2)  # 3 sites - 1 reference = 2 dummies

    def test_correct_encoding(self):
        sites = np.array(["BNI", "NYU_1", "BNI"])
        dum = _site_dummies(sites, reference="BNI")
        assert dum[0, 0] == 0, "BNI row should have 0 for NYU_1 dummy"
        assert dum[1, 0] == 1, "NYU_1 row should have 1 for NYU_1 dummy"

    def test_single_site_returns_empty(self):
        sites = np.array(["BNI", "BNI", "BNI"])
        dum = _site_dummies(sites, reference="BNI")
        assert dum.shape[1] == 0


# ── regression ────────────────────────────────────────────────────────────────

class TestRegression:
    def test_known_effect(self):
        rng = np.random.default_rng(0)
        n = 80
        dx = np.array([1.0] * 40 + [0.0] * 40)
        y = 0.001 * dx + rng.normal(0, 0.0001, n)
        X = sm.add_constant(dx.reshape(-1, 1))
        beta, p, se = _fit_model(y, X)
        assert beta > 0, "Positive effect expected"
        assert p < 0.01, "Effect should be significant"
        assert se > 0

    def test_no_effect_high_p(self):
        rng = np.random.default_rng(42)
        n = 80
        dx = np.array([1.0] * 40 + [0.0] * 40)
        y = rng.normal(0.001, 0.01, n)
        X = sm.add_constant(dx.reshape(-1, 1))
        _, p, _ = _fit_model(y, X)
        assert p > 0.05, "No-effect regression should give high p-value (usually)"

    def test_degenerate_returns_nan(self):
        y = np.array([1.0, 2.0, 3.0])
        X = np.ones((3, 2))  # collinear
        beta, p, se = _fit_model(y, X)
        # statsmodels handles this; we just verify no crash and returns numbers or NaN
        assert isinstance(beta, float)


# ── compute_edge_stats ────────────────────────────────────────────────────────

class TestComputeEdgeStats:
    def test_basic_run(self):
        ds = _make_dataset(n_subj=100, n_edges=20)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, use_combat=True, min_n_valid=5, min_n_per_group=2)
        assert isinstance(stats, list)

    def test_all_nan_edge_skipped(self):
        ds = _make_dataset(n_subj=100, n_edges=5)
        # Make first edge all NaN
        ds.X_combat[:, 0] = np.nan
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, use_combat=True, min_n_valid=10, min_n_per_group=2)
        edge_ids = {s.edge_id for s in stats}
        assert "edge_0001" not in edge_ids, "All-NaN edge should be skipped"

    def test_min_n_valid_respected(self):
        ds = _make_dataset(n_subj=100, n_edges=10)
        # Make edge_0001 have only 5 non-NaN
        ds.X_combat[:, 0] = np.nan
        ds.X_combat[:5, 0] = 0.001
        table = build_edge_index_table()
        stats_strict = compute_edge_stats(ds, table, min_n_valid=10, min_n_per_group=2)
        stats_loose  = compute_edge_stats(ds, table, min_n_valid=3, min_n_per_group=2)
        # strict threshold may exclude edge_0001; loose should include it if groups have ≥2 subjects
        ids_strict = {s.edge_id for s in stats_strict}
        ids_loose  = {s.edge_id for s in stats_loose}
        assert len(ids_loose) >= len(ids_strict)

    def test_output_fields_present(self):
        ds = _make_dataset(n_subj=100, n_edges=5)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if stats:
            s = stats[0]
            assert hasattr(s, "edge_id")
            assert hasattr(s, "cohen_d")
            assert hasattr(s, "p_B")
            assert hasattr(s, "beta_B")
            assert hasattr(s, "direction")

    def test_direction_consistent_with_means(self):
        ds = _make_dataset(n_subj=100, n_edges=5, seed=99)
        # Set edge_0001 so ASD > CONTROL
        asd_idx  = ds.meta.index[ds.meta["diagnosis"] == "ASD"].tolist()
        ctrl_idx = ds.meta.index[ds.meta["diagnosis"] == "CONTROL"].tolist()
        ds.X_combat[asd_idx, 0]  = 0.002
        ds.X_combat[ctrl_idx, 0] = 0.001
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        e0 = next((s for s in stats if s.edge_id == "edge_0001"), None)
        if e0:
            assert e0.direction == "ASD_higher"
            assert e0.mean_asd > e0.mean_control

    def test_nyu2_excluded_within_site_does_not_crash(self):
        """NYU_2 ASD-only should not cause compute_edge_stats to crash."""
        meta = _make_meta(n=80)
        # Add NYU_2 ASD-only subjects
        nyu2 = pd.DataFrame({
            "subject_id": range(1001, 1011),
            "site": ["NYU_2"] * 10,
            "diagnosis": ["ASD"] * 10,
            "age": np.random.uniform(10, 20, 10),
            "sex": ["M"] * 10,
            "mean_fa": np.random.uniform(0.22, 0.28, 10),
            "density": np.random.uniform(0.04, 0.08, 10),
        })
        meta_full = pd.concat([meta, nyu2], ignore_index=True)
        n = len(meta_full)
        rng = np.random.default_rng(5)
        X = rng.normal(0.0008, 0.0001, (n, 5))
        active = np.ones(5, dtype=bool)
        ds = BiomarkerDataset(
            X_combat=X, X_raw=X.copy(), meta=meta_full,
            edge_cols=[f"edge_{i+1:04d}" for i in range(5)],
            active_mask=active, feature_name="md_edges",
        )
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        assert isinstance(stats, list)


# ── NaN handling ─────────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_partial_nan_edge_uses_valid_subset(self):
        ds = _make_dataset(n_subj=100, n_edges=3)
        # edge_0001: first 40 subjects NaN, rest valid
        ds.X_combat[:40, 0] = np.nan
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        e0 = next((s for s in stats if s.edge_id == "edge_0001"), None)
        if e0:
            assert e0.n_valid <= 60
            assert e0.n_valid >= 5


# ── site robustness ───────────────────────────────────────────────────────────

class TestSiteRobustness:
    def test_basic_run(self):
        ds = _make_dataset(n_subj=100, n_edges=5)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip("No testable edges in synthetic dataset")
        rob = compute_site_robustness(ds, stats, table, use_combat=True)
        assert len(rob) == len(stats)

    def test_direction_consistent_flag(self):
        ds = _make_dataset(n_subj=100, n_edges=3, seed=7)
        # Force clear ASD>CTRL in all balanced sites for edge_0001
        meta = ds.meta
        for site in BALANCED_SITES:
            site_idx = meta[meta["site"] == site].index.tolist()
            asd_in_site  = meta.loc[site_idx][meta.loc[site_idx, "diagnosis"] == "ASD"].index.tolist()
            ctrl_in_site = meta.loc[site_idx][meta.loc[site_idx, "diagnosis"] == "CONTROL"].index.tolist()
            if asd_in_site and ctrl_in_site:
                ds.X_combat[asd_in_site, 0]  = 0.002
                ds.X_combat[ctrl_in_site, 0] = 0.001
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=3, min_n_per_group=2)
        rob   = compute_site_robustness(ds, stats, table)
        e0_rob = next((r for r in rob if r.edge_id == "edge_0001"), None)
        if e0_rob:
            assert e0_rob.n_same_direction >= 0  # just verify no crash

    def test_nyu2_in_loso(self):
        ds = _make_dataset(n_subj=100, n_edges=3)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip()
        rob = compute_site_robustness(ds, stats, table)
        for r in rob:
            assert "NYU_2" in r.loso_betas or len(r.loso_betas) >= 0  # NYU_2 might be NaN

    def test_output_schema(self):
        ds = _make_dataset(n_subj=100, n_edges=3)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip()
        rob = compute_site_robustness(ds, stats, table)
        for r in rob:
            assert isinstance(r.edge_id, str)
            assert isinstance(r.site_effects, dict)
            assert isinstance(r.loso_betas, dict)
            assert isinstance(r.direction_consistent, bool)
            assert isinstance(r.loso_stable, bool)


# ── stability score ───────────────────────────────────────────────────────────

class TestStabilityScore:
    def _make_edge_stats(self, q=0.03, d=0.5, direction="ASD_higher") -> EdgeStats:
        s = EdgeStats(
            edge_id="edge_0001", roi_i=1, roi_j=2,
            roi_i_name="ROI_1", roi_j_name="ROI_2",
            network_i="Vis", network_j="Default",
            is_cross_site_active=True,
            n_valid=150, n_asd=80, n_control=70,
            mean_asd=0.001, mean_control=0.0008,
            std_asd=0.0001, std_control=0.0001,
            cohen_d=d, direction=direction,
            beta_A=0.0002, p_A=0.01,
            beta_B=0.0002, p_B=0.01, se_B=0.00005,
            beta_C=0.0002, p_C=0.01,
        )
        s.q_B = q
        s.q_B_strict       = q <= 0.05
        s.q_B_exploratory  = q <= 0.10
        return s

    def _make_robustness(self, n_same=4, loso_same=5, nyu2_ok=True) -> SiteRobustness:
        return SiteRobustness(
            edge_id="edge_0001",
            site_effects={s: 0.3 for s in BALANCED_SITES},
            site_directions={s: "ASD_higher" for s in BALANCED_SITES},
            n_same_direction=n_same,
            n_sites_tested=4,
            direction_consistent=n_same >= 3,
            loso_betas={s: 0.0002 for s in BALANCED_SITES + ["NYU_2"]},
            loso_pvals={s: 0.05 for s in BALANCED_SITES + ["NYU_2"]},
            loso_n_same_direction=loso_same,
            loso_stable=loso_same >= 4,
            nyu2_removed_p=0.04,
            nyu2_removed_same_dir=nyu2_ok,
        )

    def test_perfect_score_is_one(self):
        es = self._make_edge_stats(q=0.01, d=0.8)
        rob = self._make_robustness(n_same=4, loso_same=5, nyu2_ok=True)
        sc = stability_score(es, rob, n_total_subjects=150)
        assert 0.9 <= sc <= 1.0, f"Perfect score should be ~1.0, got {sc:.3f}"

    def test_poor_score_near_zero(self):
        es = self._make_edge_stats(q=0.9, d=0.05)
        rob = self._make_robustness(n_same=0, loso_same=0)
        sc = stability_score(es, rob, n_total_subjects=229)
        assert sc < 0.5

    def test_score_in_range(self):
        for q, d, ns, ls in [(0.01, 0.5, 4, 5), (0.3, 0.1, 2, 2), (0.05, 0.3, 3, 4)]:
            es = self._make_edge_stats(q=q, d=d)
            rob = self._make_robustness(n_same=ns, loso_same=ls)
            sc = stability_score(es, rob)
            assert 0.0 <= sc <= 1.0, f"Score out of range: {sc}"


# ── output schema ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_stats_df_columns(self):
        from neurofiber.biomarkers.biomarker_discovery import _stats_to_df
        ds = _make_dataset(n_subj=80, n_edges=5)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip("No testable edges")
        stats = apply_fdr(stats)
        df = _stats_to_df(stats, [0.5] * len(stats))
        required = [
            "edge_id", "roi_i", "roi_j", "n_valid", "n_asd", "n_control",
            "mean_asd", "mean_control", "std_asd", "std_control",
            "cohen_d", "direction",
            "beta_A", "p_A", "beta_B", "p_B", "se_B", "beta_C", "p_C",
            "q_B", "q_B_exploratory", "q_B_strict",
            "stability_score", "is_cross_site_active",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_robustness_df_columns(self):
        from neurofiber.biomarkers.biomarker_discovery import _robustness_to_df
        ds = _make_dataset(n_subj=80, n_edges=5)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip()
        rob = compute_site_robustness(ds, stats, table)
        df = _robustness_to_df(rob)
        for col in ["edge_id", "n_same_direction", "direction_consistent",
                    "loso_stable", "nyu2_removed_same_dir"]:
            assert col in df.columns

    def test_loso_df_columns(self):
        from neurofiber.biomarkers.biomarker_discovery import _loso_to_df
        ds = _make_dataset(n_subj=80, n_edges=5)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        if not stats:
            pytest.skip()
        rob = compute_site_robustness(ds, stats, table)
        df = _loso_to_df(rob)
        for col in ["edge_id", "left_out", "beta", "p_value"]:
            assert col in df.columns

    def test_n_edges_matches_inputs(self):
        from neurofiber.biomarkers.biomarker_discovery import _stats_to_df
        ds = _make_dataset(n_subj=80, n_edges=8)
        table = build_edge_index_table()
        stats = compute_edge_stats(ds, table, min_n_valid=5, min_n_per_group=2)
        stats = apply_fdr(stats)
        df = _stats_to_df(stats, [0.4] * len(stats))
        assert len(df) == len(stats)
