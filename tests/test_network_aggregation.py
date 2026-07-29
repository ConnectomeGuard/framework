"""
Tests for Phase 4.2 — network_aggregation.py

Covers: pair index building, aggregation, per-pair stats, FDR, LOSO,
stability summary, stats_to_df, and _recover_networks_from_label.
Uses synthetic data; does NOT hit the file system or run ComBat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from neurofiber.biomarkers.network_aggregation import (
    NETWORK_ORDER,
    NetworkPairInfo,
    _recover_networks_from_label,
    aggregate_to_network_pairs,
    apply_fdr_pairs,
    build_pair_index,
    compute_pair_stats,
    loso_network_pairs,
    loso_stability_summary,
    stats_to_df,
    site_means_per_pair,
    PairStats,
)
from neurofiber.biomarkers.biomarker_discovery import BiomarkerDataset


# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_meta(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic metadata with 4 balanced sites, each with equal ASD/Control."""
    rng = np.random.default_rng(seed)
    n_per_site = max(4, (n // 4 // 2) * 2)
    n_actual   = n_per_site * 4
    sites = (["BNI"] * n_per_site + ["NYU_1"] * n_per_site +
             ["SDSU_1"] * n_per_site + ["TCD_1"] * n_per_site)
    dx    = (["ASD"] * (n_per_site // 2) + ["CONTROL"] * (n_per_site // 2)) * 4
    ages  = rng.uniform(10.0, 50.0, n_actual)
    sexes = rng.choice(["M", "F"], n_actual).tolist()
    subj  = [f"sub_{i:03d}" for i in range(n_actual)]
    return pd.DataFrame({"site": sites, "diagnosis": dx,
                         "age": ages, "sex": sexes},
                        index=pd.Index(subj, name="subject_id"))


def _make_pair_index(n_pairs: int = 5) -> list[NetworkPairInfo]:
    """Simple synthetic pair index with 3 active edges each."""
    pairs = []
    for i, label in enumerate(["Cont_Default", "DorsAttn_SomMot", "Cont_Cont",
                                "Default_Vis", "Limbic_Vis"][:n_pairs]):
        ni, nj = _recover_networks_from_label(label)
        pairs.append(NetworkPairInfo(
            label=label, net_i=ni, net_j=nj,
            within=(ni == nj),
            n_active_edges=3,
            active_edge_positions=[i * 3, i * 3 + 1, i * 3 + 2],
        ))
    return pairs


def _make_feat_df(meta: pd.DataFrame, pair_index: list[NetworkPairInfo],
                  seed: int = 7) -> pd.DataFrame:
    """Random feature DataFrame, subjects × pairs."""
    rng = np.random.default_rng(seed)
    n_subj = len(meta)
    pair_labels = [p.label for p in pair_index]
    data = rng.normal(0.0007, 0.0001, (n_subj, len(pair_index)))
    # Introduce a small ASD effect on the first pair
    dx = (meta["diagnosis"].values == "ASD").astype(float)
    data[:, 0] += 0.00015 * dx
    df = pd.DataFrame(data, columns=pair_labels, index=meta.index)
    return df


def _make_pair_stats_list(
    pair_index: list[NetworkPairInfo], meta: pd.DataFrame, seed: int = 99
) -> list[PairStats]:
    """Create PairStats without running full regression."""
    rng = np.random.default_rng(seed)
    stats = []
    for p in pair_index:
        beta = rng.normal(0, 0.0001)
        stats.append(PairStats(
            label=p.label, net_i=p.net_i, net_j=p.net_j,
            within=p.within, n_active_edges=p.n_active_edges,
            n_valid_subjects=len(meta),
            n_asd=len(meta) // 2, n_control=len(meta) // 2,
            mean_asd=0.0008, mean_control=0.0007,
            std_asd=0.0001, std_control=0.0001,
            mean_coverage=0.90,
            cohen_d=float(beta / 0.0001),
            direction="ASD_higher" if beta >= 0 else "ASD_lower",
            beta_B=float(beta), p_B=float(rng.uniform(0.01, 0.99)), se_B=1e-5,
            q_B=np.nan, q_B_strict=False, q_B_exploratory=False,
        ))
    return stats


# ── _recover_networks_from_label ──────────────────────────────────────────────

class TestRecoverNetworksFromLabel:
    def test_simple_pair(self):
        ni, nj = _recover_networks_from_label("Cont_Default")
        assert {ni, nj} == {"Cont", "Default"}

    def test_within_pair(self):
        ni, nj = _recover_networks_from_label("Cont_Cont")
        assert ni == "Cont" and nj == "Cont"

    def test_salventattr(self):
        ni, nj = _recover_networks_from_label("Default_SalVentAttn")
        assert {ni, nj} == {"Default", "SalVentAttn"}

    def test_vis_pair(self):
        ni, nj = _recover_networks_from_label("SomMot_Vis")
        assert {ni, nj} == {"SomMot", "Vis"}

    def test_all_28_pairs_recoverable(self):
        """Every canonical sorted pair from NETWORK_ORDER must be parseable."""
        for i, n1 in enumerate(NETWORK_ORDER):
            for n2 in NETWORK_ORDER[i:]:
                label = "_".join(sorted([n1, n2]))
                ni, nj = _recover_networks_from_label(label)
                assert "_".join(sorted([ni, nj])) == label, \
                    f"Round-trip failed for {label}: got ({ni}, {nj})"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _recover_networks_from_label("Nonexistent_Network")


# ── aggregate_to_network_pairs ─────────────────────────────────────────────────

class TestAggregateToNetworkPairs:

    def _make_ds(self, n_subj=40, n_edges=30, seed=1):
        """Minimal BiomarkerDataset stub for aggregation tests."""
        rng = np.random.default_rng(seed)
        n_per_site = n_subj // 4
        meta = _make_meta(n_subj)
        edge_cols = [f"edge_{i+1:04d}" for i in range(n_edges)]
        X_raw = rng.normal(0.0007, 0.0001, (len(meta), n_edges))
        X_combat = X_raw.copy()
        active_mask = np.ones(n_edges, dtype=bool)
        ds = MagicMock(spec=BiomarkerDataset)
        ds.meta       = meta
        ds.X_raw      = X_raw
        ds.X_combat   = X_combat
        ds.edge_cols  = edge_cols
        ds.active_mask = active_mask
        return ds

    def test_output_shape(self):
        meta = _make_meta(40)
        pair_index = _make_pair_index(3)
        ds = self._make_ds(n_subj=len(meta))
        feat_df, cover_df = aggregate_to_network_pairs(ds, pair_index, use_combat=False)
        assert feat_df.shape  == (len(meta), 3)
        assert cover_df.shape == (len(meta), 3)

    def test_coverage_in_range(self):
        meta = _make_meta(40)
        pair_index = _make_pair_index(3)
        ds = self._make_ds(n_subj=len(meta))
        _, cover_df = aggregate_to_network_pairs(ds, pair_index, use_combat=False)
        vals = cover_df.values[~np.isnan(cover_df.values)]
        assert (vals >= 0).all() and (vals <= 1).all()

    def test_nan_propagation(self):
        """If all edge values for a subject-pair are NaN, feat is NaN."""
        meta = _make_meta(20)
        pair_index = _make_pair_index(2)
        ds = self._make_ds(n_subj=len(meta))
        # Set positions of first pair to NaN for first subject
        for pos in pair_index[0].active_edge_positions:
            ds.X_combat[0, pos] = np.nan
            ds.X_raw[0, pos]    = np.nan
        feat_df, _ = aggregate_to_network_pairs(ds, pair_index, use_combat=False)
        assert np.isnan(feat_df.iloc[0, 0])

    def test_mean_vs_median_differ(self):
        """Mean and median should differ when one edge has an extreme outlier."""
        meta = _make_meta(40)
        pair_index = _make_pair_index(2)
        ds = self._make_ds(n_subj=len(meta))
        # Keep first two positions near normal, inject outlier in third only
        positions = pair_index[0].active_edge_positions
        ds.X_combat[0, positions[0]] = 0.0007
        ds.X_combat[0, positions[1]] = 0.0007
        ds.X_combat[0, positions[2]] = 999.0    # massive outlier → mean ≠ median
        ds.X_raw[0, positions[0]] = 0.0007
        ds.X_raw[0, positions[1]] = 0.0007
        ds.X_raw[0, positions[2]] = 999.0
        feat_mean,   _ = aggregate_to_network_pairs(ds, pair_index, use_combat=False, agg="mean")
        feat_median, _ = aggregate_to_network_pairs(ds, pair_index, use_combat=False, agg="median")
        assert feat_mean.iloc[0, 0] != feat_median.iloc[0, 0]

    def test_columns_match_pair_labels(self):
        meta = _make_meta(40)
        pair_index = _make_pair_index(4)
        ds = self._make_ds(n_subj=len(meta))
        feat_df, _ = aggregate_to_network_pairs(ds, pair_index, use_combat=False)
        assert list(feat_df.columns) == [p.label for p in pair_index]


# ── compute_pair_stats ─────────────────────────────────────────────────────────

class TestComputePairStats:

    def test_returns_list_of_pair_stats(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(4)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        assert isinstance(stats, list)
        assert all(isinstance(s, PairStats) for s in stats)

    def test_p_values_in_unit_interval(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        for s in stats:
            assert 0.0 <= s.p_B <= 1.0

    def test_cohen_d_sign_matches_direction(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        for s in stats:
            if not np.isnan(s.cohen_d):
                expected = "ASD_higher" if s.mean_asd >= s.mean_control else "ASD_lower"
                assert s.direction == expected, f"Direction mismatch for {s.label}"

    def test_skips_pair_with_all_nan(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        feat_df = _make_feat_df(meta, pair_index)
        feat_df.iloc[:, 0] = np.nan     # First pair → all NaN
        stats = compute_pair_stats(feat_df, meta, pair_index)
        labels = [s.label for s in stats]
        assert pair_index[0].label not in labels

    def test_q_values_not_set_before_fdr(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        for s in stats:
            assert np.isnan(s.q_B)

    def test_within_network_flag_propagated(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)   # includes 'Cont_Cont' (within)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        within = [s for s in stats if s.label == "Cont_Cont"]
        if within:
            assert within[0].within is True

    def test_n_asd_n_control_positive(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(4)
        feat_df = _make_feat_df(meta, pair_index)
        stats = compute_pair_stats(feat_df, meta, pair_index)
        for s in stats:
            assert s.n_asd > 0 and s.n_control > 0


# ── apply_fdr_pairs ────────────────────────────────────────────────────────────

class TestApplyFdrPairs:

    def test_q_values_assigned(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        stats = _make_pair_stats_list(pair_index, meta)
        stats = apply_fdr_pairs(stats)
        for s in stats:
            assert not np.isnan(s.q_B)

    def test_q_values_in_unit_interval(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        stats = _make_pair_stats_list(pair_index, meta)
        stats = apply_fdr_pairs(stats)
        for s in stats:
            assert 0.0 <= s.q_B <= 1.0

    def test_q_geq_p(self):
        """Under BH, q ≥ p for the smallest p-value (conservative)."""
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        stats = _make_pair_stats_list(pair_index, meta)
        stats = apply_fdr_pairs(stats)
        for s in stats:
            assert s.q_B >= s.p_B - 1e-10

    def test_strict_flag_consistent(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        stats = _make_pair_stats_list(pair_index, meta)
        stats = apply_fdr_pairs(stats)
        for s in stats:
            assert s.q_B_strict == (s.q_B <= 0.05)
            assert s.q_B_exploratory == (s.q_B <= 0.10)

    def test_empty_list(self):
        result = apply_fdr_pairs([])
        assert result == []

    def test_single_entry(self):
        pair_index = _make_pair_index(1)
        meta = _make_meta(40)
        stats = _make_pair_stats_list(pair_index, meta)
        stats[0].p_B = 0.03
        result = apply_fdr_pairs(stats)
        assert not np.isnan(result[0].q_B)


# ── loso_network_pairs ─────────────────────────────────────────────────────────

class TestLosoNetworkPairs:

    def test_output_columns(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        feat_df = _make_feat_df(meta, pair_index)
        loso_df = loso_network_pairs(feat_df, meta, pair_index)
        assert set(loso_df.columns) == {"pair_label", "left_out", "beta", "p_value"}

    def test_row_count(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        feat_df = _make_feat_df(meta, pair_index)
        loso_df = loso_network_pairs(feat_df, meta, pair_index)
        n_sites = meta["site"].nunique()
        assert len(loso_df) == len(pair_index) * n_sites

    def test_each_site_appears(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(2)
        feat_df = _make_feat_df(meta, pair_index)
        loso_df = loso_network_pairs(feat_df, meta, pair_index)
        expected_sites = sorted(meta["site"].unique())
        assert sorted(loso_df["left_out"].unique()) == expected_sites

    def test_betas_are_float(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(2)
        feat_df = _make_feat_df(meta, pair_index)
        loso_df = loso_network_pairs(feat_df, meta, pair_index)
        valid = loso_df.dropna(subset=["beta"])
        assert valid["beta"].dtype in [np.float64, float]


# ── loso_stability_summary ─────────────────────────────────────────────────────

class TestLosoStabilitySummary:

    def _make_loso_df(self, pair_index, meta, same_sign: bool = True):
        """Synthetic LOSO DF where all betas have the same sign."""
        sites = sorted(meta["site"].unique())
        rows = []
        sign = 1.0 if same_sign else -1.0
        for p in pair_index:
            for s in sites:
                rows.append({
                    "pair_label": p.label, "left_out": s,
                    "beta": sign * 1e-4, "p_value": 0.05,
                })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        stats = _make_pair_stats_list(pair_index, meta)
        stats = apply_fdr_pairs(stats)
        loso_df = self._make_loso_df(pair_index, meta)
        summary = loso_stability_summary(stats, loso_df)
        expected_cols = {"pair_label", "loso_n_same_sign", "loso_n_run",
                         "loso_stable", "nyu2_removed_p", "nyu2_removed_same_dir"}
        assert expected_cols.issubset(set(summary.columns))

    def test_stable_when_all_same_sign(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(2)
        stats = _make_pair_stats_list(pair_index, meta)
        for s in stats:
            s.beta_B = 1e-4   # positive
        loso_df = self._make_loso_df(pair_index, meta, same_sign=True)
        summary = loso_stability_summary(stats, loso_df)
        assert summary["loso_stable"].all()

    def test_not_stable_when_all_flipped(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(2)
        stats = _make_pair_stats_list(pair_index, meta)
        for s in stats:
            s.beta_B = 1e-4   # positive global
        # LOSO betas all negative → all flipped
        loso_df = self._make_loso_df(pair_index, meta, same_sign=False)
        summary = loso_stability_summary(stats, loso_df)
        assert not summary["loso_stable"].any()

    def test_row_count_matches_stats(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(4)
        stats = _make_pair_stats_list(pair_index, meta)
        loso_df = self._make_loso_df(pair_index, meta)
        summary = loso_stability_summary(stats, loso_df)
        assert len(summary) == len(stats)


# ── stats_to_df ────────────────────────────────────────────────────────────────

class TestStatsToDf:

    def test_returns_dataframe(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        stats = _make_pair_stats_list(pair_index, meta)
        df = stats_to_df(stats)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(stats)

    def test_required_columns_present(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        stats = _make_pair_stats_list(pair_index, meta)
        df = stats_to_df(stats)
        for col in ["pair_label", "net_i", "net_j", "cohen_d", "beta_B", "p_B", "q_B"]:
            assert col in df.columns

    def test_loso_columns_added_when_provided(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        stats = _make_pair_stats_list(pair_index, meta)
        sites = sorted(meta["site"].unique())
        rows = [{"pair_label": p.label, "left_out": s,
                 "beta": 1e-4, "p_value": 0.1}
                for p in pair_index for s in sites]
        loso_df = pd.DataFrame(rows)
        loso_summary = loso_stability_summary(stats, loso_df)
        df = stats_to_df(stats, loso_summary)
        assert "loso_stable" in df.columns


# ── site_means_per_pair ────────────────────────────────────────────────────────

class TestSiteMeansPerPair:

    def test_output_shape(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(3)
        feat_df = _make_feat_df(meta, pair_index)
        pair_labels = [p.label for p in pair_index]
        site_df = site_means_per_pair(feat_df, meta, pair_labels)
        n_sites = meta["site"].nunique()
        assert site_df.shape == (n_sites, len(pair_labels))

    def test_index_is_sites(self):
        meta = _make_meta(80)
        pair_index = _make_pair_index(2)
        feat_df = _make_feat_df(meta, pair_index)
        pair_labels = [p.label for p in pair_index]
        site_df = site_means_per_pair(feat_df, meta, pair_labels)
        assert set(site_df.index) == set(meta["site"].unique())


# ── integration-level smoke test ──────────────────────────────────────────────

class TestNetworkAggregationIntegration:
    """End-to-end smoke test using synthetic BiomarkerDataset stub."""

    def _make_ds(self, seed=1):
        rng = np.random.default_rng(seed)
        meta = _make_meta(80)
        n_subj = len(meta)
        n_edges = 50
        edge_cols = [f"edge_{i+1:04d}" for i in range(n_edges)]
        X = rng.normal(0.0007, 0.0001, (n_subj, n_edges))
        active_mask = np.ones(n_edges, dtype=bool)
        ds = MagicMock(spec=BiomarkerDataset)
        ds.meta       = meta
        ds.X_raw      = X
        ds.X_combat   = X.copy()
        ds.edge_cols  = edge_cols
        ds.active_mask = active_mask
        return ds

    def test_pair_index_has_28_or_fewer(self):
        """With synthetic edges we may get fewer than 28 non-empty pairs."""
        ds = self._make_ds()
        with patch("neurofiber.biomarkers.network_aggregation.build_edge_index_table") as mock_idx:
            # Build minimal edge_index_table with 50 edges spanning a few pairs
            rows = []
            for i in range(50):
                # Assign alternating ROIs from two networks (Cont and Default)
                roi_i = 41 + (i % 6)   # Cont parcels 41-46
                roi_j = 47 + (i % 4)   # Default parcels 47-50
                rows.append({"edge_idx": i, "roi_i": roi_i, "roi_j": roi_j})
            mock_idx.return_value = pd.DataFrame(rows)
            pair_index = build_pair_index(ds)
        assert len(pair_index) >= 1

    def test_full_pipeline_smoke(self):
        """Run compute_pair_stats + apply_fdr_pairs + loso + summary end-to-end."""
        meta = _make_meta(80)
        pair_index = _make_pair_index(5)
        feat_df = _make_feat_df(meta, pair_index)

        stats = compute_pair_stats(feat_df, meta, pair_index)
        assert len(stats) >= 1

        stats = apply_fdr_pairs(stats)
        for s in stats:
            assert not np.isnan(s.q_B)

        loso_df = loso_network_pairs(feat_df, meta, pair_index)
        assert len(loso_df) > 0

        loso_sum = loso_stability_summary(stats, loso_df)
        assert len(loso_sum) == len(stats)

        df = stats_to_df(stats, loso_sum)
        assert len(df) == len(stats)
        assert "loso_stable" in df.columns
