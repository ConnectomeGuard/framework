"""
Tests for Phase 3R.4 — Connectome QC and Harmonization Preparation

Coverage:
  - Matrix upper triangle vectorization correctness
  - Graph metric calculation
  - Zero-matrix handling (all-zero count_matrix)
  - Site summary schema
  - Harmonization metadata schema
  - Raw write guard
  - Subject-level failure isolation
  - Z-score edge table shape matches original
  - QC flag logic
  - Statistical test output schema
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.connectome.connectome_qc_harmonization import (
    ATLAS_N_ROIS,
    GRAPH_METRICS_FIELDS,
    HARMONIZATION_META_FIELDS,
    N_EDGES,
    SITE_EFFECT_FIELDS,
    STAT_TEST_FIELDS,
    GraphMetrics,
    compute_graph_metrics,
    compute_site_effects,
    compute_zscore_edge_table,
    extract_upper_triangle,
    load_matrices,
    run_statistical_tests,
    write_all_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_count_matrix(n: int = ATLAS_N_ROIS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m   = rng.integers(0, 10, (n, n)).astype(float)
    m   = np.triu(m, k=1)
    m   = m + m.T
    np.fill_diagonal(m, 0)
    return m


def _full_matrices(n: int = ATLAS_N_ROIS, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    count = _random_count_matrix(n, seed)
    nan_mask = count == 0
    length = rng.uniform(20, 200, (n, n)).astype(float)
    length[nan_mask] = np.nan
    length = np.triu(length, k=1); length = length + length.T
    fa_mat = rng.uniform(0.1, 0.6, (n, n)).astype(float)
    fa_mat[nan_mask] = np.nan
    fa_mat = np.triu(fa_mat, k=1); fa_mat = fa_mat + fa_mat.T
    md_mat = rng.uniform(0.0005, 0.001, (n, n)).astype(float)
    md_mat[nan_mask] = np.nan
    md_mat = np.triu(md_mat, k=1); md_mat = md_mat + md_mat.T
    return {"count": count, "length": length, "fa": fa_mat, "md": md_mat,
            "ad": None, "rd": None}


def _make_metrics(site: str = "BNI", n: int = 5) -> list[GraphMetrics]:
    rng = np.random.default_rng(42)
    mets = []
    for i in range(n):
        mats = _full_matrices(ATLAS_N_ROIS, seed=i)
        gm   = compute_graph_metrics(f"subj_{i}", site, "DATASET", mats)
        mets.append(gm)
    return mets


# ---------------------------------------------------------------------------
# TestUpperTriangleVectorization
# ---------------------------------------------------------------------------

class TestUpperTriangleVectorization:
    def test_output_length(self):
        m   = np.ones((ATLAS_N_ROIS, ATLAS_N_ROIS))
        vec = extract_upper_triangle(m)
        assert len(vec) == N_EDGES

    def test_known_values(self):
        n   = 4
        m   = np.arange(16, dtype=float).reshape(4, 4)
        vec = extract_upper_triangle(m, n=4)
        # Upper triangle (k=1): (0,1)=1, (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7, (2,3)=11
        expected = [1, 2, 3, 6, 7, 11]
        assert list(vec) == expected

    def test_length_matches_n_edges(self):
        m   = _random_count_matrix(ATLAS_N_ROIS)
        vec = extract_upper_triangle(m)
        assert len(vec) == N_EDGES


# ---------------------------------------------------------------------------
# TestGraphMetrics
# ---------------------------------------------------------------------------

class TestGraphMetrics:
    def test_density_in_range(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        assert gm.status == "success"
        assert 0.0 <= gm.density <= 1.0

    def test_nonzero_edges_non_negative(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        assert gm.nonzero_edges >= 0

    def test_degree_mean_non_negative(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        assert gm.degree_mean >= 0

    def test_zero_matrix_returns_success_with_zero_density(self):
        zero_mats = {"count": np.zeros((ATLAS_N_ROIS, ATLAS_N_ROIS)),
                     "length": None, "fa": None, "md": None, "ad": None, "rd": None}
        gm = compute_graph_metrics("s1", "BNI", "D", zero_mats)
        assert gm.status == "success"
        assert gm.nonzero_edges == 0
        assert gm.density == 0.0

    def test_zero_matrix_qc_flag(self):
        zero_mats = {"count": np.zeros((ATLAS_N_ROIS, ATLAS_N_ROIS)),
                     "length": None, "fa": None, "md": None, "ad": None, "rd": None}
        gm = compute_graph_metrics("s1", "BNI", "D", zero_mats)
        assert "edges<" in gm.qc_flag or "density<" in gm.qc_flag

    def test_fa_edge_mean_in_range(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        if gm.global_fa_edge_mean is not None:
            assert 0.0 <= gm.global_fa_edge_mean <= 1.0

    def test_isolated_nodes_non_negative(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        assert gm.isolated_nodes >= 0

    def test_connected_components_at_least_one(self):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        assert gm.connected_components >= 1


# ---------------------------------------------------------------------------
# TestQCFlags
# ---------------------------------------------------------------------------

class TestQCFlags:
    def test_high_density_flagged(self):
        mats = _full_matrices()
        mats["count"][:] = 50  # very high weights
        np.fill_diagonal(mats["count"], 0)
        gm = compute_graph_metrics("s1", "BNI", "D", mats)
        if gm.density > 0.10:
            assert "density>" in gm.qc_flag

    def test_low_density_flagged(self):
        zero_mats = {"count": np.zeros((ATLAS_N_ROIS, ATLAS_N_ROIS)),
                     "length": None, "fa": None, "md": None, "ad": None, "rd": None}
        gm = compute_graph_metrics("s1", "BNI", "D", zero_mats)
        assert "density<" in gm.qc_flag or "edges<" in gm.qc_flag


# ---------------------------------------------------------------------------
# TestSiteEffects
# ---------------------------------------------------------------------------

class TestSiteEffects:
    def test_site_effects_returns_rows(self):
        mets = _make_metrics("BNI", 5) + _make_metrics("NYU_1", 5)
        rows = compute_site_effects(mets, ["density"])
        assert len(rows) > 0

    def test_site_effects_has_required_fields(self):
        mets = _make_metrics("BNI", 5)
        rows = compute_site_effects(mets, ["density"])
        for field in SITE_EFFECT_FIELDS:
            for row in rows:
                assert field in row, f"Missing: {field}"


# ---------------------------------------------------------------------------
# TestStatisticalTests
# ---------------------------------------------------------------------------

class TestStatisticalTests:
    def test_stat_tests_returns_rows(self):
        mets = _make_metrics("BNI", 10) + _make_metrics("NYU_1", 10) + _make_metrics("SDSU_1", 10)
        rows = run_statistical_tests(mets, ["density"])
        assert len(rows) >= 1

    def test_stat_test_has_required_fields(self):
        mets = _make_metrics("BNI", 5) + _make_metrics("NYU_1", 5) + _make_metrics("SDSU_1", 5)
        rows = run_statistical_tests(mets, ["density"])
        for row in rows:
            for field in STAT_TEST_FIELDS:
                assert field in row, f"Missing: {field}"

    def test_p_value_between_0_and_1(self):
        mets = _make_metrics("BNI", 5) + _make_metrics("NYU_1", 5) + _make_metrics("SDSU_1", 5)
        rows = run_statistical_tests(mets, ["density"])
        for row in rows:
            assert 0.0 <= row["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# TestZscoreEdgeTable
# ---------------------------------------------------------------------------

class TestZscoreEdgeTable:
    def test_zscore_output_same_shape(self):
        header = ["subject_id", "site", "dataset", "diagnosis", "age", "sex"] + \
                 [f"edge_{i}" for i in range(N_EDGES)]
        meta = ["s1", "BNI", "D", "", "", ""]
        rng  = np.random.default_rng(0)
        edges = [str(v) for v in rng.random(N_EDGES)]
        rows  = [meta + edges, meta + edges]
        zrows = compute_zscore_edge_table(header, rows)
        assert len(zrows) == len(rows)
        assert len(zrows[0]) == len(rows[0])


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_write_raises_if_inside_raw(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        with pytest.raises((ValueError, AssertionError)):
            write_all_outputs(
                results={"metrics": [], "edge_tables": {k: ([], []) for k in ["count","length","fa","md"]},
                         "harmeta_rows": [], "site_effects": [], "stat_tests": []},
                out_root=raw,
                raw_root=raw,
            )


# ---------------------------------------------------------------------------
# TestSummarySchema
# ---------------------------------------------------------------------------

class TestSummarySchema:
    def test_graph_metrics_csv_columns(self, tmp_path):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        results = {
            "metrics":     [gm],
            "edge_tables": {k: ([], []) for k in ["count","length","fa","md"]},
            "harmeta_rows": [],
            "site_effects": [],
            "stat_tests":   [],
        }
        paths = write_all_outputs(results, tmp_path, tmp_path / "raw")
        with open(paths["graph_metrics"]) as f:
            cols = csv.DictReader(f).fieldnames
        for field in GRAPH_METRICS_FIELDS:
            assert field in cols, f"Missing: {field}"

    def test_harmonization_metadata_schema(self, tmp_path):
        mats = _full_matrices()
        gm   = compute_graph_metrics("s1", "BNI", "D", mats)
        harmeta = {f: "" for f in HARMONIZATION_META_FIELDS}
        harmeta["subject_id"] = "s1"; harmeta["site"] = "BNI"; harmeta["dataset"] = "D"
        results = {
            "metrics":      [gm],
            "edge_tables":  {k: ([], []) for k in ["count","length","fa","md"]},
            "harmeta_rows": [harmeta],
            "site_effects": [],
            "stat_tests":   [],
        }
        paths = write_all_outputs(results, tmp_path, tmp_path / "raw")
        with open(paths["harmonization_metadata"]) as f:
            cols = csv.DictReader(f).fieldnames
        for field in HARMONIZATION_META_FIELDS:
            assert field in cols, f"Missing: {field}"
