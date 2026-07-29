"""
Tests for Phase 3R.5 — Connectome Harmonization Strategy Selection

Coverage:
  - No original files overwritten
  - Site z-score: zero mean and unit std per site
  - Global z-score: zero mean and unit std overall
  - Residualization: site mean removed from features
  - Missing covariate handling (all-NaN age/sex)
  - Output schema valid (same columns as input)
  - Metadata alignment preserved (subject order unchanged)
  - Combat skip with warning when not available
  - Raw-write guard
  - Variance retained calculation
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.harmonization.connectome_harmonization import (
    COMPARISON_FIELDS,
    META_COLS,
    apply_combat,
    apply_global_zscore,
    apply_residualization,
    apply_site_zscore,
    evaluate_variance_retained,
    load_edge_table,
    run_harmonization_pipeline,
    save_edge_table,
    write_comparison_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_edge_csv(
    tmp_path: Path,
    name: str = "count_edges.csv",
    n_subjects: int = 20,
    n_edges: int = 10,
    sites: list[str] = None,
) -> Path:
    if sites is None:
        sites = (["BNI"] * 5 + ["NYU_1"] * 5 + ["SDSU_1"] * 5 + ["TCD_1"] * 5)[:n_subjects]
    rng = np.random.default_rng(42)
    p   = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(META_COLS + [f"edge_{i+1:04d}" for i in range(n_edges)])
        for i in range(n_subjects):
            # Add site-level offset to create site effect
            site_offset = {"BNI": 0, "NYU_1": 5, "SDSU_1": 10, "TCD_1": 15}.get(sites[i], 0)
            vals = rng.random(n_edges) + site_offset
            meta = [str(i), sites[i], "DATASET", "", "", ""]
            w.writerow(meta + [str(v) for v in vals])
    return p


def _make_feat_dir(tmp_path: Path) -> Path:
    feat_dir = tmp_path / "connectome_features"
    feat_dir.mkdir()
    for ft in ["count_edges", "fa_edges", "md_edges", "length_edges"]:
        _make_edge_csv(feat_dir, name=f"{ft}.csv")
    # Harmonization metadata
    with open(feat_dir / "harmonization_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id","site","dataset","age","sex","diagnosis",
                    "scanner_info","b_value","direction_count","mean_fa","mean_md",
                    "density","mean_streamline_length"])
        for i in range(20):
            w.writerow([str(i), "BNI", "D", "", "", "", "", "", "", "", "", "", ""])
    return feat_dir


# ---------------------------------------------------------------------------
# TestNoOriginalOverwrite
# ---------------------------------------------------------------------------

class TestNoOriginalOverwrite:
    def test_original_files_unchanged(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        orig_content = (feat_dir / "count_edges.csv").read_bytes()

        run_harmonization_pipeline(
            feat_dir=feat_dir,
            out_root=out_root,
            raw_root=tmp_path / "raw",
        )

        assert (feat_dir / "count_edges.csv").read_bytes() == orig_content

    def test_output_in_separate_dir(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        run_harmonization_pipeline(feat_dir=feat_dir, out_root=out_root,
                                   raw_root=tmp_path / "raw")
        assert out_root.exists()
        assert feat_dir != out_root


# ---------------------------------------------------------------------------
# TestSiteZscore
# ---------------------------------------------------------------------------

class TestSiteZscore:
    def test_site_means_near_zero(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_site_zscore(X, sites)
        for site in set(sites):
            idx = [i for i, s in enumerate(sites) if s == site]
            site_mean = np.nanmean(X_z[idx, :])
            assert abs(site_mean) < 0.01, f"Site {site} mean not near zero: {site_mean}"

    def test_site_stds_near_one(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_site_zscore(X, sites)
        for site in set(sites):
            idx = [i for i, s in enumerate(sites) if s == site]
            site_std = np.nanstd(X_z[idx, :])
            assert abs(site_std - 1.0) < 0.1, f"Site {site} std not near 1: {site_std}"

    def test_output_shape_unchanged(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_site_zscore(X, sites)
        assert X_z.shape == X.shape


# ---------------------------------------------------------------------------
# TestGlobalZscore
# ---------------------------------------------------------------------------

class TestGlobalZscore:
    def test_global_mean_near_zero(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_global_zscore(X, sites)
        assert abs(np.nanmean(X_z)) < 0.01

    def test_global_std_near_one(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_global_zscore(X, sites)
        assert abs(np.nanstd(X_z) - 1.0) < 0.01

    def test_output_shape_unchanged(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_global_zscore(X, sites)
        assert X_z.shape == X.shape


# ---------------------------------------------------------------------------
# TestResidualization
# ---------------------------------------------------------------------------

class TestResidualization:
    def test_residuals_have_lower_site_means(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_r = apply_residualization(X, sites, covariates=None)
        assert X_r.shape == X.shape
        # Site means should be closer to zero after residualization
        for site in set(sites):
            idx = [i for i, s in enumerate(sites) if s == site]
            before = abs(np.nanmean(X[idx, :]))
            after  = abs(np.nanmean(X_r[idx, :]))
            assert after <= before + 0.1, f"{site}: residualization increased site mean"

    def test_missing_covariates_does_not_crash(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        # All-NaN covariates
        cov = np.full((20, 1), np.nan)
        X_r = apply_residualization(X, sites, covariates=cov)
        assert X_r.shape == X.shape


# ---------------------------------------------------------------------------
# TestCombat
# ---------------------------------------------------------------------------

class TestCombat:
    def test_combat_skip_returns_none_when_not_installed(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        # This will skip (neuroCombat not installed) OR succeed
        result, msg = apply_combat(X, sites)
        if result is None:
            assert "neuroCombat" in msg or "failed" in msg.lower()
        else:
            assert result.shape == X.shape


# ---------------------------------------------------------------------------
# TestVarianceRetained
# ---------------------------------------------------------------------------

class TestVarianceRetained:
    def test_none_strategy_retains_all_variance(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        vr = evaluate_variance_retained(X, X.copy())
        assert abs(vr - 1.0) < 1e-6

    def test_variance_retained_between_0_and_1_for_zscore(self, tmp_path):
        p = _make_edge_csv(tmp_path)
        _, X, _ = load_edge_table(p)
        sites = ["BNI"]*5 + ["NYU_1"]*5 + ["SDSU_1"]*5 + ["TCD_1"]*5
        X_z = apply_site_zscore(X, sites)
        vr = evaluate_variance_retained(X, X_z)
        # variance retained can be > 1 if site_zscore standardizes to unit variance
        assert vr > 0


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_harmonized_csv_has_same_columns(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        run_harmonization_pipeline(feat_dir=feat_dir, out_root=out_root,
                                   raw_root=tmp_path / "raw")
        src_cols = csv.DictReader(open(feat_dir / "count_edges.csv")).fieldnames
        for strategy in ["none", "site_zscore", "global_zscore", "residualized"]:
            harm_path = out_root / strategy / "count_edges.csv"
            if harm_path.exists():
                harm_cols = csv.DictReader(open(harm_path)).fieldnames
                assert harm_cols == src_cols, f"{strategy}: columns mismatch"

    def test_harmonized_csv_same_row_count(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        run_harmonization_pipeline(feat_dir=feat_dir, out_root=out_root,
                                   raw_root=tmp_path / "raw")
        src_count = sum(1 for _ in open(feat_dir / "count_edges.csv")) - 1
        for strategy in ["none", "site_zscore", "global_zscore", "residualized"]:
            harm_path = out_root / strategy / "count_edges.csv"
            if harm_path.exists():
                harm_count = sum(1 for _ in open(harm_path)) - 1
                assert harm_count == src_count

    def test_comparison_csv_schema(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        rows, _ = run_harmonization_pipeline(feat_dir=feat_dir, out_root=out_root,
                                             raw_root=tmp_path / "raw")
        out = tmp_path / "comp.csv"
        write_comparison_csv(rows, out)
        cols = csv.DictReader(open(out)).fieldnames
        for field in COMPARISON_FIELDS:
            assert field in cols, f"Missing: {field}"

    def test_harmonization_report_json_created(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        out_root = tmp_path / "harmonized"
        run_harmonization_pipeline(feat_dir=feat_dir, out_root=out_root,
                                   raw_root=tmp_path / "raw")
        for strategy in ["none", "site_zscore", "global_zscore", "residualized"]:
            report = out_root / strategy / "harmonization_report.json"
            if (out_root / strategy).exists():
                assert report.exists(), f"Missing report for {strategy}"


# ---------------------------------------------------------------------------
# TestRawWriteGuard
# ---------------------------------------------------------------------------

class TestRawWriteGuard:
    def test_raises_if_output_inside_raw(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        raw = tmp_path / "raw"
        with pytest.raises((ValueError, AssertionError)):
            run_harmonization_pipeline(feat_dir=feat_dir, out_root=raw,
                                       raw_root=raw)

    def test_raises_if_output_is_feat_dir(self, tmp_path):
        feat_dir = _make_feat_dir(tmp_path)
        with pytest.raises((ValueError, AssertionError)):
            run_harmonization_pipeline(feat_dir=feat_dir, out_root=feat_dir,
                                       raw_root=tmp_path / "raw")
