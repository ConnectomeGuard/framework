"""Tests for Phase 3.5A — Deterministic Tractography Baseline."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofiber.connectome.phase35a import (
    _LH_CUTOFF,
    _gini,
    build_qc_flags,
    compare_connectomes,
    compute_graph_properties,
    generate_phase35a_report,
)
from neurofiber.tractography.det_tractography import (
    DetRecord,
    _compute_lengths,
    _load_record_from_report,
    save_det_tractography_csvs,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_det_record(
    site: str = "bni",
    subject_id: str = "s001",
    status: str = "success",
    streamline_count: int = 5000,
    mean_length: float = 60.0,
    seed_count: int = 10_000,
) -> DetRecord:
    return DetRecord(
        site=site, dataset="ABIDEII-BNI_1", subject_id=subject_id,
        fa_threshold=0.10, interface_fa_low=0.08, interface_fa_high=0.20,
        seed_count=seed_count, streamline_count=streamline_count,
        mean_length_mm=mean_length, median_length_mm=55.0,
        min_length_mm=10.0, max_length_mm=150.0,
        output_file_size_mb=12.5, status=status,
    )


def _make_simple_adj(n: int = 10, density: float = 0.3) -> np.ndarray:
    """Symmetric adjacency matrix with ~density fraction of edges."""
    rng = np.random.default_rng(42)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.integers(1, 20)
                adj[i, j] = w
                adj[j, i] = w
    return adj


def _make_label_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "index": list(range(1, n + 1)),
        "name":  [f"ROI_{i:03d}" for i in range(1, n + 1)],
    })


def _make_subj_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "connectome_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _default_subj_row(site="bni", subject_id="s001", edge_count=200, density=0.04):
    return {
        "site": site, "dataset": "ABIDEII-BNI_1", "subject_id": subject_id,
        "edge_count": edge_count, "graph_density": density,
        "mean_edge_fa": 0.40, "mean_edge_md": 0.0007,
        "mean_edge_length": 80.0, "mapping_success_ratio": 0.10,
        "streamline_count_total": 10000, "mapped_streamline_count": 1000,
        "status": "success", "review_required": False,
    }


# ---------------------------------------------------------------------------
# TestDetRecord
# ---------------------------------------------------------------------------

class TestDetRecord:

    def test_fields_populated(self):
        rec = _make_det_record()
        assert rec.site == "bni"
        assert rec.fa_threshold == 0.10
        assert rec.interface_fa_low == 0.08
        assert rec.interface_fa_high == 0.20

    def test_to_dict_roundtrip(self):
        rec = _make_det_record()
        d = rec.to_dict()
        assert d["streamline_count"] == 5000
        assert d["status"] == "success"

    def test_to_summary_row_rounds_lengths(self):
        rec = _make_det_record(mean_length=60.12345)
        row = rec.to_summary_row()
        assert row["mean_length_mm"] == 60.12

    def test_timestamp_is_set(self):
        rec = _make_det_record()
        assert rec.timestamp is not None
        assert len(rec.timestamp) > 10

    def test_failed_record_has_error_message(self):
        rec = DetRecord(
            site="bni", dataset="d", subject_id="s1",
            fa_threshold=0.10, interface_fa_low=0.08, interface_fa_high=0.20,
            seed_count=0, streamline_count=0,
            mean_length_mm=None, median_length_mm=None,
            min_length_mm=None, max_length_mm=None,
            output_file_size_mb=None,
            status="failed", error_message="test error",
        )
        assert rec.status == "failed"
        assert rec.error_message == "test error"


# ---------------------------------------------------------------------------
# TestLoadRecordFromReport
# ---------------------------------------------------------------------------

class TestLoadRecordFromReport:

    def test_round_trip_via_dict(self):
        rec = _make_det_record(site="sdsu", subject_id="sub-42", streamline_count=3000)
        loaded = _load_record_from_report(rec.to_dict())
        assert loaded.site == "sdsu"
        assert loaded.subject_id == "sub-42"
        assert loaded.streamline_count == 3000
        assert loaded.status == "success"

    def test_missing_optional_fields_get_defaults(self):
        minimal = {"site": "bni", "dataset": "d", "subject_id": "s1",
                   "seed_count": 100, "streamline_count": 50}
        rec = _load_record_from_report(minimal)
        assert rec.fa_threshold == 0.10
        assert rec.interface_fa_low == 0.08
        assert rec.status == "success"


# ---------------------------------------------------------------------------
# TestComputeLengths
# ---------------------------------------------------------------------------

class TestComputeLengths:

    def test_straight_two_mm_segment(self):
        sl = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])]
        lengths = _compute_lengths(sl)
        assert pytest.approx(lengths[0], abs=1e-5) == 2.0

    def test_three_d_diagonal(self):
        # Diagonal from origin to (1,1,1): length = sqrt(3) ≈ 1.732
        sl = [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
        lengths = _compute_lengths(sl)
        assert pytest.approx(lengths[0], abs=1e-4) == np.sqrt(3)

    def test_single_point_is_zero(self):
        sl = [np.array([[0.0, 0.0, 0.0]])]
        lengths = _compute_lengths(sl)
        assert lengths[0] == 0.0

    def test_multiple_streamlines(self):
        sl = [
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]),  # length = 5.0
        ]
        lengths = _compute_lengths(sl)
        assert pytest.approx(lengths[0], abs=1e-5) == 5.0
        assert pytest.approx(lengths[1], abs=1e-5) == 5.0


# ---------------------------------------------------------------------------
# TestSaveDetTractographyCsvs
# ---------------------------------------------------------------------------

class TestSaveDetTractographyCsvs:

    def test_creates_both_files(self, tmp_path):
        records = [_make_det_record("bni", f"s{i:03d}") for i in range(3)]
        subj, site = save_det_tractography_csvs(records, tmp_path)
        assert subj.exists()
        assert site.exists()

    def test_subject_csv_row_count(self, tmp_path):
        records = [_make_det_record("bni", f"s{i:03d}") for i in range(5)]
        subj, _ = save_det_tractography_csvs(records, tmp_path)
        df = pd.read_csv(subj)
        assert len(df) == 5

    def test_site_csv_aggregates_correctly(self, tmp_path):
        records = (
            [_make_det_record("bni", f"s{i:03d}", streamline_count=1000) for i in range(3)] +
            [_make_det_record("nyu1", f"n{i:03d}", streamline_count=2000) for i in range(2)]
        )
        _, site = save_det_tractography_csvs(records, tmp_path)
        df = pd.read_csv(site)
        assert set(df["site"]) == {"bni", "nyu1"}
        bni_row = df[df["site"] == "bni"].iloc[0]
        assert bni_row["mean_streamline_count"] == pytest.approx(1000.0, abs=1.0)


# ---------------------------------------------------------------------------
# TestGini
# ---------------------------------------------------------------------------

class TestGini:

    def test_uniform_gini_zero(self):
        arr = np.ones(10)
        assert _gini(arr) == pytest.approx(0.0, abs=0.01)

    def test_maximal_skew(self):
        arr = np.zeros(9)
        arr = np.append(arr, 100.0)
        assert _gini(arr) > 0.80

    def test_empty_returns_zero(self):
        assert _gini(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# TestComputeGraphProperties
# ---------------------------------------------------------------------------

class TestComputeGraphProperties:

    def test_basic_structure(self):
        adj = _make_simple_adj(n=10, density=0.5)
        label_df = _make_label_df(n=10)
        props = compute_graph_properties(adj, label_df)
        assert props["n_nodes"] == 10
        assert props["n_edges"] > 0
        assert 0.0 <= props["density"] <= 1.0

    def test_hub_by_degree_length(self):
        adj = _make_simple_adj(n=10, density=0.8)
        label_df = _make_label_df(n=10)
        props = compute_graph_properties(adj, label_df)
        assert len(props["hub_by_degree"]) <= 10
        assert "roi_index" in props["hub_by_degree"][0]
        assert "name" in props["hub_by_degree"][0]

    def test_hemisphere_counts_sum_to_total_edges(self):
        adj = _make_simple_adj(n=10, density=0.5)
        label_df = _make_label_df(n=10)
        props = compute_graph_properties(adj, label_df)
        total = props["intra_lh_edges"] + props["intra_rh_edges"] + props["interhemispheric_edges"]
        assert total == props["n_edges"]

    def test_lh_cutoff_splits_correctly(self):
        # With 10 nodes: 0-4 = LH (indices < 5), 5-9 = RH
        # But _LH_CUTOFF is 50 for Schaefer100, so all 10 nodes are LH → 0 interhemispheric
        adj = np.zeros((10, 10), dtype=int)
        adj[0, 1] = adj[1, 0] = 5   # both LH (indices 0,1 < 50)
        label_df = _make_label_df(n=10)
        props = compute_graph_properties(adj, label_df)
        assert props["interhemispheric_edges"] == 0
        assert props["intra_lh_edges"] == 1

    def test_empty_graph_density_zero(self):
        adj = np.zeros((10, 10), dtype=int)
        label_df = _make_label_df(n=10)
        props = compute_graph_properties(adj, label_df)
        assert props["n_edges"] == 0
        assert props["density"] == 0.0

    def test_heavy_tail_flag_on_skewed_degree(self):
        # Star graph: node 0 connects to all others → very skewed
        n = 20
        adj = np.zeros((n, n), dtype=int)
        for i in range(1, n):
            adj[0, i] = adj[i, 0] = 1
        label_df = _make_label_df(n=n)
        props = compute_graph_properties(adj, label_df)
        assert props["heavy_tail_degree"] is True


# ---------------------------------------------------------------------------
# TestCompareConnectomes
# ---------------------------------------------------------------------------

class TestCompareConnectomes:

    def test_delta_columns_created(self, tmp_path):
        base_rows = [_default_subj_row("bni", f"s{i:03d}", edge_count=200) for i in range(5)]
        det_rows  = [_default_subj_row("bni", f"s{i:03d}", edge_count=180) for i in range(5)]
        base_csv = tmp_path / "base.csv"
        det_csv  = tmp_path / "det.csv"
        pd.DataFrame(base_rows).to_csv(base_csv, index=False)
        pd.DataFrame(det_rows).to_csv(det_csv, index=False)

        df = compare_connectomes(base_csv, det_csv)
        assert "delta_edge_count" in df.columns
        assert "pct_edge_count" in df.columns

    def test_delta_values_correct(self, tmp_path):
        base_rows = [_default_subj_row("bni", "s001", edge_count=200)]
        det_rows  = [_default_subj_row("bni", "s001", edge_count=250)]
        base_csv = tmp_path / "base.csv"
        det_csv  = tmp_path / "det.csv"
        pd.DataFrame(base_rows).to_csv(base_csv, index=False)
        pd.DataFrame(det_rows).to_csv(det_csv, index=False)

        df = compare_connectomes(base_csv, det_csv)
        assert df.iloc[0]["delta_edge_count"] == pytest.approx(50.0)
        assert df.iloc[0]["pct_edge_count"] == pytest.approx(25.0)

    def test_merge_on_site_and_subject_id(self, tmp_path):
        base_rows = [
            _default_subj_row("bni", "s001", edge_count=200),
            _default_subj_row("bni", "s002", edge_count=300),
        ]
        det_rows = [
            _default_subj_row("bni", "s001", edge_count=180),
            _default_subj_row("bni", "s002", edge_count=320),
        ]
        base_csv = tmp_path / "base.csv"
        det_csv  = tmp_path / "det.csv"
        pd.DataFrame(base_rows).to_csv(base_csv, index=False)
        pd.DataFrame(det_rows).to_csv(det_csv, index=False)

        df = compare_connectomes(base_csv, det_csv)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# TestBuildQcFlags
# ---------------------------------------------------------------------------

class TestBuildQcFlags:

    def test_failed_status_flagged(self):
        det_df = pd.DataFrame([_default_subj_row() | {"status": "failed"}])
        flags = build_qc_flags(det_df, pd.DataFrame())
        assert len(flags) == 1
        assert "failed" in flags.iloc[0]["reasons"]

    def test_zero_edges_flagged(self):
        det_df = pd.DataFrame([_default_subj_row() | {"edge_count": 0}])
        flags = build_qc_flags(det_df, pd.DataFrame())
        assert len(flags) == 1
        assert "zero_edges" in flags.iloc[0]["reasons"]

    def test_low_mapping_flagged(self):
        det_df = pd.DataFrame([_default_subj_row() | {"mapping_success_ratio": 0.01}])
        flags = build_qc_flags(det_df, pd.DataFrame())
        assert len(flags) == 1
        assert "critically_low_mapping" in flags.iloc[0]["reasons"]

    def test_large_regression_vs_baseline_flagged(self):
        det_df  = pd.DataFrame([_default_subj_row("bni", "s001", edge_count=50)])
        base_df = pd.DataFrame([_default_subj_row("bni", "s001", edge_count=200)])
        flags = build_qc_flags(det_df, base_df)
        assert len(flags) == 1
        assert "edge_count<50%_of_baseline" in flags.iloc[0]["reasons"]

    def test_healthy_subject_not_flagged(self):
        det_df  = pd.DataFrame([_default_subj_row("bni", "s001", edge_count=200)])
        base_df = pd.DataFrame([_default_subj_row("bni", "s001", edge_count=200)])
        flags = build_qc_flags(det_df, base_df)
        assert len(flags) == 0

    def test_multiple_reasons_combined(self):
        det_df = pd.DataFrame([_default_subj_row() | {
            "status": "failed", "edge_count": 0, "mapping_success_ratio": 0.001
        }])
        flags = build_qc_flags(det_df, pd.DataFrame())
        reasons = flags.iloc[0]["reasons"]
        assert "failed" in reasons
        assert "zero_edges" in reasons


# ---------------------------------------------------------------------------
# TestGeneratePhase35aReport
# ---------------------------------------------------------------------------

class TestGeneratePhase35aReport:

    def _make_site_csvs(self, tmp_path):
        det_rows = [
            {"site": "bni",  "subjects_processed": 58, "subjects_failed": 0,
             "mean_edge_count": 260, "mean_graph_density": 0.053,
             "mean_edge_fa": 0.42, "mean_edge_md": 0.0007,
             "mean_edge_length": 80.0, "mean_mapping_success_ratio": 0.14},
            {"site": "nyu1", "subjects_processed": 55, "subjects_failed": 0,
             "mean_edge_count": 300, "mean_graph_density": 0.061,
             "mean_edge_fa": 0.31, "mean_edge_md": 0.0009,
             "mean_edge_length": 36.0, "mean_mapping_success_ratio": 0.18},
        ]
        base_rows = [
            {"site": "bni",  "subjects_processed": 58, "subjects_failed": 0,
             "mean_edge_count": 242, "mean_graph_density": 0.049,
             "mean_edge_fa": 0.40, "mean_edge_md": 0.00065,
             "mean_edge_length": 92.0, "mean_mapping_success_ratio": 0.10},
            {"site": "nyu1", "subjects_processed": 55, "subjects_failed": 0,
             "mean_edge_count": 280, "mean_graph_density": 0.057,
             "mean_edge_fa": 0.29, "mean_edge_md": 0.00093,
             "mean_edge_length": 34.0, "mean_mapping_success_ratio": 0.16},
        ]
        det_csv  = tmp_path / "det_site.csv"
        base_csv = tmp_path / "base_site.csv"
        pd.DataFrame(det_rows).to_csv(det_csv, index=False)
        pd.DataFrame(base_rows).to_csv(base_csv, index=False)
        return det_csv, base_csv

    def test_report_file_created(self, tmp_path):
        det_csv, base_csv = self._make_site_csvs(tmp_path)
        report_path = tmp_path / "PHASE_3_5A_REPORT.md"
        generate_phase35a_report(
            det_site_csv=det_csv, base_site_csv=base_csv,
            comparison_df=pd.DataFrame(),
            graph_props=[],
            output_path=report_path,
            n_total=113, n_success=113, n_failed=0, n_review=2,
        )
        assert report_path.exists()

    def test_report_contains_key_sections(self, tmp_path):
        det_csv, base_csv = self._make_site_csvs(tmp_path)
        report_path = tmp_path / "PHASE_3_5A_REPORT.md"
        generate_phase35a_report(
            det_site_csv=det_csv, base_site_csv=base_csv,
            comparison_df=pd.DataFrame(),
            graph_props=[],
            output_path=report_path,
            n_total=113, n_success=113, n_failed=0, n_review=2,
        )
        text = report_path.read_text()
        assert "Phase 3.5A" in text
        assert "Site-Level Comparison" in text
        assert "Published Literature Comparison" in text
        assert "Conservative Behaviour" in text
        assert "Limitations" in text

    def test_report_shows_subject_counts(self, tmp_path):
        det_csv, base_csv = self._make_site_csvs(tmp_path)
        report_path = tmp_path / "PHASE_3_5A_REPORT.md"
        generate_phase35a_report(
            det_site_csv=det_csv, base_site_csv=base_csv,
            comparison_df=pd.DataFrame(),
            graph_props=[],
            output_path=report_path,
            n_total=113, n_success=110, n_failed=3, n_review=2,
        )
        text = report_path.read_text()
        assert "113" in text
        assert "110" in text
