"""Tests for Phase 3.4 — Conventional Atlas-Based Connectome Construction."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofiber.connectome.atlas_registration import RegisteredAtlas
from neurofiber.connectome.connectome_builder import (
    ConnectomeRecord,
    _build_adjacency_matrices,
    _compute_streamline_lengths,
    _sample_scalars_along_streamlines,
    _save_baseline_artifacts,
    build_subject_connectome,
)
from neurofiber.connectome.connectome_qc import save_summary_csvs
from neurofiber.connectome.graph_metrics import compute_graph_metrics, compute_node_metrics


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_label_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "index": list(range(1, n + 1)),
        "name":  [f"ROI_{i:02d}" for i in range(1, n + 1)],
    })


def _make_registered_atlas(
    tmp_path: Path,
    vol_shape: tuple = (6, 6, 6),
    n_rois: int = 2,
) -> RegisteredAtlas:
    """Two-ROI atlas: left half = ROI 1, right half = ROI 2."""
    affine = np.eye(4)
    atlas_data = np.zeros(vol_shape, dtype=np.int16)
    mid = vol_shape[0] // 2
    atlas_data[:mid, :, :] = 1
    atlas_data[mid:, :, :] = 2

    tmp_path.mkdir(parents=True, exist_ok=True)
    atlas_path = tmp_path / "atlas_subject_space.nii.gz"
    nib.save(nib.Nifti1Image(atlas_data, affine), str(atlas_path))
    label_df = pd.DataFrame({"index": [1, 2], "name": ["ROI_01", "ROI_02"]})

    return RegisteredAtlas(
        atlas_subject_space_path=atlas_path,
        affine=affine,
        n_rois=n_rois,
        label_df=label_df,
        registration_report={"atlas_n_rois": n_rois},
    )


def _make_subject_dti_dir(
    tmp_path: Path,
    n_streamlines: int = 20,
    vol_shape: tuple = (6, 6, 6),
    site: str = "bni",
    dataset: str = "ABIDEII-BNI_1",
    subject_id: str = "sub-001",
) -> Path:
    """
    Create a minimal subject DTI directory.

    Streamlines go from mid-ROI1 (x=1) to mid-ROI2 (x=4), connecting label 1 and 2.
    FA/MD/AD/RD are constant synthetic images.
    """
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_trk

    dti_dir = tmp_path / site / dataset / subject_id / "session_1" / "dti_1"
    (dti_dir / "tractography").mkdir(parents=True)
    (dti_dir / "tensor").mkdir(parents=True)

    affine  = np.eye(4)
    ref_img = nib.Nifti1Image(np.zeros(vol_shape, dtype=np.float32), affine)
    mid     = vol_shape[0] // 2
    cy, cz  = vol_shape[1] / 2, vol_shape[2] / 2

    # Straight lines: endpoint in ROI1 (x=1) → endpoint in ROI2 (x=mid+1)
    streamlines = [
        np.array([[1.0, cy, cz], [2.0, cy, cz], [float(mid + 1), cy, cz]], dtype=np.float32)
        for _ in range(n_streamlines)
    ]
    sft = StatefulTractogram(streamlines, ref_img, Space.RASMM)
    save_trk(sft, str(dti_dir / "tractography" / "streamlines.trk"), bbox_valid_check=False)

    fa_data = np.full(vol_shape, 0.5,   dtype=np.float32)
    md_data = np.full(vol_shape, 0.001, dtype=np.float32)
    for name, data in [("FA", fa_data), ("MD", md_data), ("AD", md_data), ("RD", md_data)]:
        nib.save(nib.Nifti1Image(data, affine), str(dti_dir / "tensor" / f"{name}.nii.gz"))

    return dti_dir


def _make_connectome_record(
    site: str = "bni",
    subject_id: str = "s001",
    status: str = "success",
    edge_count: int = 20,
) -> ConnectomeRecord:
    return ConnectomeRecord(
        site=site, dataset="ABIDEII-BNI_1", subject_id=subject_id,
        node_count=100, edge_count=edge_count, graph_density=0.1, mean_degree=5.0,
        mean_edge_fa=0.45, mean_edge_md=0.0008, mean_edge_length=80.0,
        streamline_count_total=500, mapped_streamline_count=400,
        unmapped_streamline_count=100, self_loop_count=10,
        mapping_success_ratio=0.8,
        status=status,
    )


# ---------------------------------------------------------------------------
# TestConnectomeRecord
# ---------------------------------------------------------------------------

class TestConnectomeRecord:

    def test_to_summary_row_has_required_fields(self):
        rec = _make_connectome_record()
        row = rec.to_summary_row()
        for key in [
            "site", "dataset", "subject_id", "node_count", "edge_count",
            "graph_density", "mean_degree", "mean_edge_fa", "mean_edge_md",
            "mean_edge_length", "streamline_count_total", "mapped_streamline_count",
            "unmapped_streamline_count", "self_loop_count", "mapping_success_ratio",
            "review_required", "status",
        ]:
            assert key in row, f"Missing key in summary row: {key}"

    def test_review_required_defaults_false(self):
        rec = _make_connectome_record()
        assert rec.review_required is False

    def test_timestamp_set_automatically(self):
        rec = _make_connectome_record()
        assert rec.timestamp is not None
        assert len(rec.timestamp) > 10  # ISO-8601 format


# ---------------------------------------------------------------------------
# TestStreamlineLengths
# ---------------------------------------------------------------------------

class TestStreamlineLengths:

    def test_known_geometry_two_mm(self):
        sl = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])]
        assert pytest.approx(_compute_streamline_lengths(sl)[0], abs=1e-5) == 2.0

    def test_single_point_is_zero(self):
        sl = [np.array([[5.0, 5.0, 5.0]])]
        assert _compute_streamline_lengths(sl)[0] == pytest.approx(0.0)

    def test_empty_list_returns_empty_array(self):
        assert len(_compute_streamline_lengths([])) == 0


# ---------------------------------------------------------------------------
# TestScalarSampling
# ---------------------------------------------------------------------------

class TestScalarSampling:

    def _uniform(self, val: float, shape=(8, 8, 8)):
        return np.full(shape, val, dtype=np.float32)

    def test_uniform_scalar_returns_correct_mean(self):
        sl = [np.array([[1.0, 4.0, 4.0], [2.0, 4.0, 4.0], [3.0, 4.0, 4.0]])]
        vals = _sample_scalars_along_streamlines(sl, self._uniform(0.6), np.eye(4))
        assert pytest.approx(vals[0], abs=0.05) == 0.6

    def test_zero_background_returns_nan(self):
        sl = [np.array([[1.0, 4.0, 4.0], [2.0, 4.0, 4.0]])]
        vals = _sample_scalars_along_streamlines(sl, np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))
        assert np.isnan(vals[0])

    def test_single_point_streamline_returns_nan(self):
        sl = [np.array([[4.0, 4.0, 4.0]])]
        vals = _sample_scalars_along_streamlines(sl, self._uniform(0.5), np.eye(4))
        assert np.isnan(vals[0])

    def test_multiple_streamlines_returns_correct_count(self):
        sls = [
            np.array([[1.0, 4.0, 4.0], [2.0, 4.0, 4.0]]),
            np.array([[3.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
        ]
        vals = _sample_scalars_along_streamlines(sls, self._uniform(0.3), np.eye(4))
        assert len(vals) == 2
        assert all(not np.isnan(v) for v in vals)


# ---------------------------------------------------------------------------
# TestAdjacencyMatrices
# ---------------------------------------------------------------------------

class TestAdjacencyMatrices:

    def _inputs(self):
        """3 streamlines: ROI1↔2 (×2) and ROI3↔4 (×1)."""
        n_rois = 4
        count_matrix = np.zeros((n_rois + 1, n_rois + 1), dtype=np.int32)
        count_matrix[1, 2] = count_matrix[2, 1] = 2
        count_matrix[3, 4] = count_matrix[4, 3] = 1
        mapping  = {(1, 2): [0, 1], (2, 1): [0, 1], (3, 4): [2], (4, 3): [2]}
        lengths  = np.array([50.0, 60.0, 70.0])
        fa       = np.array([0.4,  0.5,  0.6])
        md       = np.array([0.001, 0.002, 0.003])
        return n_rois, count_matrix, mapping, lengths, fa, md

    def test_count_matrix_fills_correctly(self):
        n_rois, cm, mapping, lengths, fa, md = self._inputs()
        adjs = _build_adjacency_matrices(n_rois, cm, mapping, [None]*3, lengths, fa, md, md, md)
        assert adjs["count"][0, 1] == 2
        assert adjs["count"][1, 0] == 2
        assert adjs["count"][2, 3] == 1

    def test_edge_stats_are_symmetric(self):
        n_rois, cm, mapping, lengths, fa, md = self._inputs()
        adjs = _build_adjacency_matrices(n_rois, cm, mapping, [None]*3, lengths, fa, md, md, md)
        for name in ["length", "fa", "md"]:
            assert adjs[name][0, 1] == pytest.approx(adjs[name][1, 0])

    def test_fa_edge_mean_correct(self):
        n_rois, cm, mapping, lengths, fa, md = self._inputs()
        adjs = _build_adjacency_matrices(n_rois, cm, mapping, [None]*3, lengths, fa, md, md, md)
        # ROI1↔2: streamlines 0 and 1, fa = [0.4, 0.5] → mean 0.45
        assert adjs["fa"][0, 1] == pytest.approx(0.45, abs=0.01)

    def test_self_loops_excluded_from_edge_stats(self):
        count_matrix = np.zeros((3, 3), dtype=np.int32)
        count_matrix[1, 1] = 10  # doubled by DIPY symmetric flag
        mapping  = {(1, 1): [0, 1, 2, 3, 4]}
        lengths  = np.ones(5) * 30.0
        fa       = np.ones(5) * 0.4
        adjs = _build_adjacency_matrices(2, count_matrix, mapping, [None]*5, lengths, fa, fa, fa, fa)
        # Diagonal entry of edge stats (self-loop) must remain NaN
        assert np.isnan(adjs["length"][0, 0])
        assert np.isnan(adjs["fa"][0, 0])


# ---------------------------------------------------------------------------
# TestBaselineArtifacts
# ---------------------------------------------------------------------------

class TestBaselineArtifacts:

    def _setup(self, n=5):
        atlas = np.zeros((8, 8, 8), dtype=np.float32)
        atlas[:4, :, :] = 1
        atlas[4:, :, :] = 2
        sls     = [np.array([[float(i), 4.0, 4.0], [float(i + 3), 4.0, 4.0]]) for i in range(n)]
        lengths = np.full(n, 30.0)
        fa      = np.full(n, 0.4)
        return atlas, sls, lengths, fa

    def test_all_three_artifact_files_created(self, tmp_path):
        atlas, sls, lengths, fa = self._setup()
        _save_baseline_artifacts(tmp_path, sls, lengths, fa, fa, atlas, np.eye(4), n_rois=2)
        art = tmp_path / "baseline_artifacts"
        assert (art / "streamline_endpoint_table.csv").exists()
        assert (art / "streamline_edge_assignment.csv").exists()
        assert (art / "edge_aggregation_report.json").exists()

    def test_endpoint_table_row_count_matches_streamlines(self, tmp_path):
        n = 7
        atlas, sls, lengths, fa = self._setup(n)
        _save_baseline_artifacts(tmp_path, sls, lengths, fa, fa, atlas, np.eye(4), n_rois=2)
        df = pd.read_csv(tmp_path / "baseline_artifacts" / "streamline_endpoint_table.csv")
        assert len(df) == n

    def test_aggregation_report_has_valid_ratio(self, tmp_path):
        atlas, sls, lengths, fa = self._setup(4)
        _save_baseline_artifacts(tmp_path, sls, lengths, fa, fa, atlas, np.eye(4), n_rois=2)
        report = json.loads(
            (tmp_path / "baseline_artifacts" / "edge_aggregation_report.json").read_text()
        )
        assert "mapping_success_ratio" in report
        assert 0.0 <= report["mapping_success_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# TestGraphMetrics
# ---------------------------------------------------------------------------

class TestGraphMetrics:

    def test_density_two_nodes_one_edge(self):
        adj = np.array([[0, 5], [5, 0]], dtype=float)
        m   = compute_graph_metrics(adj, _make_label_df(2))
        assert m["graph_density"] == pytest.approx(1.0)

    def test_edge_and_node_count(self):
        adj = np.array([[0, 5], [5, 0]], dtype=float)
        m   = compute_graph_metrics(adj, _make_label_df(2))
        assert m["edge_count"] == 1
        assert m["node_count"] == 2

    def test_disconnected_graph_reports_components(self):
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = 3
        m = compute_graph_metrics(adj, _make_label_df(4))
        # {0,1} connected, {2} isolated, {3} isolated → 3 components
        assert m["connected_components"] == 3

    def test_empty_adjacency_returns_zero_density(self):
        m = compute_graph_metrics(np.zeros((4, 4)), _make_label_df(4))
        assert m["edge_count"] == 0
        assert m["graph_density"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestNodeMetrics
# ---------------------------------------------------------------------------

class TestNodeMetrics:

    def test_returns_one_row_per_roi(self):
        n = 4
        df = compute_node_metrics(np.zeros((n, n)), _make_label_df(n))
        assert len(df) == n

    def test_required_columns_present(self):
        df = compute_node_metrics(np.zeros((3, 3)), _make_label_df(3))
        for col in ["roi_index", "name", "degree", "weighted_degree", "betweenness", "clustering"]:
            assert col in df.columns

    def test_hub_node_has_highest_degree(self):
        n = 4
        adj = np.zeros((n, n))
        for j in range(1, n):           # node 0 connects to all others
            adj[0, j] = adj[j, 0] = 1
        df = compute_node_metrics(adj, _make_label_df(n))
        assert df[df["roi_index"] == 1]["degree"].iloc[0] == n - 1


# ---------------------------------------------------------------------------
# TestBuildSubjectConnectome
# ---------------------------------------------------------------------------

class TestBuildSubjectConnectome:

    def test_happy_path_status_success(self, tmp_path):
        dti_dir = _make_subject_dti_dir(tmp_path, n_streamlines=20)
        atlas   = _make_registered_atlas(tmp_path / "atlas")
        rec     = build_subject_connectome(dti_dir, atlas, tmp_path / "out")

        assert rec.status == "success"
        assert rec.node_count == 2
        assert rec.edge_count == 1
        assert rec.mapped_streamline_count == 20
        assert rec.unmapped_streamline_count == 0

    def test_output_files_created(self, tmp_path):
        dti_dir = _make_subject_dti_dir(tmp_path, n_streamlines=10)
        atlas   = _make_registered_atlas(tmp_path / "atlas")
        out     = tmp_path / "out"
        build_subject_connectome(dti_dir, atlas, out)

        for fname in [
            "adjacency_streamline_count.npy",
            "adjacency_mean_fa.npy",
            "edge_table.csv",
            "node_table.csv",
            "graph_metrics.json",
            "connectome_report.json",
            "graph.graphml",
        ]:
            assert (out / fname).exists(), f"Missing output file: {fname}"

    def test_missing_trk_returns_failed(self, tmp_path):
        dti_dir = _make_subject_dti_dir(tmp_path, n_streamlines=10)
        (dti_dir / "tractography" / "streamlines.trk").unlink()
        rec = build_subject_connectome(dti_dir, _make_registered_atlas(tmp_path / "atlas"), tmp_path / "out")
        assert rec.status == "failed"
        assert rec.error_message

    def test_skip_if_exists_returns_success_without_reprocessing(self, tmp_path):
        dti_dir = _make_subject_dti_dir(tmp_path, n_streamlines=10)
        atlas   = _make_registered_atlas(tmp_path / "atlas")
        out     = tmp_path / "out"
        build_subject_connectome(dti_dir, atlas, out)
        rec2 = build_subject_connectome(dti_dir, atlas, out, skip_if_exists=True)
        assert rec2.status == "success"

    def test_known_outlier_flagged_for_review(self, tmp_path):
        dti_dir = _make_subject_dti_dir(tmp_path, subject_id="29012")
        atlas   = _make_registered_atlas(tmp_path / "atlas")
        rec     = build_subject_connectome(dti_dir, atlas, tmp_path / "out")
        assert rec.review_required is True

    def test_adjacency_matrix_shape_matches_n_rois(self, tmp_path):
        n_rois  = 2
        dti_dir = _make_subject_dti_dir(tmp_path, n_streamlines=10)
        atlas   = _make_registered_atlas(tmp_path / "atlas", n_rois=n_rois)
        out     = tmp_path / "out"
        build_subject_connectome(dti_dir, atlas, out)
        adj = np.load(str(out / "adjacency_streamline_count.npy"))
        assert adj.shape == (n_rois, n_rois)


# ---------------------------------------------------------------------------
# TestSaveConnectomeCSVs
# ---------------------------------------------------------------------------

class TestSaveConnectomeCSVs:

    def _records(self, n_sites=2, per_site=3):
        rows = []
        for i in range(n_sites):
            for j in range(per_site):
                rows.append(_make_connectome_record(site=f"site_{i}", subject_id=f"s{j:03d}"))
        return rows

    def test_both_csvs_created(self, tmp_path):
        subj_csv, site_csv = save_summary_csvs(self._records(), tmp_path)
        assert subj_csv.exists()
        assert site_csv.exists()

    def test_subject_csv_row_count(self, tmp_path):
        records  = self._records(n_sites=2, per_site=3)
        subj_csv, _ = save_summary_csvs(records, tmp_path)
        assert len(pd.read_csv(subj_csv)) == 6

    def test_site_csv_contains_all_sites(self, tmp_path):
        records  = self._records(n_sites=2, per_site=3)
        _, site_csv = save_summary_csvs(records, tmp_path)
        sites = set(pd.read_csv(site_csv)["site"].tolist())
        assert "site_0" in sites
        assert "site_1" in sites
