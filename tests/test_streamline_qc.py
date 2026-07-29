"""Tests for Phase 3.3 — Streamline QC + Site-Normalized Tractography Metrics."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofiber.tractography.streamline_qc import (
    CLEAN_DTI_SITES_DISPLAY,
    StreamlineQCRecord,
    _compute_lengths_mm,
    compute_site_normalizations,
    generate_qc_plots,
    load_subject_qc,
    run_qc_batch,
    save_summary_csvs,
    site_to_folder,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_synthetic_trk(path: Path, n_streamlines: int = 50, rng_seed: int = 42) -> None:
    """Save a synthetic .trk file with streamlines of varied lengths."""
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_trk

    rng = np.random.default_rng(rng_seed)
    streamlines = []
    for _ in range(n_streamlines):
        n_pts = int(rng.integers(5, 25))
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-9
        pts = np.cumsum(
            direction[None, :] * np.ones((n_pts, 1)) * 1.5, axis=0
        ).astype(np.float32)
        pts += rng.standard_normal((n_pts, 3)).astype(np.float32) * 0.05
        streamlines.append(pts)

    ref = nib.Nifti1Image(np.zeros((10, 10, 6), dtype=np.float32), np.eye(4))
    sft = StatefulTractogram(streamlines, ref, Space.RASMM)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_trk(sft, str(path), bbox_valid_check=False)


def _make_subject_tree(
    root: Path,
    site: str = "bni",
    dataset: str = "ABIDEII-BNI_1",
    subject_id: str = "sub-001",
    n_streamlines: int = 50,
    create_trk: bool = True,
) -> Path:
    """Create a minimal subject tree with optional .trk file."""
    dti_dir   = root / site / dataset / subject_id / "session_1" / "dti_1"
    tract_dir = dti_dir / "tractography"
    tract_dir.mkdir(parents=True, exist_ok=True)

    if create_trk:
        _make_synthetic_trk(tract_dir / "streamlines.trk", n_streamlines=n_streamlines)

    return dti_dir


def _make_success_record(
    site: str = "bni",
    subject_id: str = "s001",
    streamline_count: int = 3000,
    mean_length: float = 80.0,
) -> StreamlineQCRecord:
    return StreamlineQCRecord(
        site=site, dataset="ABIDEII-BNI_1", subject_id=subject_id,
        streamline_count=streamline_count,
        mean_length_mm=mean_length,  median_length_mm=mean_length - 2.0,
        std_length_mm=15.0,
        p10_length_mm=50.0, p25_length_mm=65.0,
        p75_length_mm=95.0, p90_length_mm=110.0,
        min_length_mm=5.0,  max_length_mm=180.0,
        iqr_length_mm=30.0,
    )


# ---------------------------------------------------------------------------
# TestSiteNormalisation
# ---------------------------------------------------------------------------

class TestSiteNormalisation:

    def test_display_names_mapped(self):
        assert site_to_folder("BNI")    == "bni"
        assert site_to_folder("NYU_1")  == "nyu1"
        assert site_to_folder("NYU_2")  == "nyu2"
        assert site_to_folder("SDSU_1") == "sdsu"
        assert site_to_folder("TCD_1")  == "tcd"
        assert site_to_folder("IP_1")   == "ip"

    def test_folder_names_pass_through(self):
        assert site_to_folder("bni")  == "bni"
        assert site_to_folder("nyu1") == "nyu1"

    def test_clean_sites_constant(self):
        for expected in ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]:
            assert expected in CLEAN_DTI_SITES_DISPLAY
        assert "IP_1" not in CLEAN_DTI_SITES_DISPLAY


# ---------------------------------------------------------------------------
# TestLengthComputation
# ---------------------------------------------------------------------------

class TestLengthComputation:

    def test_straight_streamline(self):
        s = np.array([[0, 0, i] for i in range(11)], dtype=float)
        assert abs(_compute_lengths_mm([s])[0] - 10.0) < 1e-6

    def test_single_point_is_zero(self):
        s = np.array([[1, 2, 3]], dtype=float)
        assert _compute_lengths_mm([s])[0] == 0.0

    def test_multiple_streamlines(self):
        s1 = np.array([[0, 0, i] for i in range(6)],  dtype=float)   # 5 mm
        s2 = np.array([[0, 0, i] for i in range(11)], dtype=float)   # 10 mm
        lengths = _compute_lengths_mm([s1, s2])
        assert abs(lengths[0] - 5.0)  < 1e-6
        assert abs(lengths[1] - 10.0) < 1e-6

    def test_diagonal_step(self):
        # step = (1,1,0) → ||step|| = sqrt(2) → 4 steps → total sqrt(2)*4 ≈ 5.657
        s = np.array([[i, i, 0] for i in range(5)], dtype=float)
        expected = 4.0 * np.sqrt(2.0)
        assert abs(_compute_lengths_mm([s])[0] - expected) < 1e-5


# ---------------------------------------------------------------------------
# TestQCRecordSchema
# ---------------------------------------------------------------------------

class TestQCRecordSchema:

    def test_to_summary_row_has_all_fields(self):
        rec = _make_success_record()
        row = rec.to_summary_row()
        expected = {
            "site", "dataset", "subject_id",
            "streamline_count", "mean_length_mm", "median_length_mm",
            "std_length_mm", "p10_length_mm", "p25_length_mm",
            "p75_length_mm", "p90_length_mm", "min_length_mm",
            "max_length_mm", "iqr_length_mm",
            "z_streamline_count", "z_mean_length_mm", "z_p90_length_mm",
            "qc_outlier", "qc_reason", "status", "error_message",
        }
        assert expected == set(row.keys())

    def test_failed_record_schema(self):
        rec = StreamlineQCRecord(
            site="bni", dataset="d", subject_id="s",
            streamline_count=0,
            mean_length_mm=0.0, median_length_mm=0.0, std_length_mm=0.0,
            p10_length_mm=0.0, p25_length_mm=0.0, p75_length_mm=0.0,
            p90_length_mm=0.0, min_length_mm=0.0, max_length_mm=0.0,
            iqr_length_mm=0.0,
            status="failed", error_message="file missing",
        )
        row = rec.to_summary_row()
        assert row["status"] == "failed"
        assert row["error_message"] == "file missing"


# ---------------------------------------------------------------------------
# TestLoadSubjectQC
# ---------------------------------------------------------------------------

class TestLoadSubjectQC:

    def test_missing_trk_returns_failed(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path, create_trk=False)
        rec = load_subject_qc(dti_dir)
        assert rec.status == "failed"
        assert "streamlines.trk" in (rec.error_message or "")

    def test_success_case_metrics_populated(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path, n_streamlines=40)
        rec = load_subject_qc(dti_dir)
        assert rec.status == "success"
        assert rec.streamline_count == 40
        assert rec.mean_length_mm > 0
        assert rec.std_length_mm >= 0
        assert rec.p10_length_mm <= rec.median_length_mm <= rec.p90_length_mm
        assert rec.iqr_length_mm == pytest.approx(rec.p75_length_mm - rec.p25_length_mm)

    def test_success_path_components_extracted(self, tmp_path):
        dti_dir = _make_subject_tree(
            tmp_path, site="sdsu", dataset="ABIDEII-SDSU_1", subject_id="sub-42",
        )
        rec = load_subject_qc(dti_dir)
        assert rec.site       == "sdsu"
        assert rec.dataset    == "ABIDEII-SDSU_1"
        assert rec.subject_id == "sub-42"

    def test_iqr_equals_p75_minus_p25(self, tmp_path):
        dti_dir = _make_subject_tree(tmp_path, n_streamlines=80)
        rec = load_subject_qc(dti_dir)
        assert rec.status == "success"
        assert abs(rec.iqr_length_mm - (rec.p75_length_mm - rec.p25_length_mm)) < 1e-4


# ---------------------------------------------------------------------------
# TestSiteNormalization
# ---------------------------------------------------------------------------

class TestSiteNormalization:

    def _make_site_records(
        self,
        site: str,
        counts: list[int],
        lengths: list[float],
    ) -> list[StreamlineQCRecord]:
        records = []
        for i, (c, l) in enumerate(zip(counts, lengths)):
            r = _make_success_record(site=site, subject_id=f"s{i:03d}",
                                     streamline_count=c, mean_length=l)
            records.append(r)
        return records

    def test_z_scores_have_zero_mean(self):
        recs = self._make_site_records("bni", [3000, 3200, 2800, 3100, 2900],
                                              [80.0,  85.0,  75.0,  82.0,  78.0])
        out  = compute_site_normalizations(recs)
        z_c  = [r.z_streamline_count for r in out if r.z_streamline_count is not None]
        z_l  = [r.z_mean_length_mm   for r in out if r.z_mean_length_mm   is not None]
        assert abs(np.mean(z_c)) < 1e-6
        assert abs(np.mean(z_l)) < 1e-6

    def test_known_zscore_value(self):
        # Uniform counts: all equal → std = 0 → z = 0 / 1e-9 ≈ 0
        recs = self._make_site_records("bni", [3000, 3000, 3000], [80.0, 80.0, 80.0])
        out  = compute_site_normalizations(recs)
        for r in out:
            assert abs(r.z_streamline_count) < 1e-3

    def test_single_extreme_outlier_flagged(self):
        # 19 normal subjects + 1 outlier.
        # z_outlier → sqrt(N-1) = sqrt(19) ≈ 4.36 > 3.0 threshold.
        recs = self._make_site_records(
            "bni",
            [3000] * 19 + [30000],
            [80.0] * 20,
        )
        out = compute_site_normalizations(recs)
        flagged = [r for r in out if r.qc_outlier]
        assert len(flagged) == 1
        assert flagged[0].streamline_count == 30000

    def test_normal_subjects_not_flagged(self):
        recs = self._make_site_records("bni",
                                       [2900, 3000, 3100, 3050, 2950],
                                       [79.0,  80.0,  81.0,  80.5,  79.5])
        out = compute_site_normalizations(recs)
        assert not any(r.qc_outlier for r in out)

    def test_failed_records_excluded_from_site_stats(self):
        good = self._make_site_records("bni", [3000, 3200], [80.0, 82.0])
        bad  = StreamlineQCRecord(
            site="bni", dataset="d", subject_id="bad",
            streamline_count=0,
            mean_length_mm=0.0, median_length_mm=0.0, std_length_mm=0.0,
            p10_length_mm=0.0, p25_length_mm=0.0, p75_length_mm=0.0,
            p90_length_mm=0.0, min_length_mm=0.0, max_length_mm=0.0,
            iqr_length_mm=0.0, status="failed",
        )
        out = compute_site_normalizations(good + [bad])
        assert bad.z_streamline_count is None  # failed record not normalized

    def test_multi_site_z_scores_are_site_local(self):
        bni_recs = self._make_site_records("bni",  [3000, 3200, 2800], [80.0, 85.0, 75.0])
        nyu_recs = self._make_site_records("nyu1", [3900, 4100, 3800], [35.0, 38.0, 32.0])
        out = compute_site_normalizations(bni_recs + nyu_recs)
        bni_z_l  = [r.z_mean_length_mm for r in out if r.site == "bni"  and r.z_mean_length_mm is not None]
        nyu_z_l  = [r.z_mean_length_mm for r in out if r.site == "nyu1" and r.z_mean_length_mm is not None]
        # Both sites' z-scores should have zero mean independently
        assert abs(np.mean(bni_z_l)) < 1e-5
        assert abs(np.mean(nyu_z_l)) < 1e-5


# ---------------------------------------------------------------------------
# TestBatchRunner
# ---------------------------------------------------------------------------

class TestBatchRunner:

    def test_missing_site_dir_returns_empty(self, tmp_path):
        proc = tmp_path / "proc"
        proc.mkdir()
        records = run_qc_batch(processed_root=proc, sites=["bni"])
        assert records == []

    def test_success_batch(self, tmp_path):
        proc = tmp_path / "proc"
        _make_subject_tree(proc, subject_id="sub-001", n_streamlines=30)
        _make_subject_tree(proc, subject_id="sub-002", n_streamlines=40)

        records = run_qc_batch(processed_root=proc, sites=["bni"])
        assert len(records) == 2
        assert all(r.status == "success" for r in records)

    def test_missing_trk_in_batch_does_not_crash(self, tmp_path):
        proc = tmp_path / "proc"
        _make_subject_tree(proc, subject_id="sub-001", n_streamlines=30)
        _make_subject_tree(proc, subject_id="sub-002", create_trk=False)

        records = run_qc_batch(processed_root=proc, sites=["bni"])
        assert len(records) == 2
        statuses = {r.subject_id: r.status for r in records}
        assert statuses["sub-001"] == "success"
        assert statuses["sub-002"] == "failed"

    def test_display_site_name_resolved(self, tmp_path):
        proc = tmp_path / "proc"
        _make_subject_tree(proc, site="bni", n_streamlines=20)
        records = run_qc_batch(processed_root=proc, sites=["BNI"])  # display name
        assert len(records) == 1
        assert records[0].status == "success"


# ---------------------------------------------------------------------------
# TestQCPlots
# ---------------------------------------------------------------------------

class TestQCPlots:

    def _make_records_with_zscores(self, tmp_path) -> list[StreamlineQCRecord]:
        proc = tmp_path / "proc"
        for i in range(4):
            _make_subject_tree(proc, subject_id=f"sub-{i:03d}", n_streamlines=30 + i * 5)
        records = run_qc_batch(processed_root=proc, sites=["bni"])
        return compute_site_normalizations(records)

    def test_plots_generated(self, tmp_path):
        records   = self._make_records_with_zscores(tmp_path)
        plot_dir  = tmp_path / "plots"
        plots     = generate_qc_plots(records, plot_dir)
        assert len(plots) >= 2  # comparison + percentiles (KDE needs scipy)
        for p in plots:
            assert p.exists()
            assert p.suffix == ".png"

    def test_site_comparison_plot_exists(self, tmp_path):
        records  = self._make_records_with_zscores(tmp_path)
        plot_dir = tmp_path / "plots"
        generate_qc_plots(records, plot_dir)
        assert (plot_dir / "qc_site_comparison.png").exists()

    def test_percentiles_plot_exists(self, tmp_path):
        records  = self._make_records_with_zscores(tmp_path)
        plot_dir = tmp_path / "plots"
        generate_qc_plots(records, plot_dir)
        assert (plot_dir / "qc_length_percentiles.png").exists()

    def test_output_inside_raw_raises(self, tmp_path):
        raw     = tmp_path / "raw"
        raw.mkdir()
        records = [_make_success_record()]
        with pytest.raises(ValueError, match="SAFETY"):
            generate_qc_plots(records, raw / "plots", raw_root=raw)


# ---------------------------------------------------------------------------
# TestSummaryCSVs
# ---------------------------------------------------------------------------

class TestSummaryCSVs:

    def _make_mixed_records(self) -> list[StreamlineQCRecord]:
        recs = [
            _make_success_record("bni", "s001", 3200, 85.0),
            _make_success_record("bni", "s002", 2800, 78.0),
            _make_success_record("nyu1", "s003", 3900, 36.0),
        ]
        failed = StreamlineQCRecord(
            site="bni", dataset="d", subject_id="s_bad",
            streamline_count=0,
            mean_length_mm=0.0, median_length_mm=0.0, std_length_mm=0.0,
            p10_length_mm=0.0, p25_length_mm=0.0, p75_length_mm=0.0,
            p90_length_mm=0.0, min_length_mm=0.0, max_length_mm=0.0,
            iqr_length_mm=0.0, status="failed",
        )
        recs.append(failed)
        return compute_site_normalizations(recs)

    def test_subject_csv_written(self, tmp_path):
        import pandas as pd
        records = self._make_mixed_records()
        subj_csv, _, _ = save_summary_csvs(records, tmp_path)
        assert subj_csv.exists()
        df = pd.read_csv(subj_csv)
        assert len(df) == 4

    def test_subject_csv_has_zscore_columns(self, tmp_path):
        import pandas as pd
        records = self._make_mixed_records()
        subj_csv, _, _ = save_summary_csvs(records, tmp_path)
        df = pd.read_csv(subj_csv)
        for col in ("z_streamline_count", "z_mean_length_mm", "z_p90_length_mm"):
            assert col in df.columns

    def test_site_csv_written(self, tmp_path):
        import pandas as pd
        records = self._make_mixed_records()
        _, site_csv, _ = save_summary_csvs(records, tmp_path)
        assert site_csv.exists()
        df = pd.read_csv(site_csv)
        assert set(df["site"]) == {"bni", "nyu1"}
        expected = {
            "site", "n_subjects", "n_outliers",
            "mean_streamline_count", "std_streamline_count",
            "mean_mean_length_mm",   "std_mean_length_mm",
            "mean_p10_length_mm",    "mean_median_length_mm",
            "mean_p90_length_mm",    "mean_iqr_length_mm",
            "mean_std_length_mm",
        }
        assert expected.issubset(set(df.columns))

    def test_outlier_csv_written(self, tmp_path):
        import pandas as pd
        # Inject an outlier
        recs  = self._make_mixed_records()
        recs[0].qc_outlier = True
        recs[0].qc_reason  = "z_count=4.50"
        _, _, outlier_csv = save_summary_csvs(recs, tmp_path)
        assert outlier_csv.exists()
        df = pd.read_csv(outlier_csv)
        assert len(df) == 1
        assert df["subject_id"].iloc[0] == "s001"

    def test_save_inside_raw_raises(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        with pytest.raises(ValueError, match="SAFETY"):
            save_summary_csvs([_make_success_record()], raw, raw_root=raw)
