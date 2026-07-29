"""
Phase 3.4 — Connectome Builder

Converts registered atlas + subject streamlines into weighted adjacency
matrices, graph objects, and baseline comparison artifacts.

Per-subject outputs (connectome/):
  adjacency_streamline_count.npy
  adjacency_mean_length.npy
  adjacency_mean_fa.npy
  adjacency_mean_md.npy
  adjacency_mean_ad.npy
  adjacency_mean_rd.npy
  edge_table.csv
  node_table.csv
  graph_metrics.json
  connectome_report.json
  baseline_artifacts/
    streamline_endpoint_table.csv
    streamline_edge_assignment.csv
    edge_aggregation_report.json
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import networkx as nx
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

from neurofiber.connectome.atlas_registration import RegisteredAtlas
from neurofiber.connectome.graph_metrics import compute_graph_metrics
from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

# Known Phase 3.3 outlier subjects (preserved, flagged for review)
_KNOWN_OUTLIERS: set[str] = {"29012", "29030", "29055", "28869", "28909", "29102"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConnectomeRecord:
    site:        str
    dataset:     str
    subject_id:  str

    # Graph topology
    node_count:    int
    edge_count:    int
    graph_density: float
    mean_degree:   float

    # Edge statistics
    mean_edge_fa:     Optional[float]
    mean_edge_md:     Optional[float]
    mean_edge_length: Optional[float]

    # Streamline mapping
    streamline_count_total:   int
    mapped_streamline_count:  int
    unmapped_streamline_count: int
    self_loop_count:          int
    mapping_success_ratio:    float

    # QC
    review_required:  bool            = False
    warning_message:  Optional[str]   = None
    status:           str             = "success"
    error_message:    Optional[str]   = None
    timestamp:        str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_summary_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scalar sampling helpers
# ---------------------------------------------------------------------------

def _sample_scalars_along_streamlines(
    streamlines,
    scalar_data: np.ndarray,
    scalar_affine: np.ndarray,
) -> np.ndarray:
    """
    Sample a scalar volume at every point of every streamline and compute
    the mean per streamline (excluding NaN / background=0).

    Returns array of shape (n_streamlines,) with mean scalar per streamline.
    NaN is returned for streamlines where no valid (>0) voxel was found.
    """
    inv_affine = np.linalg.inv(scalar_affine)
    shape = np.array(scalar_data.shape, dtype=float)
    per_sl = np.full(len(streamlines), np.nan, dtype=np.float64)

    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            continue
        vox = apply_affine(inv_affine, pts)
        # Clamp to valid voxel range
        for d in range(3):
            vox[:, d] = np.clip(vox[:, d], 0.0, shape[d] - 1.0)
        vals = map_coordinates(scalar_data, vox.T, order=1, mode="constant", cval=0.0)
        valid = vals > 0
        if valid.any():
            per_sl[i] = float(np.mean(vals[valid]))

    return per_sl


def _compute_streamline_lengths(streamlines) -> np.ndarray:
    """Compute arc-length (mm) for each streamline."""
    lengths = np.zeros(len(streamlines), dtype=np.float64)
    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl)
        if len(pts) >= 2:
            diffs = np.diff(pts, axis=0)
            lengths[i] = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    return lengths


# ---------------------------------------------------------------------------
# Adjacency matrix construction
# ---------------------------------------------------------------------------

def _build_adjacency_matrices(
    n_rois:         int,
    count_matrix:   np.ndarray,
    mapping:        dict,
    streamlines,
    lengths:        np.ndarray,
    fa_per_sl:      np.ndarray,
    md_per_sl:      np.ndarray,
    ad_per_sl:      np.ndarray,
    rd_per_sl:      np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build NxN weighted adjacency matrices from DIPY connectivity_matrix output.
    count_matrix is (max_label+1) x (max_label+1) from DIPY (0 = background).
    Adjacency matrices are (n_rois x n_rois), indexed 0 … n_rois-1.
    ROI label l → matrix index l-1.
    """
    N = n_rois
    adj_count   = np.zeros((N, N), dtype=np.float64)
    adj_length  = np.full((N, N), np.nan)
    adj_fa      = np.full((N, N), np.nan)
    adj_md      = np.full((N, N), np.nan)
    adj_ad      = np.full((N, N), np.nan)
    adj_rd      = np.full((N, N), np.nan)

    # Fill count from DIPY matrix (labels 1..N → indices 0..N-1)
    size = min(count_matrix.shape[0], N + 1)
    adj_count[:size - 1, :size - 1] = count_matrix[1:size, 1:size].astype(float)

    # Edge-level statistics from per-streamline values
    for (i, j), sl_indices in mapping.items():
        if i < 1 or j < 1 or i > N or j > N:
            continue
        ii, jj = i - 1, j - 1
        if ii == jj:
            continue  # skip self-loops in edge stats

        idxs = np.array(sl_indices, dtype=int)
        l_vals  = lengths[idxs]
        fa_vals = fa_per_sl[idxs]
        md_vals = md_per_sl[idxs]
        ad_vals = ad_per_sl[idxs]
        rd_vals = rd_per_sl[idxs]

        def _mean_valid(arr):
            v = arr[~np.isnan(arr)]
            return float(np.mean(v)) if len(v) > 0 else np.nan

        adj_length[ii, jj] = adj_length[jj, ii] = _mean_valid(l_vals)
        adj_fa[ii, jj]     = adj_fa[jj, ii]     = _mean_valid(fa_vals)
        adj_md[ii, jj]     = adj_md[jj, ii]     = _mean_valid(md_vals)
        adj_ad[ii, jj]     = adj_ad[jj, ii]     = _mean_valid(ad_vals)
        adj_rd[ii, jj]     = adj_rd[jj, ii]     = _mean_valid(rd_vals)

    return {
        "count":  adj_count,
        "length": adj_length,
        "fa":     adj_fa,
        "md":     adj_md,
        "ad":     adj_ad,
        "rd":     adj_rd,
    }


# ---------------------------------------------------------------------------
# Baseline artifact export
# ---------------------------------------------------------------------------

def _save_baseline_artifacts(
    output_dir:     Path,
    streamlines,
    lengths:        np.ndarray,
    fa_per_sl:      np.ndarray,
    md_per_sl:      np.ndarray,
    atlas_fa_data:  np.ndarray,
    atlas_affine:   np.ndarray,
    n_rois:         int,
) -> None:
    """
    Save three CSVs for future fiber-centric comparison:
      - streamline_endpoint_table.csv  : per-streamline endpoints + basic metrics
      - streamline_edge_assignment.csv : per-streamline ROI assignment
      - edge_aggregation_report.json   : what was discarded / aggregated
    """
    art_dir = output_dir / "baseline_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    n = len(streamlines)
    inv_affine = np.linalg.inv(atlas_affine)

    # Per-streamline endpoint voxel coords → atlas label
    start_labels = np.zeros(n, dtype=int)
    end_labels   = np.zeros(n, dtype=int)
    shape = np.array(atlas_fa_data.shape, dtype=float)

    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl)
        if len(pts) < 2:
            continue
        for k, pt in enumerate([pts[0], pts[-1]]):
            vox = apply_affine(inv_affine, pt)
            vox = np.round(vox).astype(int)
            if all(0 <= vox[d] < atlas_fa_data.shape[d] for d in range(3)):
                label = int(atlas_fa_data[vox[0], vox[1], vox[2]])
            else:
                label = 0
            if k == 0:
                start_labels[i] = label
            else:
                end_labels[i] = label

    # Endpoint table
    ep_df = pd.DataFrame({
        "streamline_id":   np.arange(n),
        "start_roi":       start_labels,
        "end_roi":         end_labels,
        "length_mm":       lengths,
        "mean_fa":         fa_per_sl,
        "mean_md":         md_per_sl,
    })
    ep_df.to_csv(art_dir / "streamline_endpoint_table.csv", index=False)

    # Edge assignment
    is_mapped    = (start_labels > 0) & (end_labels > 0) & (start_labels != end_labels)
    is_selfloop  = (start_labels > 0) & (end_labels > 0) & (start_labels == end_labels)
    is_unmapped  = ~is_mapped & ~is_selfloop

    assign_df = pd.DataFrame({
        "streamline_id":   np.arange(n),
        "start_roi":       start_labels,
        "end_roi":         end_labels,
        "assigned_edge":   np.where(is_mapped, True, False),
        "is_self_loop":    is_selfloop,
        "is_unmapped":     is_unmapped,
        "length_mm":       lengths,
        "mean_fa":         fa_per_sl,
    })
    assign_df.to_csv(art_dir / "streamline_edge_assignment.csv", index=False)

    # Summary report
    report = {
        "total_streamlines":    n,
        "mapped_streamlines":   int(is_mapped.sum()),
        "unmapped_streamlines": int(is_unmapped.sum()),
        "self_loop_streamlines": int(is_selfloop.sum()),
        "mapping_success_ratio": round(float(is_mapped.sum()) / max(n, 1), 4),
        "note": (
            "streamline_endpoint_table.csv and streamline_edge_assignment.csv "
            "preserve per-streamline geometry metadata discarded by the atlas "
            "ROI aggregation step — intended for Phase 3.5 fiber-centric comparison."
        ),
    }
    (art_dir / "edge_aggregation_report.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main per-subject builder
# ---------------------------------------------------------------------------

def build_subject_connectome(
    dti_dir:          Path,
    registered_atlas: RegisteredAtlas,
    output_dir:       Path,
    raw_root:         Optional[Path] = None,
    skip_if_exists:   bool = False,
    trk_path:         Optional[Path] = None,
) -> ConnectomeRecord:
    """
    Build atlas-based connectome for one subject.

    Loads streamlines.trk + FA/MD/AD/RD maps, maps streamline endpoints
    to atlas ROIs, constructs weighted adjacency matrices, computes graph
    metrics, and saves all outputs.
    """
    if raw_root is not None:
        guard_no_raw_write(output_dir, raw_root)

    subject_id = dti_dir.parent.parent.name
    dataset    = dti_dir.parent.parent.parent.name
    site       = dti_dir.parent.parent.parent.parent.name

    def _failed(msg: str) -> ConnectomeRecord:
        return ConnectomeRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            node_count=0, edge_count=0, graph_density=0.0, mean_degree=0.0,
            mean_edge_fa=None, mean_edge_md=None, mean_edge_length=None,
            streamline_count_total=0, mapped_streamline_count=0,
            unmapped_streamline_count=0, self_loop_count=0,
            mapping_success_ratio=0.0,
            status="failed", error_message=msg,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "connectome_report.json"
    if skip_if_exists and report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
            if existing.get("status") == "success":
                logger.info("[%s/%s] skipping — already processed", site, subject_id)
                return _load_record_from_report(existing, site, dataset, subject_id)
        except Exception:
            pass

    # --- Load inputs --------------------------------------------------------
    if trk_path is None:
        trk_path = dti_dir / "tractography" / "streamlines.trk"
    fa_path   = dti_dir / "tensor" / "FA.nii.gz"
    md_path   = dti_dir / "tensor" / "MD.nii.gz"
    ad_path   = dti_dir / "tensor" / "AD.nii.gz"
    rd_path   = dti_dir / "tensor" / "RD.nii.gz"

    for p in [trk_path, fa_path, md_path, ad_path, rd_path]:
        if not p.exists():
            return _failed(f"Missing input: {p.name}")

    try:
        from dipy.io.streamline import load_trk
        from dipy.tracking.utils import connectivity_matrix

        sft        = load_trk(str(trk_path), "same", bbox_valid_check=False)
        streamlines = list(sft.streamlines)
        n_total     = len(streamlines)

        if n_total == 0:
            return _failed("No streamlines in .trk file")

        fa_img = nib.load(str(fa_path))
        md_img = nib.load(str(md_path))
        ad_img = nib.load(str(ad_path))
        rd_img = nib.load(str(rd_path))

        fa_data = np.asarray(fa_img.dataobj, dtype=np.float32)
        md_data = np.asarray(md_img.dataobj, dtype=np.float32)
        ad_data = np.asarray(ad_img.dataobj, dtype=np.float32)
        rd_data = np.asarray(rd_img.dataobj, dtype=np.float32)

        # Load registered atlas (already in FA space)
        atlas_img  = nib.load(str(registered_atlas.atlas_subject_space_path))
        atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)
        atlas_affine = atlas_img.affine   # == fa_img.affine

        # --- Pre-compute per-streamline scalars ---------------------------
        lengths    = _compute_streamline_lengths(streamlines)
        fa_per_sl  = _sample_scalars_along_streamlines(streamlines, fa_data, fa_img.affine)
        md_per_sl  = _sample_scalars_along_streamlines(streamlines, md_data, md_img.affine)
        ad_per_sl  = _sample_scalars_along_streamlines(streamlines, ad_data, ad_img.affine)
        rd_per_sl  = _sample_scalars_along_streamlines(streamlines, rd_data, rd_img.affine)

        # --- Streamline → ROI mapping via DIPY ----------------------------
        count_matrix, mapping = connectivity_matrix(
            streamlines,
            fa_img.affine,          # atlas is in FA space → use FA affine
            atlas_data.astype(np.int32),
            symmetric=True,
            return_mapping=True,
            mapping_as_streamlines=False,
        )

        # count_matrix is symmetric (DIPY adds to both [i,j] and [j,i]).
        # Diagonal entries are also doubled. Use upper triangle to count correctly.
        roi_mat         = count_matrix[1:, 1:]
        self_loop_count = int(np.diag(roi_mat).sum()) // 2   # each self-loop counted twice
        mapped_count    = int(np.triu(roi_mat, k=1).sum())   # off-diagonal unique connections
        unmapped_count  = n_total - mapped_count - self_loop_count

        # --- Build adjacency matrices ------------------------------------
        n_rois = registered_atlas.n_rois
        adjs   = _build_adjacency_matrices(
            n_rois, count_matrix, mapping,
            streamlines, lengths,
            fa_per_sl, md_per_sl, ad_per_sl, rd_per_sl,
        )

        # --- Graph metrics ------------------------------------------------
        graph_met = compute_graph_metrics(adjs["count"], registered_atlas.label_df)

        # --- QC flags -------------------------------------------------------
        # Thresholds calibrated to WM-seeded EuDX tractography on ABIDE-II:
        # typical density 4-6%, typical mapping ratio 8-13%.
        # Flag only true failures (registration collapse, empty atlas overlap).
        warnings = []
        if graph_met["edge_count"] == 0:
            warnings.append("zero edges — atlas registration may have failed")
        if graph_met["graph_density"] < 0.01:
            warnings.append(f"critically low density ({graph_met['graph_density']:.4f})")
        ratio = mapped_count / max(n_total, 1)
        if ratio < 0.03:
            warnings.append(f"critically low mapping ratio ({ratio:.3f})")

        # --- Save outputs ---------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(output_dir / "adjacency_streamline_count.npy"), adjs["count"])
        np.save(str(output_dir / "adjacency_mean_length.npy"),      adjs["length"])
        np.save(str(output_dir / "adjacency_mean_fa.npy"),          adjs["fa"])
        np.save(str(output_dir / "adjacency_mean_md.npy"),          adjs["md"])
        np.save(str(output_dir / "adjacency_mean_ad.npy"),          adjs["ad"])
        np.save(str(output_dir / "adjacency_mean_rd.npy"),          adjs["rd"])

        # Node table
        node_df = registered_atlas.label_df.copy()
        node_df["degree"] = (adjs["count"] > 0).sum(axis=1)
        node_df["weighted_degree"] = adjs["count"].sum(axis=1)
        node_df.to_csv(output_dir / "node_table.csv", index=False)

        # Edge table
        edge_rows = []
        for (i, j), sl_indices in mapping.items():
            if i < 1 or j < 1 or i > n_rois or j > n_rois or i == j:
                continue
            idxs = np.array(sl_indices, dtype=int)
            edge_rows.append({
                "roi_i":       i,
                "roi_j":       j,
                "roi_i_name":  _get_label(registered_atlas.label_df, i),
                "roi_j_name":  _get_label(registered_atlas.label_df, j),
                "streamline_count": len(idxs),
                "mean_length_mm":   float(np.nanmean(lengths[idxs])),
                "mean_fa":          float(np.nanmean(fa_per_sl[idxs])),
                "mean_md":          float(np.nanmean(md_per_sl[idxs])),
                "mean_ad":          float(np.nanmean(ad_per_sl[idxs])),
                "mean_rd":          float(np.nanmean(rd_per_sl[idxs])),
            })
        pd.DataFrame(edge_rows).to_csv(output_dir / "edge_table.csv", index=False)

        # Save NetworkX graph
        G = _build_networkx_graph(adjs, registered_atlas.label_df, n_rois)
        nx.write_graphml(G, str(output_dir / "graph.graphml"))

        # Graph metrics JSON
        (output_dir / "graph_metrics.json").write_text(json.dumps(graph_met, indent=2))

        # Baseline artifacts
        _save_baseline_artifacts(
            output_dir, streamlines, lengths, fa_per_sl, md_per_sl,
            atlas_data, atlas_affine, n_rois,
        )

        # Connectome report
        valid_fa  = fa_per_sl[~np.isnan(fa_per_sl)]
        valid_md  = md_per_sl[~np.isnan(md_per_sl)]
        mean_edge_fa  = float(np.nanmean(adjs["fa"][~np.isnan(adjs["fa"])])) if np.any(~np.isnan(adjs["fa"])) else None
        mean_edge_md  = float(np.nanmean(adjs["md"][~np.isnan(adjs["md"])])) if np.any(~np.isnan(adjs["md"])) else None
        mean_edge_len = float(np.nanmean(adjs["length"][~np.isnan(adjs["length"])])) if np.any(~np.isnan(adjs["length"])) else None

        review = subject_id in _KNOWN_OUTLIERS or bool(warnings)
        warning_msg = "; ".join(warnings) if warnings else None

        report_data = {
            "site": site, "dataset": dataset, "subject_id": subject_id,
            "node_count":   graph_met["node_count"],
            "edge_count":   graph_met["edge_count"],
            "graph_density": graph_met["graph_density"],
            "mean_degree":  graph_met.get("mean_degree", 0.0),
            "mean_edge_fa":     mean_edge_fa,
            "mean_edge_md":     mean_edge_md,
            "mean_edge_length": mean_edge_len,
            "streamline_count_total":    n_total,
            "mapped_streamline_count":   mapped_count,
            "unmapped_streamline_count": unmapped_count,
            "self_loop_count":           self_loop_count,
            "mapping_success_ratio":     round(ratio, 4),
            "review_required":  review,
            "warning_message":  warning_msg,
            "status":           "success",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_text(json.dumps(report_data, indent=2))

        logger.info(
            "[%s/%s] nodes=%d  edges=%d  density=%.3f  mapped=%d/%d  FA=%.3f",
            site, subject_id,
            graph_met["node_count"], graph_met["edge_count"],
            graph_met["graph_density"],
            mapped_count, n_total,
            mean_edge_fa if mean_edge_fa else 0,
        )

        return ConnectomeRecord(
            site=site, dataset=dataset, subject_id=subject_id,
            node_count=graph_met["node_count"],
            edge_count=graph_met["edge_count"],
            graph_density=graph_met["graph_density"],
            mean_degree=graph_met.get("mean_degree", 0.0),
            mean_edge_fa=mean_edge_fa,
            mean_edge_md=mean_edge_md,
            mean_edge_length=mean_edge_len,
            streamline_count_total=n_total,
            mapped_streamline_count=mapped_count,
            unmapped_streamline_count=unmapped_count,
            self_loop_count=self_loop_count,
            mapping_success_ratio=round(ratio, 4),
            review_required=review,
            warning_message=warning_msg,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s/%s] FAILED: %s", site, subject_id, exc)
        return _failed(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_label(label_df: pd.DataFrame, idx: int) -> str:
    row = label_df[label_df["index"] == idx]
    if not row.empty:
        return str(row["name"].iloc[0])
    return f"roi_{idx:03d}"


def _build_networkx_graph(
    adjs:     dict[str, np.ndarray],
    label_df: pd.DataFrame,
    n_rois:   int,
) -> nx.Graph:
    G = nx.Graph()
    for i in range(n_rois):
        label = i + 1
        name  = _get_label(label_df, label)
        G.add_node(i, roi_index=label, name=name)
    count = adjs["count"]
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            c = int(count[i, j])
            if c > 0:
                G.add_edge(
                    i, j,
                    streamline_count=c,
                    mean_length=float(adjs["length"][i, j]) if not np.isnan(adjs["length"][i, j]) else 0.0,
                    mean_fa=float(adjs["fa"][i, j])   if not np.isnan(adjs["fa"][i, j])   else 0.0,
                    mean_md=float(adjs["md"][i, j])   if not np.isnan(adjs["md"][i, j])   else 0.0,
                )
    return G


def _load_record_from_report(
    r: dict,
    site: str,
    dataset: str,
    subject_id: str,
) -> ConnectomeRecord:
    return ConnectomeRecord(
        site=site, dataset=dataset, subject_id=subject_id,
        node_count=r.get("node_count", 0),
        edge_count=r.get("edge_count", 0),
        graph_density=r.get("graph_density", 0.0),
        mean_degree=r.get("mean_degree", 0.0),
        mean_edge_fa=r.get("mean_edge_fa"),
        mean_edge_md=r.get("mean_edge_md"),
        mean_edge_length=r.get("mean_edge_length"),
        streamline_count_total=r.get("streamline_count_total", 0),
        mapped_streamline_count=r.get("mapped_streamline_count", 0),
        unmapped_streamline_count=r.get("unmapped_streamline_count", 0),
        self_loop_count=r.get("self_loop_count", 0),
        mapping_success_ratio=r.get("mapping_success_ratio", 0.0),
        review_required=r.get("review_required", False),
        warning_message=r.get("warning_message"),
        status=r.get("status", "success"),
    )
