"""
NeuroFiber Phase 3R.4 — Connectome QC and Harmonization Preparation

Computes graph-level QC metrics, vectorizes edge features, quantifies site effects,
and creates harmonization-ready feature tables from Phase 3R.3 connectivity matrices.

This phase does NOT remove site effects. It analyzes and prepares.

Input per subject:
  data/processed_v2b/abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1/connectome/
    count_matrix.npy, mean_length_matrix.npy, mean_fa_matrix.npy,
    mean_md_matrix.npy, mean_ad_matrix.npy, mean_rd_matrix.npy

Also uses:
  data/processed_v2b/phase3r_3_connectome_summary.csv  (density, streamline stats)
  data/processed_v2b/phase2r_1_standard_dwi_preprocessing_summary.csv (bvals, dirs)
  data/processed_v2b/phase2r_3_brain_mask_tensor_qc_summary.csv (FA/MD QC)

Outputs:
  data/processed_v2b/connectome_features/
    count_edges.csv
    length_edges.csv
    fa_edges.csv
    md_edges.csv
    subject_graph_metrics.csv
    harmonization_metadata.csv
  data/processed_v2b/connectome_features/experimental_site_zscore/
    count_edges_zscore.csv   (experimental — not for direct modeling)
    ...

  data/processed_v2b/
    phase3r_4_connectome_qc_summary.csv
    phase3r_4_site_effect_summary.csv
    phase3r_4_statistical_tests.csv

Safety contract:
  - Reads matrices read-only from data/processed_v2b/
  - guard_no_raw_write() enforced at every write entry point
  - Never overwrites count_matrix.npy or other raw matrices
  - IP_1 excluded
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

PIPELINE_VERSION = "3R.4"
ATLAS_N_ROIS     = 100
N_EDGES          = ATLAS_N_ROIS * (ATLAS_N_ROIS - 1) // 2   # 4950 upper-triangle edges

CLEAN_SITES:   list[str] = ["BNI", "NYU_1", "NYU_2", "SDSU_1", "TCD_1"]
EXCLUDED_SITES: list[str] = ["IP_1"]

_SITE_FOLDER_MAP: dict[str, str] = {
    "BNI":    "bni",
    "IP_1":   "ip",
    "NYU_1":  "nyu1",
    "NYU_2":  "nyu2",
    "SDSU_1": "sdsu",
    "TCD_1":  "tcd",
}

EXPECTED_COUNTS: dict[str, int] = {
    "BNI":    58,
    "NYU_1":  55,
    "NYU_2":  19,
    "SDSU_1": 57,
    "TCD_1":  40,
}

# QC thresholds
_DENSITY_LOW   = 0.01
_DENSITY_HIGH  = 0.10
_MIN_EDGES     = 10
_OUTLIER_SIGMA = 2.5


def _r(v: Optional[float], decimals: int = 4) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), decimals)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GraphMetrics:
    subject_id:             str
    site:                   str
    dataset:                str
    nonzero_edges:          int   = 0
    density:                float = 0.0
    total_streamline_weight: float = 0.0
    mean_edge_weight:       float = 0.0
    median_edge_weight:     float = 0.0
    mean_edge_length:       Optional[float] = None
    global_fa_edge_mean:    Optional[float] = None
    global_md_edge_mean:    Optional[float] = None
    degree_mean:            float = 0.0
    degree_std:             float = 0.0
    weighted_degree_mean:   float = 0.0
    connected_components:   int   = 1
    isolated_nodes:         int   = 0
    qc_flag:                str   = ""
    site_zscore_density:    Optional[float] = None
    status:                 str   = "success"
    warning_message:        Optional[str]  = None

    def to_row(self) -> dict:
        return {
            "subject_id":              self.subject_id,
            "site":                    self.site,
            "dataset":                 self.dataset,
            "nonzero_edges":           self.nonzero_edges,
            "density":                 _r(self.density, 6),
            "total_streamline_weight": _r(self.total_streamline_weight, 2),
            "mean_edge_weight":        _r(self.mean_edge_weight, 4),
            "median_edge_weight":      _r(self.median_edge_weight, 4),
            "mean_edge_length":        _r(self.mean_edge_length, 2) if self.mean_edge_length is not None else None,
            "global_fa_edge_mean":     _r(self.global_fa_edge_mean, 6) if self.global_fa_edge_mean is not None else None,
            "global_md_edge_mean":     _r(self.global_md_edge_mean, 8) if self.global_md_edge_mean is not None else None,
            "degree_mean":             _r(self.degree_mean, 4),
            "degree_std":              _r(self.degree_std, 4),
            "weighted_degree_mean":    _r(self.weighted_degree_mean, 4),
            "connected_components":    self.connected_components,
            "isolated_nodes":          self.isolated_nodes,
            "qc_flag":                 self.qc_flag,
            "site_zscore_density":     _r(self.site_zscore_density, 4) if self.site_zscore_density is not None else None,
            "status":                  self.status,
            "warning_message":         self.warning_message,
        }


GRAPH_METRICS_FIELDS = list(GraphMetrics("", "", "").to_row().keys())

HARMONIZATION_META_FIELDS = [
    "subject_id", "site", "dataset",
    "age", "sex", "diagnosis",
    "scanner_info",
    "b_value", "direction_count",
    "mean_fa", "mean_md",
    "density", "mean_streamline_length",
]

SITE_EFFECT_FIELDS = [
    "site", "metric", "mean", "std", "cv",
    "grand_mean", "grand_std", "site_z",
    "outlier_flag",
]

STAT_TEST_FIELDS = [
    "metric", "test", "statistic", "p_value",
    "interpretation",
]


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------

def load_matrices(connectome_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Load all six matrices. Returns None if count_matrix.npy is missing."""
    count_path = connectome_dir / "count_matrix.npy"
    if not count_path.exists():
        return None
    try:
        return {
            "count":  np.load(str(connectome_dir / "count_matrix.npy")),
            "length": np.load(str(connectome_dir / "mean_length_matrix.npy")) if (connectome_dir / "mean_length_matrix.npy").exists() else None,
            "fa":     np.load(str(connectome_dir / "mean_fa_matrix.npy"))     if (connectome_dir / "mean_fa_matrix.npy").exists() else None,
            "md":     np.load(str(connectome_dir / "mean_md_matrix.npy"))     if (connectome_dir / "mean_md_matrix.npy").exists() else None,
            "ad":     np.load(str(connectome_dir / "mean_ad_matrix.npy"))     if (connectome_dir / "mean_ad_matrix.npy").exists() else None,
            "rd":     np.load(str(connectome_dir / "mean_rd_matrix.npy"))     if (connectome_dir / "mean_rd_matrix.npy").exists() else None,
        }
    except Exception as exc:
        logger.error("Failed to load matrices from %s: %s", connectome_dir, exc)
        return None


# ---------------------------------------------------------------------------
# Graph metric computation
# ---------------------------------------------------------------------------

def compute_graph_metrics(
    subject_id: str,
    site:       str,
    dataset:    str,
    matrices:   dict[str, np.ndarray],
) -> GraphMetrics:
    """Compute graph-level QC metrics from connectivity matrices."""
    m = GraphMetrics(subject_id=subject_id, site=site, dataset=dataset)
    count = matrices["count"]
    n = count.shape[0]

    if count is None or count.shape[0] == 0:
        m.status = "failed"
        m.warning_message = "count_matrix is empty"
        return m

    upper = np.triu(count, k=1)
    nonzero_mask = upper > 0
    m.nonzero_edges = int(nonzero_mask.sum())

    possible = n * (n - 1) / 2
    m.density = _r(m.nonzero_edges / possible, 6) if possible > 0 else 0.0

    edge_weights = upper[nonzero_mask]
    if len(edge_weights) > 0:
        m.total_streamline_weight = _r(float(edge_weights.sum()), 2)
        m.mean_edge_weight        = _r(float(edge_weights.mean()), 4)
        m.median_edge_weight      = _r(float(np.median(edge_weights)), 4)

    # Length / FA / MD averages over nonzero edges
    length_mat = matrices.get("length")
    if length_mat is not None:
        length_upper = np.triu(length_mat, k=1)
        valid_len = length_upper[nonzero_mask & ~np.isnan(length_upper)]
        if len(valid_len) > 0:
            m.mean_edge_length = _r(float(valid_len.mean()), 2)

    fa_mat = matrices.get("fa")
    if fa_mat is not None:
        fa_upper = np.triu(fa_mat, k=1)
        valid_fa = fa_upper[nonzero_mask & ~np.isnan(fa_upper)]
        if len(valid_fa) > 0:
            m.global_fa_edge_mean = _r(float(valid_fa.mean()), 6)

    md_mat = matrices.get("md")
    if md_mat is not None:
        md_upper = np.triu(md_mat, k=1)
        valid_md = md_upper[nonzero_mask & ~np.isnan(md_upper)]
        if len(valid_md) > 0:
            m.global_md_edge_mean = _r(float(valid_md.mean()), 8)

    # Degree
    binary = (count > 0).astype(float)
    np.fill_diagonal(binary, 0)
    degrees = binary.sum(axis=1)
    m.degree_mean = _r(float(degrees.mean()), 4)
    m.degree_std  = _r(float(degrees.std()), 4)

    # Weighted degree (strength)
    count_sym = count.copy(); np.fill_diagonal(count_sym, 0)
    strengths = count_sym.sum(axis=1)
    m.weighted_degree_mean = _r(float(strengths.mean()), 4)

    # Connected components and isolated nodes (simple BFS)
    m.connected_components, m.isolated_nodes = _count_components(binary)

    # QC flags
    flags: list[str] = []
    if m.density < _DENSITY_LOW:
        flags.append(f"density<{_DENSITY_LOW}")
    if m.density > _DENSITY_HIGH:
        flags.append(f"density>{_DENSITY_HIGH}")
    if m.nonzero_edges < _MIN_EDGES:
        flags.append(f"edges<{_MIN_EDGES}")
    if m.isolated_nodes > n // 4:
        flags.append(f"isolated_nodes={m.isolated_nodes}")

    m.qc_flag = "; ".join(flags)
    m.status   = "success"
    return m


def _count_components(binary: np.ndarray) -> tuple[int, int]:
    """BFS component counting. Returns (n_components, n_isolated_nodes)."""
    n = binary.shape[0]
    visited = np.zeros(n, dtype=bool)
    n_comp  = 0
    n_isolated = 0
    for start in range(n):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        component_size = 0
        while queue:
            node = queue.pop(0)
            component_size += 1
            neighbors = np.where(binary[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        n_comp += 1
        if component_size == 1:
            n_isolated += 1
    return n_comp, n_isolated


# ---------------------------------------------------------------------------
# Edge vectorization
# ---------------------------------------------------------------------------

def extract_upper_triangle(matrix: np.ndarray, n: int = ATLAS_N_ROIS) -> np.ndarray:
    """Return flattened upper triangle (k=1) as 1D array of length N*(N-1)/2."""
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def build_edge_table(
    subjects: list[dict],
    matrix_key: str,
    n: int = ATLAS_N_ROIS,
) -> tuple[list[str], list[list]]:
    """
    Build edge feature table for one matrix type.
    subjects: list of {'subject_id', 'site', 'dataset', 'matrices', 'meta'}
    Returns (header, rows) where each row has meta columns + N_EDGES edge values.
    """
    header = ["subject_id", "site", "dataset",
              "diagnosis", "age", "sex"] + [f"edge_{i+1:04d}" for i in range(n*(n-1)//2)]
    rows = []
    for s in subjects:
        mat = s["matrices"].get(matrix_key)
        if mat is None:
            continue
        meta = s.get("meta", {})
        edge_vals = extract_upper_triangle(mat, n)
        # Replace NaN with empty string for CSV
        edge_str = [str(v) if not np.isnan(v) else "" for v in edge_vals]
        row = [
            s["subject_id"], s["site"], s["dataset"],
            meta.get("diagnosis", ""),
            meta.get("age", ""),
            meta.get("sex", ""),
        ] + edge_str
        rows.append(row)
    return header, rows


# ---------------------------------------------------------------------------
# Site-effect analysis
# ---------------------------------------------------------------------------

def compute_site_effects(
    metrics: list[GraphMetrics],
    metric_names: list[str],
) -> list[dict]:
    """
    For each metric, compute per-site mean/std/CV and site z-score vs grand mean.
    Returns list of rows for phase3r_4_site_effect_summary.csv.
    """
    rows: list[dict] = []
    all_vals: dict[str, list[float]] = {m: [] for m in metric_names}

    # Gather all valid values
    for rec in metrics:
        for mn in metric_names:
            v = getattr(rec, mn, None)
            if v is not None and not np.isnan(float(v)):
                all_vals[mn].append(float(v))

    for mn in metric_names:
        grand_vals = all_vals[mn]
        if not grand_vals:
            continue
        grand_mean = float(np.mean(grand_vals))
        grand_std  = float(np.std(grand_vals))

        for site in CLEAN_SITES:
            site_recs = [r for r in metrics if r.site == site and r.status == "success"]
            site_vals = [float(getattr(r, mn)) for r in site_recs
                         if getattr(r, mn) is not None and not np.isnan(float(getattr(r, mn)))]
            if not site_vals:
                continue
            site_mean = float(np.mean(site_vals))
            site_std  = float(np.std(site_vals))
            cv        = site_std / site_mean if site_mean != 0 else 0.0
            site_z    = (site_mean - grand_mean) / grand_std if grand_std > 0 else 0.0
            outlier   = abs(site_z) > _OUTLIER_SIGMA

            rows.append({
                "site":        site,
                "metric":      mn,
                "mean":        _r(site_mean, 6),
                "std":         _r(site_std, 6),
                "cv":          _r(cv, 4),
                "grand_mean":  _r(grand_mean, 6),
                "grand_std":   _r(grand_std, 6),
                "site_z":      _r(site_z, 4),
                "outlier_flag": "OUTLIER" if outlier else "",
            })

    return rows


def compute_site_zscores_for_metrics(
    metrics: list[GraphMetrics],
) -> None:
    """Fill in site_zscore_density for each subject in-place."""
    for site in CLEAN_SITES:
        site_recs = [r for r in metrics if r.site == site and r.status == "success"]
        vals = [r.density for r in site_recs]
        if len(vals) < 2:
            continue
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma == 0:
            continue
        for r in site_recs:
            r.site_zscore_density = _r((r.density - mu) / sigma, 4)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(
    metrics:     list[GraphMetrics],
    metric_names: list[str],
) -> list[dict]:
    """
    One-way Kruskal-Wallis test across sites for each metric.
    Returns list of test rows.
    """
    from scipy.stats import kruskal, f_oneway, shapiro

    rows: list[dict] = []
    for mn in metric_names:
        groups = []
        for site in CLEAN_SITES:
            vals = [float(getattr(r, mn)) for r in metrics
                    if r.site == site and r.status == "success"
                    and getattr(r, mn) is not None
                    and not np.isnan(float(getattr(r, mn)))]
            if vals:
                groups.append(vals)

        if len(groups) < 2:
            continue

        try:
            stat, p = kruskal(*groups)
            test    = "kruskal_wallis"
        except Exception:
            continue

        interpretation = (
            "significant site effect (p<0.05) — harmonization recommended"
            if p < 0.05
            else "no significant site effect at p<0.05 (exploratory)"
        )

        rows.append({
            "metric":         mn,
            "test":           test,
            "statistic":      _r(stat, 4),
            "p_value":        _r(p, 6),
            "interpretation": interpretation,
        })

    return rows


# ---------------------------------------------------------------------------
# Experimental site z-score normalization
# ---------------------------------------------------------------------------

def compute_zscore_edge_table(
    edge_header: list[str],
    edge_rows:   list[list],
    site_col_idx: int = 1,
) -> list[list]:
    """
    Per-site z-score normalization of edge values.
    Returns new rows with same structure — edge values z-scored by site.
    EXPERIMENTAL: not for direct modeling.
    """
    meta_cols = 6  # subject_id, site, dataset, diagnosis, age, sex
    arr = np.full((len(edge_rows), N_EDGES), np.nan)

    for i, row in enumerate(edge_rows):
        for j, v in enumerate(row[meta_cols:meta_cols + N_EDGES]):
            if v != "":
                try:
                    arr[i, j] = float(v)
                except ValueError:
                    pass

    sites = [row[site_col_idx] for row in edge_rows]
    arr_z = arr.copy()

    for site in CLEAN_SITES:
        idx = [i for i, s in enumerate(sites) if s == site]
        if len(idx) < 2:
            continue
        site_arr = arr[idx, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu    = np.nanmean(site_arr, axis=0)
            sigma = np.nanstd(site_arr, axis=0)
            sigma[sigma == 0] = 1.0  # avoid division by zero
        arr_z[np.ix_(idx, np.arange(N_EDGES))] = (site_arr - mu) / sigma

    new_rows = []
    for i, row in enumerate(edge_rows):
        edge_vals = [str(v) if not np.isnan(v) else "" for v in arr_z[i]]
        new_rows.append(list(row[:meta_cols]) + edge_vals)

    return new_rows


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def run_qc_harmonization(
    processed_v2b_root: Path,
    raw_root:           Path,
    sites:              list[str] = CLEAN_SITES,
    skip_if_exists:     bool = False,
) -> dict:
    """
    Full Phase 3R.4 pipeline.
    Returns dict with metrics, edge tables, site effects, stat tests.
    """
    guard_no_raw_write(processed_v2b_root, raw_root)

    # Load preprocessing summary for b_value/direction_count
    preproc_lookup: dict[str, dict] = {}
    preproc_path = processed_v2b_root.parent / "phase2r_1_standard_dwi_preprocessing_summary.csv"
    if preproc_path.exists():
        with open(preproc_path) as f:
            for row in csv.DictReader(f):
                preproc_lookup[row["subject_id"]] = row

    # Load QC summary for FA/MD means
    qc_lookup: dict[str, dict] = {}
    qc_path = processed_v2b_root.parent / "phase2r_3_brain_mask_tensor_qc_summary.csv"
    if qc_path.exists():
        with open(qc_path) as f:
            for row in csv.DictReader(f):
                qc_lookup[row["subject_id"]] = row

    # Load connectome summary for density/streamline stats
    conn_lookup: dict[str, dict] = {}
    conn_path = processed_v2b_root.parent / "phase3r_3_connectome_summary.csv"
    if conn_path.exists():
        with open(conn_path) as f:
            for row in csv.DictReader(f):
                conn_lookup[row["subject_id"]] = row

    all_metrics:  list[GraphMetrics] = []
    all_subjects: list[dict] = []

    for site in sites:
        if site in EXCLUDED_SITES:
            continue
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = processed_v2b_root / folder
        if not site_root.exists():
            logger.warning("[%s] not found", site)
            continue

        connectome_dirs = sorted(site_root.rglob("connectome"))
        connectome_dirs = [d for d in connectome_dirs if d.is_dir() and
                          (d / "count_matrix.npy").exists()]
        logger.info("[%s] %d subjects with connectomes", site, len(connectome_dirs))

        for conn_dir in connectome_dirs:
            # Parse identity: .../dti_1/connectome → subject_id, dataset
            subject_id  = conn_dir.parents[2].name
            dataset     = conn_dir.parents[3].name

            matrices = load_matrices(conn_dir)
            if matrices is None:
                logger.warning("[%s/%s] failed to load matrices", site, subject_id)
                gm = GraphMetrics(subject_id=subject_id, site=site, dataset=dataset,
                                  status="failed", warning_message="matrix load failed")
                all_metrics.append(gm)
                continue

            gm = compute_graph_metrics(subject_id, site, dataset, matrices)
            all_metrics.append(gm)

            # Build meta from lookups
            meta = {}
            preproc = preproc_lookup.get(subject_id, {})
            qc      = qc_lookup.get(subject_id, {})
            conn    = conn_lookup.get(subject_id, {})
            if preproc:
                unique_bvals = preproc.get("unique_bvals", "")
                try:
                    import ast
                    bvals_list = ast.literal_eval(unique_bvals)
                    b_val = max(int(b) for b in bvals_list if int(b) > 100) if bvals_list else None
                except Exception:
                    b_val = None
                meta["b_value"]          = b_val
                meta["direction_count"]  = preproc.get("dwi_count", "")
            if qc:
                meta["mean_fa"] = qc.get("fa_mean", "")
                meta["mean_md"] = qc.get("md_mean", "")
            if conn:
                meta["density"]               = conn.get("density", "")
                meta["mean_streamline_length"] = conn.get("mean_edge_length", "")
            # Phenotype: not available in this project
            meta["age"]        = ""
            meta["sex"]        = ""
            meta["diagnosis"]  = ""
            meta["scanner_info"] = ""

            all_subjects.append({
                "subject_id": subject_id,
                "site":       site,
                "dataset":    dataset,
                "matrices":   matrices,
                "meta":       meta,
            })

    # Site z-scores
    compute_site_zscores_for_metrics(all_metrics)

    # Site effects
    metric_names = ["density", "nonzero_edges", "mean_edge_weight",
                    "mean_edge_length", "global_fa_edge_mean",
                    "degree_mean", "weighted_degree_mean"]
    site_effects = compute_site_effects(all_metrics, metric_names)

    # Statistical tests
    stat_tests = run_statistical_tests(all_metrics, metric_names)

    # Edge tables
    edge_tables: dict[str, tuple[list, list]] = {}
    for key in ["count", "length", "fa", "md"]:
        header, rows = build_edge_table(all_subjects, key)
        edge_tables[key] = (header, rows)

    # Harmonization metadata
    harmeta_rows = []
    for s in all_subjects:
        meta = s["meta"]
        gm   = next((m for m in all_metrics if m.subject_id == s["subject_id"] and m.site == s["site"]), None)
        harmeta_rows.append({
            "subject_id":             s["subject_id"],
            "site":                   s["site"],
            "dataset":                s["dataset"],
            "age":                    meta.get("age", ""),
            "sex":                    meta.get("sex", ""),
            "diagnosis":              meta.get("diagnosis", ""),
            "scanner_info":           meta.get("scanner_info", ""),
            "b_value":                meta.get("b_value", ""),
            "direction_count":        meta.get("direction_count", ""),
            "mean_fa":                meta.get("mean_fa", ""),
            "mean_md":                meta.get("mean_md", ""),
            "density":                _r(gm.density, 6) if gm else "",
            "mean_streamline_length": meta.get("mean_streamline_length", ""),
        })

    return {
        "metrics":      all_metrics,
        "edge_tables":  edge_tables,
        "harmeta_rows": harmeta_rows,
        "site_effects": site_effects,
        "stat_tests":   stat_tests,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_all_outputs(
    results:     dict,
    out_root:    Path,
    raw_root:    Path,
) -> dict[str, Path]:
    """Write all Phase 3R.4 outputs. Returns dict of {name: path}."""
    guard_no_raw_write(out_root, raw_root)

    feat_dir   = out_root / "connectome_features"
    zscore_dir = feat_dir / "experimental_site_zscore"
    feat_dir.mkdir(parents=True, exist_ok=True)
    zscore_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Graph metrics CSV
    metrics_path = feat_dir / "subject_graph_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRAPH_METRICS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_row() for r in results["metrics"])
    paths["graph_metrics"] = metrics_path
    logger.info("Graph metrics → %s  (%d rows)", metrics_path, len(results["metrics"]))

    # Edge tables
    for key, label in [("count","count"), ("length","length"), ("fa","fa"), ("md","md")]:
        header, rows = results["edge_tables"][key]
        p = feat_dir / f"{label}_edges.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        paths[f"{label}_edges"] = p
        logger.info("%s edges → %s  (%d subjects)", label, p, len(rows))

        # Experimental z-score version
        if rows:
            zrows = compute_zscore_edge_table(header, rows)
            zp    = zscore_dir / f"{label}_edges_zscore.csv"
            with open(zp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(zrows)
            paths[f"{label}_edges_zscore"] = zp

    # Harmonization metadata
    hm_path = feat_dir / "harmonization_metadata.csv"
    with open(hm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HARMONIZATION_META_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results["harmeta_rows"])
    paths["harmonization_metadata"] = hm_path
    logger.info("Harmonization metadata → %s", hm_path)

    # QC summary (same as graph metrics but in phase output dir)
    qc_path = out_root / "phase3r_4_connectome_qc_summary.csv"
    with open(qc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GRAPH_METRICS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_row() for r in results["metrics"])
    paths["qc_summary"] = qc_path

    # Site effect summary
    se_path = out_root / "phase3r_4_site_effect_summary.csv"
    with open(se_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SITE_EFFECT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results["site_effects"])
    paths["site_effects"] = se_path
    logger.info("Site effect summary → %s", se_path)

    # Statistical tests
    st_path = out_root / "phase3r_4_statistical_tests.csv"
    with open(st_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STAT_TEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results["stat_tests"])
    paths["stat_tests"] = st_path
    logger.info("Statistical tests → %s", st_path)

    return paths


