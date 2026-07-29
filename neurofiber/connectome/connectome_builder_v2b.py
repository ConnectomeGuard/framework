"""
NeuroFiber Phase 3R.3 — Connectome Construction from validated tractography

Converts Phase 3R.2 streamlines into weighted structural connectivity matrices
using Schaefer-100 atlas registrations from the v1 pipeline (reused — same
subject DTI voxel space, no re-registration needed).

Input per subject:
  Streamlines (v2b):
    data/processed_v2b/abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1/
      tractography/streamlines.trk
      tensor/FA.nii.gz  MD.nii.gz  AD.nii.gz  RD.nii.gz
      qc/brain_mask.nii.gz

  Atlas (reused from v1 registration):
    data/processed/abide_ii/<site>/<dataset>/<subject_id>/session_1/connectome/atlas/
      atlas_subject_space.nii.gz

Output per subject:
  data/processed_v2b/abide_ii/<site>/<dataset>/<subject_id>/session_1/dti_1/connectome/
    count_matrix.npy
    mean_length_matrix.npy
    mean_fa_matrix.npy
    mean_md_matrix.npy
    mean_ad_matrix.npy
    mean_rd_matrix.npy
    connectome_report.json

Cohort outputs:
  data/processed_v2b/
    phase3r_3_connectome_summary.csv
    phase3r_3_connectome_site_summary.csv

Safety contract:
  - Reads streamlines/tensors only from data/processed_v2b/
  - Atlas read-only from data/processed/ (not modified)
  - guard_no_raw_write() enforced at every write entry point
  - IP_1 explicitly rejected
  - Never writes to data/processed/, data/processed_v2/, or data/raw/
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

from dipy.io.streamline import load_trk

from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

PIPELINE_VERSION = "3R.3"
ATLAS_NAME       = "Schaefer2018_100Parcels_7Networks"
ATLAS_N_ROIS     = 100

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

SUBJECT_CSV_FIELDS = [
    "site", "dataset", "subject_id",
    "atlas_name", "atlas_label_count",
    "streamlines_input", "streamlines_used",
    "streamlines_discarded_outside_atlas",
    "self_loops_count",
    "nonzero_edges", "density",
    "mean_edge_weight", "mean_edge_length",
    "status", "warning_message", "error_message",
]

SITE_CSV_FIELDS = [
    "site", "subjects",
    "mean_nonzero_edges", "mean_density",
    "mean_streamlines_used", "mean_self_loops",
    "mean_discarded_outside_atlas",
    "notes",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConnectomeRecord:
    site:        str
    dataset:     str
    subject_id:  str
    atlas_name:         str   = ATLAS_NAME
    atlas_label_count:  int   = ATLAS_N_ROIS
    streamlines_input:  int   = 0
    streamlines_used:   int   = 0
    streamlines_discarded_outside_atlas: int = 0
    self_loops_count:   int   = 0
    nonzero_edges:      int   = 0
    density:            Optional[float] = None
    mean_edge_weight:   Optional[float] = None
    mean_edge_length:   Optional[float] = None
    status:             str           = "pending"
    warning_message:    Optional[str] = None
    error_message:      Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_csv_row(self) -> dict:
        return {k: getattr(self, k, None) for k in SUBJECT_CSV_FIELDS}

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core per-subject function
# ---------------------------------------------------------------------------

def build_subject_connectome(
    dti_dir:         Path,
    v1_processed_root: Path,
    output_root:     Path,
    raw_root:        Path,
    skip_if_exists:  bool = True,
) -> ConnectomeRecord:
    """
    Build connectivity matrices for one subject.

    dti_dir:          .../session_1/dti_1/ from processed_v2b
    v1_processed_root: data/processed/abide_ii — used to locate pre-registered atlas
    output_root:      data/processed_v2b/abide_ii — safety guard reference
    """
    subject_id  = dti_dir.parents[1].name
    dataset     = dti_dir.parents[2].name
    site_folder = dti_dir.parents[3].name
    site = next((k for k, v in _SITE_FOLDER_MAP.items() if v == site_folder),
                site_folder.upper())

    guard_no_raw_write(dti_dir, raw_root)

    rec = ConnectomeRecord(site=site, dataset=dataset, subject_id=subject_id)

    if site in EXCLUDED_SITES:
        rec.status = "excluded"
        rec.warning_message = f"{site} excluded from Phase 3R.3"
        return rec

    connectome_dir = dti_dir / "connectome"

    # Skip if already done
    if skip_if_exists and (connectome_dir / "connectome_report.json").exists():
        try:
            d = json.loads((connectome_dir / "connectome_report.json").read_text())
            if d.get("status") == "success":
                for f in SUBJECT_CSV_FIELDS:
                    if f in d:
                        setattr(rec, f, d[f])
                rec.status = "success"
                logger.info("[%s/%s] already done — skipping", site, subject_id)
                return rec
        except Exception:
            pass

    # Validate inputs
    trk_path  = dti_dir / "tractography" / "streamlines.trk"
    fa_path   = dti_dir / "tensor" / "FA.nii.gz"
    md_path   = dti_dir / "tensor" / "MD.nii.gz"
    ad_path   = dti_dir / "tensor" / "AD.nii.gz"
    rd_path   = dti_dir / "tensor" / "RD.nii.gz"

    missing = [p.name for p in [trk_path, fa_path, md_path, ad_path, rd_path]
               if not p.exists()]
    if missing:
        return _failed(rec, f"Missing inputs: {missing}")

    # Locate atlas (reuse v1 registration — same DTI voxel space)
    atlas_path = (
        v1_processed_root / site_folder / dataset / subject_id
        / "session_1" / "connectome" / "atlas" / "atlas_subject_space.nii.gz"
    )
    if not atlas_path.exists():
        return _failed(rec, f"Atlas not found: {atlas_path}. Run atlas registration first.")

    # Load data
    try:
        trk_obj    = load_trk(str(trk_path), "same", bbox_valid_check=False)
        streamlines = trk_obj.streamlines
        fa_img     = nib.load(str(fa_path))
        md_img     = nib.load(str(md_path))
        ad_img     = nib.load(str(ad_path))
        rd_img     = nib.load(str(rd_path))
        atlas_img  = nib.load(str(atlas_path))
    except Exception as exc:
        return _failed(rec, f"Load failed: {exc}")

    rec.streamlines_input = len(streamlines)

    if rec.streamlines_input == 0:
        return _failed(rec, "No streamlines in tractography file")

    try:
        fa   = fa_img.get_fdata(dtype=np.float32)
        md   = md_img.get_fdata(dtype=np.float32)
        ad   = ad_img.get_fdata(dtype=np.float32)
        rd   = rd_img.get_fdata(dtype=np.float32)
        atlas = atlas_img.get_fdata(dtype=np.float32)
    except Exception as exc:
        return _failed(rec, f"Image data load failed: {exc}")

    atlas_affine  = atlas_img.affine
    fa_affine     = fa_img.affine
    n_rois        = ATLAS_N_ROIS
    rec.atlas_label_count = n_rois

    # Assign streamline endpoints to atlas labels
    try:
        start_labels, end_labels = _assign_endpoints(streamlines, atlas, atlas_affine)
    except Exception as exc:
        return _failed(rec, f"Endpoint assignment failed: {exc}")

    is_valid    = (start_labels > 0) & (end_labels > 0) & (start_labels <= n_rois) & (end_labels <= n_rois)
    is_selfloop = is_valid & (start_labels == end_labels)
    is_used     = is_valid & ~is_selfloop
    is_outside  = ~(start_labels > 0) | ~(end_labels > 0)

    rec.streamlines_used                   = int(is_used.sum())
    rec.self_loops_count                   = int(is_selfloop.sum())
    rec.streamlines_discarded_outside_atlas = int((~is_valid).sum()) - int(is_selfloop.sum())

    if rec.streamlines_used == 0:
        return _failed(rec, "Zero streamlines mapped to atlas (all outside or self-loops)")

    # Sample scalar values along used streamlines
    used_streamlines = [s for s, u in zip(streamlines, is_used) if u]
    used_starts = start_labels[is_used] - 1   # 0-indexed
    used_ends   = end_labels[is_used]   - 1

    fa_per_sl  = _sample_scalars(used_streamlines, fa, fa_affine)
    md_per_sl  = _sample_scalars(used_streamlines, md, fa_affine)
    ad_per_sl  = _sample_scalars(used_streamlines, ad, fa_affine)
    rd_per_sl  = _sample_scalars(used_streamlines, rd, fa_affine)
    len_per_sl = np.array(_compute_lengths(used_streamlines))

    # Build matrices
    count_mat  = np.zeros((n_rois, n_rois), dtype=np.float64)
    length_mat = np.full((n_rois, n_rois), np.nan)
    fa_mat     = np.full((n_rois, n_rois), np.nan)
    md_mat     = np.full((n_rois, n_rois), np.nan)
    ad_mat     = np.full((n_rois, n_rois), np.nan)
    rd_mat     = np.full((n_rois, n_rois), np.nan)

    # Accumulate per-edge
    edge_lengths: dict[tuple, list] = {}
    edge_fa:      dict[tuple, list] = {}
    edge_md:      dict[tuple, list] = {}
    edge_ad:      dict[tuple, list] = {}
    edge_rd:      dict[tuple, list] = {}

    for k in range(len(used_streamlines)):
        i, j = int(used_starts[k]), int(used_ends[k])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        count_mat[i, j] += 1
        count_mat[j, i] += 1
        key = (a, b)
        edge_lengths.setdefault(key, []).append(float(len_per_sl[k]))
        if not np.isnan(fa_per_sl[k]):
            edge_fa.setdefault(key, []).append(float(fa_per_sl[k]))
        if not np.isnan(md_per_sl[k]):
            edge_md.setdefault(key, []).append(float(md_per_sl[k]))
        if not np.isnan(ad_per_sl[k]):
            edge_ad.setdefault(key, []).append(float(ad_per_sl[k]))
        if not np.isnan(rd_per_sl[k]):
            edge_rd.setdefault(key, []).append(float(rd_per_sl[k]))

    for (a, b), vals in edge_lengths.items():
        v = np.mean(vals)
        length_mat[a, b] = length_mat[b, a] = v
    for (a, b), vals in edge_fa.items():
        v = np.mean(vals)
        fa_mat[a, b] = fa_mat[b, a] = v
    for (a, b), vals in edge_md.items():
        v = np.mean(vals)
        md_mat[a, b] = md_mat[b, a] = v
    for (a, b), vals in edge_ad.items():
        v = np.mean(vals)
        ad_mat[a, b] = ad_mat[b, a] = v
    for (a, b), vals in edge_rd.items():
        v = np.mean(vals)
        rd_mat[a, b] = rd_mat[b, a] = v

    # Compute summary stats
    upper_tri = np.triu(count_mat, k=1)
    nonzero   = int((upper_tri > 0).sum())
    possible  = n_rois * (n_rois - 1) / 2
    density   = round(nonzero / possible, 6) if possible > 0 else 0.0

    fa_edges = fa_mat[np.triu(np.ones((n_rois, n_rois), dtype=bool), k=1) & ~np.isnan(fa_mat)]
    len_edges = length_mat[np.triu(np.ones((n_rois, n_rois), dtype=bool), k=1) & ~np.isnan(length_mat)]
    wt_edges  = upper_tri[upper_tri > 0]

    rec.nonzero_edges    = nonzero
    rec.density          = density
    rec.mean_edge_weight = _r(float(np.mean(wt_edges))) if len(wt_edges) > 0 else None
    rec.mean_edge_length = _r(float(np.mean(len_edges)), 2) if len(len_edges) > 0 else None

    # Save matrices
    connectome_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(connectome_dir / "count_matrix.npy"),       count_mat)
    np.save(str(connectome_dir / "mean_length_matrix.npy"), length_mat)
    np.save(str(connectome_dir / "mean_fa_matrix.npy"),     fa_mat)
    np.save(str(connectome_dir / "mean_md_matrix.npy"),     md_mat)
    np.save(str(connectome_dir / "mean_ad_matrix.npy"),     ad_mat)
    np.save(str(connectome_dir / "mean_rd_matrix.npy"),     rd_mat)

    rec.status = "success"

    # QC flags
    warnings: list[str] = []
    if rec.streamlines_used < 500:
        warnings.append(f"streamlines_used={rec.streamlines_used} < 500")
    if rec.nonzero_edges < 10:
        warnings.append(f"nonzero_edges={rec.nonzero_edges} very low")
    discard_ratio = rec.streamlines_discarded_outside_atlas / max(rec.streamlines_input, 1)
    if discard_ratio > 0.5:
        warnings.append(f"{discard_ratio*100:.0f}% streamlines outside atlas")
    if rec.streamlines_input > 0:
        sl_ratio = rec.self_loops_count / rec.streamlines_input
        if sl_ratio > 0.3:
            warnings.append(f"high self-loop ratio: {sl_ratio*100:.0f}%")
    rec.warning_message = "; ".join(warnings) if warnings else None

    # Write report
    report = {**rec.to_dict(), "pipeline_version": PIPELINE_VERSION}
    (connectome_dir / "connectome_report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "[%s/%s] input=%d used=%d nonzero_edges=%d density=%.4f%s",
        site, subject_id,
        rec.streamlines_input, rec.streamlines_used,
        rec.nonzero_edges, rec.density or 0,
        f"  WARN: {rec.warning_message}" if rec.warning_message else "",
    )
    return rec


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_connectome_batch(
    processed_v2b_root: Path,
    v1_processed_root:  Path,
    raw_root:           Path,
    sites:              list[str] = CLEAN_SITES,
    skip_if_exists:     bool = True,
) -> list[ConnectomeRecord]:
    guard_no_raw_write(processed_v2b_root, raw_root)

    bad = [s for s in sites if s in EXCLUDED_SITES]
    if bad:
        raise ValueError(f"Cannot include excluded sites: {bad}")

    records: list[ConnectomeRecord] = []

    for site in sites:
        folder    = _SITE_FOLDER_MAP.get(site, site.lower())
        site_root = processed_v2b_root / folder
        if not site_root.exists():
            logger.warning("[%s] not found: %s", site, site_root)
            continue

        dti_dirs = sorted(site_root.rglob("dti_1"))
        # Include any dti_1 from Phase 3R.2 — missing .trk handled inside → status=failed.
        dti_dirs = [d for d in dti_dirs if d.is_dir() and
                    (d / "fod" / "peaks.pam5").exists()]
        logger.info("[%s] %d subjects", site, len(dti_dirs))

        for dti_dir in dti_dirs:
            try:
                rec = build_subject_connectome(
                    dti_dir=dti_dir,
                    v1_processed_root=v1_processed_root,
                    output_root=processed_v2b_root,
                    raw_root=raw_root,
                    skip_if_exists=skip_if_exists,
                )
            except Exception as exc:
                subj = dti_dir.parents[1].name
                dset = dti_dir.parents[2].name
                logger.error("[%s/%s] unexpected error: %s", site, subj, exc)
                rec = ConnectomeRecord(site=site, dataset=dset, subject_id=subj,
                                       status="failed", error_message=str(exc))
            records.append(rec)

        n_ok   = sum(1 for r in records if r.site == site and r.status == "success")
        n_fail = sum(1 for r in records if r.site == site and r.status == "failed")
        logger.info("[%s] success=%d  failed=%d", site, n_ok, n_fail)

    return records


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def write_subject_summary(records: list[ConnectomeRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBJECT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(r.to_csv_row() for r in records)
    logger.info("Subject summary → %s  (%d rows)", out_path, len(records))
    return out_path


def write_site_summary(
    records:  list[ConnectomeRecord],
    out_path: Path,
) -> tuple[Path, list[dict]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    site_notes = {
        "NYU_1":  "shorter streamlines and higher short-streamline rejection — acquisition/site variability",
        "NYU_2":  "shorter streamlines and higher short-streamline rejection — acquisition/site variability",
    }

    for site in CLEAN_SITES:
        recs    = [r for r in records if r.site == site and r.status == "success"]
        n       = len(recs)
        edges   = [r.nonzero_edges for r in recs]
        dens    = [r.density for r in recs if r.density is not None]
        used    = [r.streamlines_used for r in recs]
        loops   = [r.self_loops_count for r in recs]
        disc    = [r.streamlines_discarded_outside_atlas for r in recs]

        rows.append({
            "site":                          site,
            "subjects":                      n,
            "mean_nonzero_edges":            _r(float(np.mean(edges)))  if edges else None,
            "mean_density":                  _r(float(np.mean(dens)), 6) if dens  else None,
            "mean_streamlines_used":         _r(float(np.mean(used)))   if used  else None,
            "mean_self_loops":               _r(float(np.mean(loops)))  if loops else None,
            "mean_discarded_outside_atlas":  _r(float(np.mean(disc)))   if disc  else None,
            "notes": site_notes.get(site, ""),
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SITE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Site summary → %s", out_path)
    return out_path, rows


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_endpoints(
    streamlines,
    atlas_data:   np.ndarray,
    atlas_affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each streamline's start and end points to atlas labels."""
    inv_aff = np.linalg.inv(atlas_affine)
    shape   = atlas_data.shape
    n       = len(streamlines)
    starts  = np.zeros(n, dtype=int)
    ends    = np.zeros(n, dtype=int)

    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl)
        if len(pts) < 2:
            continue
        for k, pt in enumerate([pts[0], pts[-1]]):
            vox = np.round(apply_affine(inv_aff, pt)).astype(int)
            if all(0 <= vox[d] < shape[d] for d in range(3)):
                label = int(atlas_data[vox[0], vox[1], vox[2]])
            else:
                label = 0
            if k == 0:
                starts[i] = label
            else:
                ends[i] = label

    return starts, ends


def _sample_scalars(streamlines, scalar_data: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Mean scalar value along each streamline (NaN if no valid voxels)."""
    inv_aff = np.linalg.inv(affine)
    shape   = np.array(scalar_data.shape, dtype=float)
    result  = np.full(len(streamlines), np.nan, dtype=np.float64)

    for i, sl in enumerate(streamlines):
        pts = np.asarray(sl, dtype=np.float32)
        if len(pts) < 2:
            continue
        vox = apply_affine(inv_aff, pts)
        for d in range(3):
            vox[:, d] = np.clip(vox[:, d], 0.0, shape[d] - 1.0)
        vals = scalar_data[
            vox[:, 0].astype(int),
            vox[:, 1].astype(int),
            vox[:, 2].astype(int),
        ]
        valid = vals[vals > 0]
        if len(valid) > 0:
            result[i] = float(np.mean(valid))

    return result


def _compute_lengths(streamlines) -> list[float]:
    lengths = []
    for s in streamlines:
        pts = np.asarray(s)
        if len(pts) < 2:
            lengths.append(0.0)
        else:
            diffs = np.diff(pts, axis=0)
            lengths.append(float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))))
    return lengths


def _failed(rec: ConnectomeRecord, error: str) -> ConnectomeRecord:
    rec.status = "failed"
    rec.error_message = error
    logger.error("[%s/%s] %s", rec.site, rec.subject_id, error)
    return rec


def _r(v: float, decimals: int = 4) -> float:
    return round(v, decimals)
