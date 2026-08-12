"""
Phase 3.5A — Deterministic Connectome Batch Runner, Comparison Analysis,
             Graph Property Characterisation, and Report Generation

Three responsibilities:
  1. run_det_connectome_batch() — build connectomes from tractography_det/ streamlines,
     reusing existing Phase 3.4 atlas registrations to avoid re-registration.
  2. compare_connectomes() — per-subject and per-site comparison of Phase 3.5A
     (interface-seeded det.) vs Phase 3.4 (WM-seeded baseline).
  3. compute_graph_properties() — degree distribution, hub regions,
     inter/intra-hemispheric balance, published-literature comparison.
"""

from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd

from neurofiber.connectome.atlas_registration import (
    AtlasInfo,
    RegisteredAtlas,
    load_registered_atlas,
    register_atlas_to_subject,
)
from neurofiber.connectome.connectome_builder import build_subject_connectome, ConnectomeRecord
from neurofiber.connectome.connectome_qc import save_summary_csvs
from neurofiber.tractography.streamline_generation import site_to_folder
from neurofiber.tractography.det_tractography import TRACTOGRAPHY_SUBDIR
from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

logger = get_logger(__name__)

CONNECTOME_SUBDIR = "connectome_det"

# Schaefer100 hemisphere split: labels 1-50 = LH, 51-100 = RH (matrix indices 0-49 / 50-99)
_LH_CUTOFF = 50  # matrix index threshold — indices < _LH_CUTOFF are left hemisphere


# ---------------------------------------------------------------------------
# 1. Det connectome batch runner
# ---------------------------------------------------------------------------

def run_det_connectome_batch(
    processed_root: Path,
    raw_root:       Path,
    sites:          list[str],
    atlas_info:     AtlasInfo,
    skip_if_exists: bool = True,
    registration_level_iters: Optional[list[int]] = None,
    registration_factors:     Optional[list[int]] = None,
) -> list[ConnectomeRecord]:
    """
    Build connectomes from tractography_det/ streamlines for all subjects.

    Atlas registrations are reused from Phase 3.4 (connectome/atlas/) where
    available, avoiding re-registration. Falls back to fresh registration when
    the Phase 3.4 atlas is missing.
    """
    records: list[ConnectomeRecord] = []

    reg_kwargs: dict = {}
    if registration_level_iters is not None:
        reg_kwargs["level_iters"] = registration_level_iters
    if registration_factors is not None:
        reg_kwargs["factors"] = registration_factors

    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = processed_root / folder
        if not site_dir.exists():
            logger.warning("[%s] site directory not found: %s", folder, site_dir)
            continue

        dti_dirs = sorted(site_dir.glob("*/*/session_1/dti_1"))
        if not dti_dirs:
            logger.warning("[%s] no dti_1 dirs found", folder)
            continue

        logger.info("[%s] det connectome %d subjects …", folder, len(dti_dirs))

        for dti_dir in dti_dirs:
            subject_id  = dti_dir.parent.parent.name
            session_dir = dti_dir.parent
            trk_path    = dti_dir / TRACTOGRAPHY_SUBDIR / "streamlines.trk"

            if not trk_path.exists():
                logger.warning("[%s/%s] tractography_det/streamlines.trk missing — skip", folder, subject_id)
                records.append(_failed_record(folder, dti_dir, "tractography_det/streamlines.trk missing"))
                continue

            connectome_dir = session_dir / CONNECTOME_SUBDIR
            # Prefer existing Phase 3.4 atlas (saves ~25 s per subject)
            p34_atlas_dir  = session_dir / "connectome" / "atlas"
            det_atlas_dir  = connectome_dir / "atlas"

            try:
                atlas_nii = det_atlas_dir / "atlas_subject_space.nii.gz"
                if skip_if_exists and atlas_nii.exists():
                    reg_atlas = load_registered_atlas(det_atlas_dir, atlas_info.label_df)
                elif (p34_atlas_dir / "atlas_subject_space.nii.gz").exists():
                    # Reuse Phase 3.4 registration — copy-load from p34 atlas dir
                    reg_atlas = load_registered_atlas(p34_atlas_dir, atlas_info.label_df)
                    logger.debug("[%s/%s] reusing Phase 3.4 atlas registration", folder, subject_id)
                else:
                    t1w_path = raw_root / folder / dti_dir.parent.parent.parent.name / subject_id / "session_1" / "anat_1" / "anat.nii.gz"
                    fa_path  = dti_dir / "tensor" / "FA.nii.gz"
                    if not t1w_path.exists():
                        raise FileNotFoundError(f"T1w not found: {t1w_path}")
                    if not fa_path.exists():
                        raise FileNotFoundError(f"FA not found: {fa_path}")
                    logger.info("[%s/%s] registering atlas (no Phase 3.4 atlas found) …", folder, subject_id)
                    reg_atlas = register_atlas_to_subject(
                        t1w_path=t1w_path, fa_path=fa_path,
                        atlas_info=atlas_info, output_dir=det_atlas_dir,
                        raw_root=None, **reg_kwargs,
                    )

                rec = build_subject_connectome(
                    dti_dir=dti_dir,
                    registered_atlas=reg_atlas,
                    output_dir=connectome_dir,
                    raw_root=raw_root,
                    skip_if_exists=skip_if_exists,
                    trk_path=trk_path,
                )
                records.append(rec)

            except Exception as exc:
                logger.warning("[%s/%s] FAILED: %s", folder, subject_id, exc)
                records.append(_failed_record(folder, dti_dir, str(exc)))

    return records


def _failed_record(folder: str, dti_dir: Path, msg: str) -> ConnectomeRecord:
    return ConnectomeRecord(
        site=folder,
        dataset=dti_dir.parent.parent.parent.name,
        subject_id=dti_dir.parent.parent.name,
        node_count=0, edge_count=0, graph_density=0.0, mean_degree=0.0,
        mean_edge_fa=None, mean_edge_md=None, mean_edge_length=None,
        streamline_count_total=0, mapped_streamline_count=0,
        unmapped_streamline_count=0, self_loop_count=0,
        mapping_success_ratio=0.0,
        status="failed", error_message=msg,
    )


# ---------------------------------------------------------------------------
# 2. Comparison analysis
# ---------------------------------------------------------------------------

def compare_connectomes(
    baseline_csv: Path,
    det_csv:      Path,
) -> pd.DataFrame:
    """
    Per-subject comparison of Phase 3.5A (det) vs Phase 3.4 (baseline).

    Returns one row per subject with both sets of metrics and their deltas.
    """
    base = pd.read_csv(baseline_csv).rename(columns={
        c: f"base_{c}" for c in [
            "edge_count", "graph_density", "mean_edge_fa", "mean_edge_md",
            "mean_edge_length", "mapping_success_ratio", "streamline_count_total",
            "mapped_streamline_count",
        ]
    })
    det  = pd.read_csv(det_csv).rename(columns={
        c: f"det_{c}" for c in [
            "edge_count", "graph_density", "mean_edge_fa", "mean_edge_md",
            "mean_edge_length", "mapping_success_ratio", "streamline_count_total",
            "mapped_streamline_count",
        ]
    })
    both = base.merge(det, on=["site", "subject_id"], suffixes=("", "_det"))

    for metric in ["edge_count", "graph_density", "mean_edge_fa",
                   "mean_edge_md", "mean_edge_length", "mapping_success_ratio"]:
        b, d = f"base_{metric}", f"det_{metric}"
        if b in both.columns and d in both.columns:
            both[f"delta_{metric}"] = both[d] - both[b]
            with np.errstate(divide="ignore", invalid="ignore"):
                both[f"pct_{metric}"] = np.where(
                    both[b] != 0,
                    100.0 * (both[d] - both[b]) / both[b].abs(),
                    np.nan,
                )

    return both


def compare_sites(
    baseline_site_csv: Path,
    det_site_csv:      Path,
) -> pd.DataFrame:
    """Per-site comparison table."""
    base = pd.read_csv(baseline_site_csv).add_prefix("base_").rename(columns={"base_site": "site"})
    det  = pd.read_csv(det_site_csv).add_prefix("det_").rename(columns={"det_site": "site"})
    return base.merge(det, on="site")


# ---------------------------------------------------------------------------
# 3. Graph property characterisation
# ---------------------------------------------------------------------------

def compute_graph_properties(
    adj_count: np.ndarray,
    label_df:  pd.DataFrame,
) -> dict:
    """
    Compute graph properties for published-literature comparison.

    Returns
    -------
    dict with:
      global metrics (density, clustering, path_length, small_world_sigma)
      degree_distribution (mean, std, max, gini, is_heavy_tail)
      hub_nodes (top-10 by degree and betweenness)
      hemisphere_balance (intra_lh, intra_rh, interhemispheric counts + ratio)
    """
    n = adj_count.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_count[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    degrees = np.array([d for _, d in G.degree()])
    n_edges = G.number_of_edges()
    density = nx.density(G)

    # Clustering
    try:
        clustering = float(nx.average_clustering(G))
    except Exception:
        clustering = 0.0

    # Characteristic path length (largest component)
    cpl = None
    if n_edges > 0:
        try:
            comps = list(nx.connected_components(G))
            lcc   = G.subgraph(max(comps, key=len)).copy()
            if lcc.number_of_edges() > 0:
                cpl = float(nx.average_shortest_path_length(lcc, weight=None))
        except Exception:
            pass

    # Small-world sigma: σ = (C/C_rand) / (L/L_rand)
    # Approximate C_rand ≈ density, L_rand ≈ log(n)/log(k_mean) for Erdos-Renyi
    small_world_sigma = None
    if cpl and density > 0 and degrees.mean() > 1:
        k_mean = float(degrees.mean())
        l_rand = np.log(n) / np.log(max(k_mean, 2))
        c_rand = density
        if c_rand > 0 and l_rand > 0:
            small_world_sigma = round((clustering / c_rand) / (cpl / l_rand), 3)

    # Degree distribution
    gini = _gini(degrees) if len(degrees) > 0 else 0.0
    is_heavy_tail = bool(gini > 0.40)  # heuristic: heavy-tailed if Gini > 0.4

    # Hub nodes (top 10 by degree and betweenness)
    try:
        between = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        between = {i: 0.0 for i in range(n)}

    top_degree = sorted(range(n), key=lambda i: degrees[i], reverse=True)[:10]
    top_between = sorted(between.keys(), key=lambda i: between[i], reverse=True)[:10]

    def _roi_name(idx: int) -> str:
        label = idx + 1
        row = label_df[label_df["index"] == label]
        return str(row["name"].iloc[0]) if not row.empty else f"roi_{label:03d}"

    hub_degree = [{"roi_index": i + 1, "name": _roi_name(i), "degree": int(degrees[i])} for i in top_degree]
    hub_between = [{"roi_index": i + 1, "name": _roi_name(i), "betweenness": round(between[i], 5)} for i in top_between]

    # Hemisphere balance
    intra_lh = intra_rh = interhemi = 0
    for i, j in G.edges():
        lh_i = i < _LH_CUTOFF
        lh_j = j < _LH_CUTOFF
        if lh_i == lh_j:
            if lh_i:
                intra_lh += 1
            else:
                intra_rh += 1
        else:
            interhemi += 1

    interhemi_ratio = interhemi / max(n_edges, 1)

    return {
        "n_nodes":            n,
        "n_edges":            n_edges,
        "density":            round(density, 5),
        "mean_clustering":    round(clustering, 5),
        "characteristic_path_length": round(cpl, 3) if cpl else None,
        "small_world_sigma":  small_world_sigma,
        "degree_mean":        round(float(degrees.mean()), 2),
        "degree_std":         round(float(degrees.std()), 2),
        "degree_max":         int(degrees.max()) if len(degrees) > 0 else 0,
        "degree_gini":        round(gini, 4),
        "heavy_tail_degree":  is_heavy_tail,
        "hub_by_degree":      hub_degree,
        "hub_by_betweenness": hub_between,
        "intra_lh_edges":     intra_lh,
        "intra_rh_edges":     intra_rh,
        "interhemispheric_edges": interhemi,
        "interhemispheric_ratio": round(interhemi_ratio, 4),
    }


def compute_cohort_graph_properties(
    processed_root: Path,
    sites:          list[str],
    label_df:       pd.DataFrame,
    connectome_subdir: str = CONNECTOME_SUBDIR,
) -> list[dict]:
    """
    Load adjacency_streamline_count.npy for every subject and aggregate
    graph properties. Returns list of per-subject dicts.
    """
    rows = []
    for site_display in sites:
        folder   = site_to_folder(site_display)
        site_dir = processed_root / folder
        for adj_path in sorted(site_dir.glob(f"*/*/session_1/{connectome_subdir}/adjacency_streamline_count.npy")):
            subject_id = adj_path.parent.parent.parent.name
            try:
                adj = np.load(str(adj_path))
                props = compute_graph_properties(adj, label_df)
                props["site"]       = folder
                props["subject_id"] = subject_id
                rows.append(props)
            except Exception as exc:
                logger.warning("[%s/%s] graph props failed: %s", folder, subject_id, exc)
    return rows


# ---------------------------------------------------------------------------
# 4. QC flags
# ---------------------------------------------------------------------------

def build_qc_flags(
    det_df:      pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag subjects needing review in Phase 3.5A:
      - status != success
      - edge_count == 0
      - mapping_success_ratio < 0.03
      - graph_density < 0.01
      - large regression vs baseline (edge_count < 50% of baseline)
    """
    flags = []
    base_map = {
        row["subject_id"]: row
        for _, row in baseline_df.iterrows()
    }
    for _, row in det_df.iterrows():
        reasons = []
        if row.get("status") != "success":
            reasons.append("failed")
        if row.get("edge_count", 0) == 0:
            reasons.append("zero_edges")
        if row.get("mapping_success_ratio", 1.0) < 0.03:
            reasons.append("critically_low_mapping")
        if row.get("graph_density", 1.0) < 0.01:
            reasons.append("critically_low_density")
        base = base_map.get(row.get("subject_id", ""))
        if base is not None and base.get("edge_count", 0) > 0:
            ratio = row.get("edge_count", 0) / base["edge_count"]
            if ratio < 0.50:
                reasons.append(f"edge_count<50%_of_baseline({ratio:.2f}x)")
        if reasons:
            flags.append({
                "site":        row.get("site"),
                "subject_id":  row.get("subject_id"),
                "reasons":     "; ".join(reasons),
                "edge_count":  row.get("edge_count"),
                "graph_density": row.get("graph_density"),
                "mapping_ratio": row.get("mapping_success_ratio"),
            })
    return pd.DataFrame(flags)


# ---------------------------------------------------------------------------
# 5. Report generator
# ---------------------------------------------------------------------------

def generate_phase35a_report(
    det_site_csv:   Path,
    base_site_csv:  Path,
    comparison_df:  pd.DataFrame,
    graph_props:    list[dict],
    output_path:    Path,
    n_total:        int,
    n_success:      int,
    n_failed:       int,
    n_review:       int,
) -> Path:
    """Write PHASE_3_5A_REPORT.md."""
    det_site  = pd.read_csv(det_site_csv)
    base_site = pd.read_csv(base_site_csv)

    # Per-site comparison table
    merged = base_site.merge(det_site, on="site", suffixes=("_base", "_det"))

    site_table_rows = []
    for _, row in merged.iterrows():
        site_table_rows.append(
            f"| {row['site']:6s} "
            f"| {int(row.get('mean_edge_count_base', 0)):5.0f} "
            f"| {int(row.get('mean_edge_count_det', 0)):5.0f} "
            f"| {row.get('mean_graph_density_base', 0)*100:5.1f}% "
            f"| {row.get('mean_graph_density_det', 0)*100:5.1f}% "
            f"| {row.get('mean_edge_fa_base', 0):.3f} "
            f"| {row.get('mean_edge_fa_det', 0):.3f} "
            f"| {row.get('mean_mapping_success_ratio_base', 0)*100:5.1f}% "
            f"| {row.get('mean_mapping_success_ratio_det', 0)*100:5.1f}% |"
        )
    site_table = "\n".join(site_table_rows)

    # Mean deltas across all subjects
    def _pct(col: str) -> str:
        if f"pct_{col}" in comparison_df.columns:
            v = comparison_df[f"pct_{col}"].dropna().mean()
            return f"{v:+.1f}%"
        return "—"

    # Graph properties summary
    if graph_props:
        gdf = pd.DataFrame(graph_props)
        gp_summary = (
            f"  mean degree       : {gdf['degree_mean'].mean():.1f} ± {gdf['degree_mean'].std():.1f}\n"
            f"  mean clustering   : {gdf['mean_clustering'].mean():.4f}\n"
            f"  interhemi ratio   : {gdf['interhemispheric_ratio'].mean()*100:.1f}%\n"
            f"  heavy-tail degree : {gdf['heavy_tail_degree'].mean()*100:.0f}% of subjects"
        )
        top_hubs = _top_hub_names(graph_props)
    else:
        gp_summary = "  (no adjacency matrices loaded)"
        top_hubs   = "  (unavailable)"

    report = textwrap.dedent(f"""\
    # Phase 3.5A — Deterministic Tractography Baseline Report

    **Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}
    **Subjects:** {n_total}  |  **Success:** {n_success}  |  **Failed:** {n_failed}  |  **Review:** {n_review}

    ---

    ## 1. What Phase 3.5A Changes vs Phase 3.4

    | Parameter | Phase 3.4 (baseline) | Phase 3.5A (det.) |
    |---|---|---|
    | Seeding | WM mask (FA ≥ 0.15) | Interface mask (0.08 ≤ FA < 0.20) |
    | Seeds/subject | 5 000 | 10 000 |
    | Direction getter | DeterministicMaximumDirectionGetter | Same |
    | Peaks source | CSA-ODF peaks.pam5 | Same |
    | Stopping criterion | FA < 0.15 | FA < 0.10 |
    | Atlas registration | Phase 3.4 (reused) | Reused |
    | Output dir | connectome/ | connectome_det/ |

    Interface seeding places streamline endpoints near cortical parcel boundaries,
    increasing the atlas-mapping ratio compared to deep-WM seeding.

    ---

    ## 2. Site-Level Comparison

    | Site | Edges(B) | Edges(D) | Density(B) | Density(D) | FA(B) | FA(D) | Map%(B) | Map%(D) |
    |---|---|---|---|---|---|---|---|---|
    {site_table}

    *(B = Phase 3.4 baseline, D = Phase 3.5A deterministic)*

    ---

    ## 3. Mean Per-Subject Deltas (3.5A − 3.4)

    | Metric | Mean change |
    |---|---|
    | Edge count | {_pct('edge_count')} |
    | Graph density | {_pct('graph_density')} |
    | Mean edge FA | {_pct('mean_edge_fa')} |
    | Mapping ratio | {_pct('mapping_success_ratio')} |
    | Mean edge length | {_pct('mean_edge_length')} |

    ---

    ## 4. Graph Property Characterisation

    {gp_summary}

    **Top hub regions (by degree, cohort average):**
    {top_hubs}

    ---

    ## 5. Published Literature Comparison

    | Property | This study (3.5A det.) | Published range |
    |---|---|---|
    | Graph density (100-parcel) | — | 3–20% (det.), 20–60% (prob.) |
    | Mean clustering coefficient | — | 0.40–0.65 |
    | Interhemispheric edge ratio | — | 18–28% |
    | Characteristic path length | — | 2.0–3.5 |
    | Small-world sigma (σ > 1) | — | 1.5–4.0 |
    | Hub regions | — | mPFC, PCC, IPL, superior frontal |

    *Published ranges: van den Heuvel & Sporns (2011), Bassett & Bullmore (2006),
    Hagmann et al. (2008), Bullmore & Sporns (2009), Sporns et al. (2004).*

    ---

    ## 6. Conservative Behaviour Check

    Interface-seeded deterministic tractography is expected to show:

    - [ ] Lower density than probabilistic (iFOD2 + ACT): **expected 5–15%**
    - [ ] Higher mapping ratio than Phase 3.4 WM-seeded baseline
    - [ ] Shorter mean streamline lengths (endpoint near cortex → shorter path)
    - [ ] Similar or higher FA (interface seeds avoid deep WM noise)
    - [ ] Preserved major long-range connections (frontal–parietal, frontal–temporal)
    - [ ] Preserved bilateral symmetry

    ---

    ## 7. Limitations

    1. **Affine-only atlas registration** — same as Phase 3.4; nonlinear registration would improve parcel accuracy.
    2. **Interface seeding without 5-tissue-type (5TT) constraint** — some seeds may land in sulcal CSF or deep grey matter near the threshold boundary.
    3. **No streamline filtering** — SIFT/SIFT2 streamline reweighting (Smith et al. 2013) would improve biological accuracy.
    4. **b=1500 vs b=2500 cross-site** — FA differences between NYU and other sites persist.

    ---

    ## 8. Output Files

    | File | Description |
    |---|---|
    | `deterministic_connectome_summary.csv` | Per-subject Phase 3.5A connectome metrics |
    | `deterministic_site_summary.csv` | Per-site Phase 3.5A summary |
    | `deterministic_vs_baseline_comparison.csv` | Per-subject delta (3.5A − 3.4) |
    | `deterministic_qc_flags.csv` | Subjects flagged for review |
    | `PHASE_3_5A_REPORT.md` | This report |
    """)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Phase 3.5A report written: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gini(arr: np.ndarray) -> float:
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr.astype(float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def _top_hub_names(graph_props: list[dict], top_k: int = 5) -> str:
    from collections import Counter
    counter: Counter = Counter()
    for gp in graph_props:
        for h in gp.get("hub_by_degree", [])[:top_k]:
            counter[h["name"]] += 1
    if not counter:
        return "  (unavailable)"
    lines = [f"  {name}: {cnt}/{len(graph_props)} subjects" for name, cnt in counter.most_common(top_k)]
    return "\n".join(lines)
