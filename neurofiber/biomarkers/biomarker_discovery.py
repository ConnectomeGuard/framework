"""
NeuroFiber Phase 4.1 — ASD Biomarker Discovery

Statistical discovery of ASD-associated structural connectome edges using
ComBat-harmonized FA, MD, and length edge features.

Scientific context
──────────────────
This is a biomarker discovery and statistical association analysis.
It does NOT build a clinical diagnostic model.
It does NOT claim diagnosis.

Edge-level regression
─────────────────────
For every testable edge three linear models are fit:

  Model A:  y = β₀ + β₁·diagnosis + ε
  Model B:  y = β₀ + β₁·diagnosis + β₂·age + β₃·sex + Σβₖ·siteₖ + ε
  Model C:  y = (Model B) + β₄·mean_fa + β₅·density + ε

β₁ is the ASD-associated effect after covariate adjustment.

Stability criteria
──────────────────
A candidate biomarker must:
  - approach or survive FDR threshold (q ≤ 0.10 exploratory, q ≤ 0.05 strict)
  - show |Cohen's d| ≥ 0.2 (small effect or larger)
  - have consistent direction across ≥ 3 of 4 balanced sites
  - remain stable in leave-one-site-out analysis

NYU_2 handling
──────────────
NYU_2 is ASD-only. It contributes to global covariate-adjusted regression
(site effect absorbed), but is excluded from within-site effect-size
estimation and direction consistency scoring.

ComBat note
───────────
The canonical_v1/combat/ files were produced with NaN covariates and are
thus invalid. This module loads the unharmonized none/ data and re-applies
ComBat with correct covariates from metadata.csv at runtime.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("biomarker_discovery")

BALANCED_SITES = ["BNI", "NYU_1", "SDSU_1", "TCD_1"]  # sites with both ASD + CONTROL
N_ROIS = 100
N_EDGES = N_ROIS * (N_ROIS - 1) // 2  # 4950

MIN_N_VALID = 15           # minimum non-NaN subjects to test an edge
MIN_N_PER_GROUP = 5        # minimum subjects per diagnosis group
COHEN_D_THRESHOLD = 0.2    # minimum effect size for practical significance
FDR_STRICT = 0.05
FDR_EXPLORATORY = 0.10

# Schaefer 100 7-network approximate parcel boundaries (1-indexed, both hemispheres)
SCHAEFER_NETWORKS = {
    "Vis":        list(range(1, 8))  + list(range(51, 58)),
    "SomMot":     list(range(8, 20)) + list(range(58, 70)),
    "DorsAttn":   list(range(20, 28))+ list(range(70, 78)),
    "SalVentAttn":list(range(28, 37))+ list(range(78, 87)),
    "Limbic":     list(range(37, 41))+ list(range(87, 91)),
    "Cont":       list(range(41, 47))+ list(range(91, 97)),
    "Default":    list(range(47, 51))+ list(range(97, 101)),
}


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class BiomarkerDataset:
    """Aligned, ComBat-corrected feature matrix with metadata."""
    X_combat: np.ndarray          # (n_subjects, n_edges) — ComBat-corrected
    X_raw: np.ndarray             # (n_subjects, n_edges) — unharmonized
    meta: pd.DataFrame            # (n_subjects,) with columns: site/age/sex/diagnosis/mean_fa/density
    edge_cols: List[str]          # length n_edges — edge column names
    active_mask: np.ndarray       # boolean (n_edges,) — cross-site active edges
    feature_name: str             # "md_edges", "fa_edges", "length_edges"
    roi_labels: Optional[pd.DataFrame] = None   # (100, 2): index, name


@dataclass
class EdgeStats:
    """Per-edge statistics for one feature type."""
    edge_id: str
    roi_i: int
    roi_j: int
    roi_i_name: str
    roi_j_name: str
    network_i: str
    network_j: str
    is_cross_site_active: bool
    n_valid: int
    n_asd: int
    n_control: int
    # raw group stats
    mean_asd: float
    mean_control: float
    std_asd: float
    std_control: float
    cohen_d: float
    direction: str                # "ASD_higher" | "ASD_lower"
    # Model A
    beta_A: float
    p_A: float
    # Model B
    beta_B: float
    p_B: float
    se_B: float
    # Model C (may be NaN if covariates unavailable)
    beta_C: float
    p_C: float
    # FDR (computed post-hoc across all edges)
    q_B: float = np.nan
    q_B_exploratory: bool = False
    q_B_strict: bool = False


@dataclass
class SiteRobustness:
    """Per-site robustness metrics for a single edge."""
    edge_id: str
    site_effects: Dict[str, float]        # site → cohen_d (NaN if not computable)
    site_directions: Dict[str, str]       # site → "ASD_higher"|"ASD_lower"|"NA"
    n_same_direction: int                 # across BALANCED_SITES only
    n_sites_tested: int
    direction_consistent: bool            # True if ≥ 3/4 balanced sites agree
    # leave-one-site-out
    loso_betas: Dict[str, float]          # left-out-site → beta
    loso_pvals: Dict[str, float]          # left-out-site → p-value
    loso_n_same_direction: int
    loso_stable: bool                     # True if ≥ 4/5 LOSO runs same direction
    # NYU_2 dependence
    nyu2_removed_p: float                 # p-value when NYU_2 excluded
    nyu2_removed_same_dir: bool


@dataclass
class DiscoveryResult:
    """Full Phase 4.1 result for one feature type."""
    feature_name: str
    edge_stats: pd.DataFrame
    site_robustness: pd.DataFrame
    loso_table: pd.DataFrame
    candidate_biomarkers: pd.DataFrame
    permutation_null_min_p: Optional[np.ndarray]  # shape (n_perms,)
    permutation_observed_min_p: float
    n_tested: int
    n_q05: int
    n_q10: int
    top10: pd.DataFrame


# ── edge-index utilities ──────────────────────────────────────────────────────

def edge_index_to_rois(edge_idx: int, n_rois: int = N_ROIS) -> Tuple[int, int]:
    """Convert 0-based edge index to (roi_i, roi_j) 1-based ROI labels."""
    k = 0
    for i in range(1, n_rois + 1):
        for j in range(i + 1, n_rois + 1):
            if k == edge_idx:
                return i, j
            k += 1
    raise ValueError(f"edge_idx {edge_idx} out of range for {n_rois} ROIs")


def build_edge_index_table(n_rois: int = N_ROIS) -> pd.DataFrame:
    """Pre-compute (edge_idx → roi_i, roi_j) for all edges."""
    rows = []
    k = 0
    for i in range(1, n_rois + 1):
        for j in range(i + 1, n_rois + 1):
            rows.append({"edge_idx": k, "roi_i": i, "roi_j": j})
            k += 1
    return pd.DataFrame(rows)


def roi_to_network(roi: int) -> str:
    """Map a 1-based Schaefer ROI index to its 7-network name."""
    for net, parcels in SCHAEFER_NETWORKS.items():
        if roi in parcels:
            return net
    return "Unknown"


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset(
    repo_root: Path,
    feature_name: str,
    min_per_site: int = 1,
) -> BiomarkerDataset:
    """
    Load unharmonized edge data, apply ComBat with correct covariates,
    and return an aligned BiomarkerDataset.

    The canonical_v1/combat files contain an error (NaN covariates produced
    all-NaN active edges). This function rebuilds the harmonization correctly.
    """
    from neurofiber.harmonization.combat_evaluation import apply_combat, active_edge_mask

    none_path = repo_root / "data" / "processed_v2b" / "harmonized_connectomes" / "none" / f"{feature_name}.csv"
    meta_path = repo_root / "data" / "canonical_v1" / "metadata.csv"
    label_path = (
        repo_root
        / "data/processed/abide_ii/sdsu/ABIDEII-SDSU_1/28853/session_1/connectome/atlas/atlas_labels.csv"
    )

    if not none_path.exists():
        raise FileNotFoundError(f"Unharmonized edge data not found: {none_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    raw_df = pd.read_csv(none_path, index_col=0)
    meta_df = pd.read_csv(meta_path).set_index("subject_id")

    # Align metadata to edge file row order
    meta_aligned = meta_df.loc[raw_df.index].copy()

    edge_cols = [c for c in raw_df.columns if c.startswith("edge_")]
    X_raw = raw_df[edge_cols].values.astype(np.float64)

    meta_rows = [
        {
            "site": str(r["site"]),
            "age": float(r["age"]),
            "sex": str(r["sex"]),
            "diagnosis": str(r["diagnosis"]),
        }
        for _, r in meta_aligned.iterrows()
    ]

    log.info("Applying ComBat to %s (%d edges, %d subjects)…", feature_name, len(edge_cols), len(meta_rows))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_combat, active, info = apply_combat(X_raw, meta_rows, min_per_site=min_per_site)

    log.info("  Active (cross-site) edges: %d / %d", info["n_active_edges"], info["n_total_edges"])
    log.info("  ComBat non-NaN: %d", int((~np.isnan(X_combat[:, active])).sum()))

    roi_labels = None
    if label_path.exists():
        roi_labels = pd.read_csv(label_path)

    return BiomarkerDataset(
        X_combat=X_combat,
        X_raw=X_raw,
        meta=meta_aligned.reset_index(),
        edge_cols=edge_cols,
        active_mask=active,
        feature_name=feature_name,
        roi_labels=roi_labels,
    )


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d: (mean_a - mean_b) / pooled_std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


# ── regression ────────────────────────────────────────────────────────────────

def _fit_model(
    y: np.ndarray,
    X_design: np.ndarray,
) -> Tuple[float, float, float]:
    """OLS: return (beta_dx, p_dx, se_dx) for the diagnosis coefficient (column 1)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X_design).fit()
        return float(res.params[1]), float(res.pvalues[1]), float(res.bse[1])
    except Exception:
        return np.nan, np.nan, np.nan


def _site_dummies(sites: np.ndarray, reference: str = "BNI") -> np.ndarray:
    """One-hot encode sites with `reference` as the dropped category."""
    site_list = sorted(s for s in set(sites) if s != reference)
    if not site_list:
        return np.zeros((len(sites), 0))
    return np.column_stack([(sites == s).astype(float) for s in site_list])


def compute_edge_stats(
    ds: BiomarkerDataset,
    edge_index_table: pd.DataFrame,
    use_combat: bool = True,
    min_n_valid: int = MIN_N_VALID,
    min_n_per_group: int = MIN_N_PER_GROUP,
) -> List[EdgeStats]:
    """
    Fit Models A, B, C for every testable edge.

    Parameters
    ----------
    ds            : BiomarkerDataset
    edge_index_table : pre-computed ROI mapping
    use_combat    : if True, use ComBat-corrected values; else raw
    min_n_valid   : skip edge if fewer non-NaN subjects
    min_n_per_group : skip edge if any diagnosis group has too few subjects
    """
    X = ds.X_combat if use_combat else ds.X_raw
    meta = ds.meta
    sites = meta["site"].values
    ages  = meta["age"].values.astype(float)
    sexes = (meta["sex"].values == "M").astype(float)
    dx    = (meta["diagnosis"].values == "ASD").astype(float)
    mean_fa  = meta["mean_fa"].values.astype(float)
    density  = meta["density"].values.astype(float)

    # site dummies (reference=BNI)
    site_dum = _site_dummies(sites, reference="BNI")
    # age / sex standardized
    age_c = (ages - np.nanmean(ages)) / (np.nanstd(ages) + 1e-9)
    mfa_c = (mean_fa - np.nanmean(mean_fa)) / (np.nanstd(mean_fa) + 1e-9)
    den_c = (density - np.nanmean(density)) / (np.nanstd(density) + 1e-9)

    roi_map = dict(zip(edge_index_table["edge_idx"], zip(edge_index_table["roi_i"], edge_index_table["roi_j"])))
    edge_name_to_idx = {ec: int(ec.split("_")[1]) - 1 for ec in ds.edge_cols}

    label_map: Dict[int, str] = {}
    if ds.roi_labels is not None:
        label_map = dict(zip(ds.roi_labels["index"], ds.roi_labels["name"]))

    results = []
    for col_pos, ec in enumerate(ds.edge_cols):
        edge_idx = edge_name_to_idx[ec]
        roi_i, roi_j = roi_map.get(edge_idx, (0, 0))

        y_full = X[:, col_pos]
        valid = ~np.isnan(y_full)
        n_valid = int(valid.sum())

        if n_valid < min_n_valid:
            continue

        y = y_full[valid]
        dx_v = dx[valid]; age_v = age_c[valid]; sex_v = sexes[valid]
        sd_v = site_dum[valid]; mfa_v = mfa_c[valid]; den_v = den_c[valid]

        asd_mask   = dx_v == 1
        ctrl_mask  = dx_v == 0
        n_asd      = int(asd_mask.sum())
        n_control  = int(ctrl_mask.sum())

        if n_asd < min_n_per_group or n_control < min_n_per_group:
            continue

        y_asd, y_ctrl = y[asd_mask], y[ctrl_mask]
        m_asd  = float(y_asd.mean())
        m_ctrl = float(y_ctrl.mean())
        s_asd  = float(y_asd.std(ddof=1)) if len(y_asd) > 1 else np.nan
        s_ctrl = float(y_ctrl.std(ddof=1)) if len(y_ctrl) > 1 else np.nan
        d      = cohen_d(y_asd, y_ctrl)
        direction = "ASD_higher" if m_asd >= m_ctrl else "ASD_lower"

        # Model A: diagnosis only
        XA = sm.add_constant(dx_v.reshape(-1, 1))
        betaA, pA, _ = _fit_model(y, XA)

        # Model B: + age + sex + site
        XB = sm.add_constant(np.column_stack([dx_v, age_v, sex_v, sd_v]))
        betaB, pB, seB = _fit_model(y, XB)

        # Model C: + mean_fa + density
        XC = sm.add_constant(np.column_stack([dx_v, age_v, sex_v, sd_v, mfa_v, den_v]))
        betaC, pC, _ = _fit_model(y, XC)

        results.append(EdgeStats(
            edge_id=ec,
            roi_i=roi_i, roi_j=roi_j,
            roi_i_name=label_map.get(roi_i, f"ROI_{roi_i}"),
            roi_j_name=label_map.get(roi_j, f"ROI_{roi_j}"),
            network_i=roi_to_network(roi_i),
            network_j=roi_to_network(roi_j),
            is_cross_site_active=bool(ds.active_mask[col_pos]),
            n_valid=n_valid, n_asd=n_asd, n_control=n_control,
            mean_asd=m_asd, mean_control=m_ctrl,
            std_asd=s_asd, std_control=s_ctrl,
            cohen_d=d, direction=direction,
            beta_A=betaA, p_A=pA,
            beta_B=betaB, p_B=pB, se_B=seB,
            beta_C=betaC, p_C=pC,
        ))

    return results


def apply_fdr(stats_list: List[EdgeStats]) -> List[EdgeStats]:
    """Benjamini-Hochberg FDR correction across all tested edges (Model B p-values)."""
    pvals = np.array([s.p_B for s in stats_list])
    valid = ~np.isnan(pvals)
    q = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        _, q_vals, _, _ = multipletests(pvals[valid], alpha=FDR_STRICT, method="fdr_bh")
        q[valid] = q_vals
    for i, s in enumerate(stats_list):
        s.q_B = float(q[i])
        s.q_B_strict       = bool(q[i] <= FDR_STRICT)
        s.q_B_exploratory  = bool(q[i] <= FDR_EXPLORATORY)
    return stats_list


# ── site robustness ───────────────────────────────────────────────────────────

def _within_site_cohen_d(
    y_full: np.ndarray,
    dx_full: np.ndarray,
    sites_full: np.ndarray,
    site: str,
) -> Tuple[float, str]:
    """Cohen's d for one site; returns (d, direction) or (nan, 'NA')."""
    mask = (sites_full == site) & ~np.isnan(y_full)
    y_s  = y_full[mask]
    dx_s = dx_full[mask]
    asd_vals  = y_s[dx_s == 1]
    ctrl_vals = y_s[dx_s == 0]
    if len(asd_vals) < 2 or len(ctrl_vals) < 2:
        return np.nan, "NA"
    d = cohen_d(asd_vals, ctrl_vals)
    direction = "ASD_higher" if asd_vals.mean() >= ctrl_vals.mean() else "ASD_lower"
    return d, direction


def _loso_regression(
    y_full: np.ndarray,
    dx_full: np.ndarray,
    ages_c: np.ndarray,
    sexes: np.ndarray,
    site_dum: np.ndarray,
    sites_full: np.ndarray,
    leave_out: str,
) -> Tuple[float, float]:
    """Re-run Model B leaving out one site. Returns (beta, p_value)."""
    mask = (sites_full != leave_out) & ~np.isnan(y_full)
    if mask.sum() < MIN_N_VALID:
        return np.nan, np.nan
    y  = y_full[mask]
    dx = dx_full[mask]
    if (dx == 1).sum() < MIN_N_PER_GROUP or (dx == 0).sum() < MIN_N_PER_GROUP:
        return np.nan, np.nan
    sd = _site_dummies(sites_full[mask], reference="BNI")
    X  = sm.add_constant(np.column_stack([dx, ages_c[mask], sexes[mask], sd]))
    beta, p, _ = _fit_model(y, X)
    return beta, p


def compute_site_robustness(
    ds: BiomarkerDataset,
    stats_list: List[EdgeStats],
    edge_index_table: pd.DataFrame,
    use_combat: bool = True,
) -> List[SiteRobustness]:
    """
    For each edge in stats_list, compute:
      - within-site Cohen's d for each balanced site
      - direction consistency across balanced sites
      - leave-one-site-out beta and p-value
      - NYU_2 dependence check
    """
    X = ds.X_combat if use_combat else ds.X_raw
    meta = ds.meta
    sites = meta["site"].values
    dx    = (meta["diagnosis"].values == "ASD").astype(float)
    ages  = meta["age"].values.astype(float)
    sexes = (meta["sex"].values == "M").astype(float)
    age_c = (ages - np.nanmean(ages)) / (np.nanstd(ages) + 1e-9)

    all_sites = sorted(set(sites))
    edge_name_to_pos = {ec: pos for pos, ec in enumerate(ds.edge_cols)}

    robustness = []
    for es in stats_list:
        pos = edge_name_to_pos[es.edge_id]
        y_full = X[:, pos]

        # within-site effects (balanced sites only for consistency scoring)
        site_effects: Dict[str, float] = {}
        site_directions: Dict[str, str] = {}
        for site in all_sites:
            d, direction = _within_site_cohen_d(y_full, dx, sites, site)
            site_effects[site] = d
            site_directions[site] = direction

        # direction consistency — balanced sites only
        global_dir = es.direction
        n_same = sum(
            1 for s in BALANCED_SITES
            if site_directions.get(s, "NA") == global_dir
        )
        n_tested = sum(1 for s in BALANCED_SITES if site_directions.get(s, "NA") != "NA")
        direction_consistent = (n_tested > 0) and (n_same >= max(1, int(np.ceil(n_tested * 0.75))))

        # leave-one-site-out
        site_dum = _site_dummies(sites, reference="BNI")
        loso_betas: Dict[str, float] = {}
        loso_pvals: Dict[str, float] = {}
        for leave_out in all_sites:
            b, p = _loso_regression(y_full, dx, age_c, sexes, site_dum, sites, leave_out)
            loso_betas[leave_out] = b
            loso_pvals[leave_out] = p

        global_beta_sign = np.sign(es.beta_B) if not np.isnan(es.beta_B) else 0
        loso_same = sum(
            1 for b in loso_betas.values()
            if not np.isnan(b) and np.sign(b) == global_beta_sign and global_beta_sign != 0
        )
        loso_stable = loso_same >= 4  # ≥4 of 5 LOSO runs agree

        # NYU_2 dependence: rerun excluding NYU_2
        b_nyu2, p_nyu2 = _loso_regression(y_full, dx, age_c, sexes, site_dum, sites, "NYU_2")
        nyu2_same_dir = (
            not np.isnan(b_nyu2) and
            global_beta_sign != 0 and
            np.sign(b_nyu2) == global_beta_sign
        )

        robustness.append(SiteRobustness(
            edge_id=es.edge_id,
            site_effects=site_effects,
            site_directions=site_directions,
            n_same_direction=n_same,
            n_sites_tested=n_tested,
            direction_consistent=direction_consistent,
            loso_betas=loso_betas,
            loso_pvals=loso_pvals,
            loso_n_same_direction=loso_same,
            loso_stable=loso_stable,
            nyu2_removed_p=float(p_nyu2) if not np.isnan(p_nyu2) else np.nan,
            nyu2_removed_same_dir=nyu2_same_dir,
        ))

    return robustness


# ── stability scoring ─────────────────────────────────────────────────────────

def stability_score(
    es: EdgeStats,
    rob: SiteRobustness,
    n_total_subjects: int = 229,
) -> float:
    """
    Composite stability score in [0, 1].

    Components (equal weight):
      1. FDR significance:  1.0 if q≤0.05, 0.5 if q≤0.10, 0 otherwise
      2. Effect magnitude:  min(|Cohen's d| / 0.5, 1.0)  [d=0.5 → full score]
      3. Direction consistency: n_same_direction / 4 balanced sites
      4. LOSO stability:    loso_n_same_direction / 5 runs
      5. Coverage:          n_valid / n_total_subjects
    """
    # 1. FDR
    if not np.isnan(es.q_B) and es.q_B <= FDR_STRICT:
        sig = 1.0
    elif not np.isnan(es.q_B) and es.q_B <= FDR_EXPLORATORY:
        sig = 0.5
    else:
        sig = 0.0

    # 2. Effect size
    eff = min(abs(es.cohen_d) / 0.5, 1.0) if not np.isnan(es.cohen_d) else 0.0

    # 3. Direction consistency (balanced sites)
    dir_score = rob.n_same_direction / max(rob.n_sites_tested, 1)

    # 4. LOSO
    loso_score = rob.loso_n_same_direction / 5.0

    # 5. Coverage
    cov = es.n_valid / n_total_subjects

    return float((sig + eff + dir_score + loso_score + cov) / 5.0)


# ── permutation test ──────────────────────────────────────────────────────────

def permutation_test(
    ds: BiomarkerDataset,
    n_perms: int = 100,
    use_combat: bool = True,
    min_n_valid: int = MIN_N_VALID,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Shuffle diagnosis labels N times and compute the minimum p-value across
    all testable active edges under Model B.

    Returns
    -------
    null_min_p  : (n_perms,) null distribution of minimum p-values
    observed_min_p : minimum Model B p-value from real data (active edges only)
    """
    rng = np.random.default_rng(seed)
    X = ds.X_combat if use_combat else ds.X_raw
    meta = ds.meta
    sites = meta["site"].values
    ages  = meta["age"].values.astype(float)
    sexes = (meta["sex"].values == "M").astype(float)
    dx    = (meta["diagnosis"].values == "ASD").astype(float)
    age_c = (ages - np.nanmean(ages)) / (np.nanstd(ages) + 1e-9)
    site_dum = _site_dummies(sites, reference="BNI")

    # Restrict to active edges with sufficient data
    active_positions = [
        pos for pos, active in enumerate(ds.active_mask) if active
    ]
    testable = []
    for pos in active_positions:
        y = X[:, pos]
        valid = ~np.isnan(y)
        if valid.sum() < min_n_valid:
            continue
        dx_v = dx[valid]
        if (dx_v == 1).sum() < MIN_N_PER_GROUP or (dx_v == 0).sum() < MIN_N_PER_GROUP:
            continue
        testable.append(pos)

    if not testable:
        log.warning("No testable active edges for permutation test.")
        return np.full(n_perms, np.nan), np.nan

    def _min_p(dx_perm: np.ndarray) -> float:
        min_p = 1.0
        for pos in testable:
            y_full = X[:, pos]
            valid = ~np.isnan(y_full)
            y = y_full[valid]
            sd = _site_dummies(sites[valid], reference="BNI")
            Xm = sm.add_constant(np.column_stack([
                dx_perm[valid], age_c[valid], sexes[valid], sd
            ]))
            _, p, _ = _fit_model(y, Xm)
            if not np.isnan(p) and p < min_p:
                min_p = p
        return min_p

    # Observed minimum p-value
    observed = _min_p(dx)

    # Permutation null
    null = np.zeros(n_perms)
    for i in range(n_perms):
        dx_perm = rng.permutation(dx)
        null[i] = _min_p(dx_perm)
        if (i + 1) % 10 == 0:
            log.info("  Permutation %d/%d …", i + 1, n_perms)

    return null, observed


# ── network-level summary ─────────────────────────────────────────────────────

def network_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count significant edges per network pair (Model B, q≤0.10).
    """
    df = stats_df[stats_df["q_B_exploratory"] == True].copy()
    if df.empty:
        return pd.DataFrame(columns=["network_i", "network_j", "n_edges", "mean_cohen_d"])
    rows = []
    for (ni, nj), grp in df.groupby(["network_i", "network_j"]):
        rows.append({
            "network_i": ni,
            "network_j": nj,
            "n_edges": len(grp),
            "mean_cohen_d": round(float(grp["cohen_d"].mean()), 4),
            "n_asd_higher": int((grp["direction"] == "ASD_higher").sum()),
            "n_asd_lower":  int((grp["direction"] == "ASD_lower").sum()),
        })
    return pd.DataFrame(rows).sort_values("n_edges", ascending=False).reset_index(drop=True)


# ── main entry point ──────────────────────────────────────────────────────────

def run_biomarker_discovery(
    repo_root: Path,
    feature_names: Optional[List[str]] = None,
    n_perms: int = 100,
    use_combat: bool = True,
    min_n_valid: int = MIN_N_VALID,
) -> Dict[str, DiscoveryResult]:
    """
    Run the full Phase 4.1 biomarker discovery pipeline for all feature types.

    Returns a dict mapping feature_name → DiscoveryResult.
    """
    if feature_names is None:
        feature_names = ["md_edges", "fa_edges", "length_edges"]

    edge_index_table = build_edge_index_table()
    results: Dict[str, DiscoveryResult] = {}

    for fn in feature_names:
        log.info("=" * 60)
        log.info("Feature: %s", fn)
        log.info("=" * 60)

        ds = load_dataset(repo_root, fn)
        n_subjects = len(ds.meta)

        log.info("Computing per-edge statistics …")
        stats_list = compute_edge_stats(
            ds, edge_index_table,
            use_combat=use_combat,
            min_n_valid=min_n_valid,
        )
        log.info("  Tested edges: %d", len(stats_list))

        stats_list = apply_fdr(stats_list)

        n_q05 = sum(1 for s in stats_list if s.q_B_strict)
        n_q10 = sum(1 for s in stats_list if s.q_B_exploratory)
        log.info("  q≤0.05: %d   q≤0.10: %d", n_q05, n_q10)

        log.info("Computing site robustness …")
        robustness_list = compute_site_robustness(ds, stats_list, edge_index_table, use_combat=use_combat)

        # Build robustness lookup
        rob_map = {r.edge_id: r for r in robustness_list}

        # Add stability scores
        scores = []
        for es in stats_list:
            rob = rob_map.get(es.edge_id)
            if rob is not None:
                scores.append(stability_score(es, rob, n_subjects))
            else:
                scores.append(0.0)

        # Convert to DataFrames
        stats_df = _stats_to_df(stats_list, scores)
        rob_df   = _robustness_to_df(robustness_list)
        loso_df  = _loso_to_df(robustness_list)

        # Candidate biomarkers: q_B ≤ 0.10 OR stability_score ≥ 0.5
        candidate_mask = stats_df["q_B_exploratory"] | (stats_df["stability_score"] >= 0.50)
        candidates = stats_df[candidate_mask].copy().sort_values("stability_score", ascending=False)

        # Mark NYU_2-dependent candidates
        candidates["nyu2_stable"] = candidates["edge_id"].map(
            lambda eid: rob_map[eid].nyu2_removed_same_dir if eid in rob_map else True
        )

        log.info("Running permutation test (%d permutations) …", n_perms)
        null_min_p, obs_min_p = permutation_test(
            ds, n_perms=n_perms, use_combat=use_combat, min_n_valid=min_n_valid
        )

        if not np.isnan(obs_min_p):
            perm_pvalue = float((null_min_p <= obs_min_p).mean())
            log.info("  Observed min-p: %.2e   Permutation p-value: %.3f", obs_min_p, perm_pvalue)
        else:
            log.info("  Observed min-p: NA (no testable active edges)")

        top10 = stats_df.nlargest(10, "stability_score")[
            ["edge_id", "roi_i_name", "roi_j_name", "network_i", "network_j",
             "cohen_d", "p_B", "q_B", "direction", "n_valid", "stability_score",
             "is_cross_site_active"]
        ]

        results[fn] = DiscoveryResult(
            feature_name=fn,
            edge_stats=stats_df,
            site_robustness=rob_df,
            loso_table=loso_df,
            candidate_biomarkers=candidates,
            permutation_null_min_p=null_min_p,
            permutation_observed_min_p=obs_min_p,
            n_tested=len(stats_list),
            n_q05=n_q05,
            n_q10=n_q10,
            top10=top10,
        )

    return results


# ── DataFrame conversion helpers ──────────────────────────────────────────────

def _stats_to_df(stats_list: List[EdgeStats], scores: List[float]) -> pd.DataFrame:
    rows = []
    for es, sc in zip(stats_list, scores):
        rows.append({
            "edge_id":              es.edge_id,
            "roi_i":                es.roi_i,
            "roi_j":                es.roi_j,
            "roi_i_name":           es.roi_i_name,
            "roi_j_name":           es.roi_j_name,
            "network_i":            es.network_i,
            "network_j":            es.network_j,
            "is_cross_site_active": es.is_cross_site_active,
            "n_valid":              es.n_valid,
            "n_asd":                es.n_asd,
            "n_control":            es.n_control,
            "mean_asd":             es.mean_asd,
            "mean_control":         es.mean_control,
            "std_asd":              es.std_asd,
            "std_control":          es.std_control,
            "cohen_d":              es.cohen_d,
            "direction":            es.direction,
            "beta_A":               es.beta_A,
            "p_A":                  es.p_A,
            "beta_B":               es.beta_B,
            "p_B":                  es.p_B,
            "se_B":                 es.se_B,
            "beta_C":               es.beta_C,
            "p_C":                  es.p_C,
            "q_B":                  es.q_B,
            "q_B_exploratory":      es.q_B_exploratory,
            "q_B_strict":           es.q_B_strict,
            "stability_score":      round(sc, 4),
        })
    return pd.DataFrame(rows)


def _robustness_to_df(robustness_list: List[SiteRobustness]) -> pd.DataFrame:
    rows = []
    for rob in robustness_list:
        row = {
            "edge_id":              rob.edge_id,
            "n_same_direction":     rob.n_same_direction,
            "n_sites_tested":       rob.n_sites_tested,
            "direction_consistent": rob.direction_consistent,
            "loso_n_same_direction": rob.loso_n_same_direction,
            "loso_stable":          rob.loso_stable,
            "nyu2_removed_p":       rob.nyu2_removed_p,
            "nyu2_removed_same_dir": rob.nyu2_removed_same_dir,
        }
        for site in BALANCED_SITES + ["NYU_2"]:
            row[f"cohen_d_{site}"] = rob.site_effects.get(site, np.nan)
            row[f"direction_{site}"] = rob.site_directions.get(site, "NA")
        rows.append(row)
    return pd.DataFrame(rows)


def _loso_to_df(robustness_list: List[SiteRobustness]) -> pd.DataFrame:
    rows = []
    all_sites = BALANCED_SITES + ["NYU_2"]
    for rob in robustness_list:
        for site in all_sites:
            rows.append({
                "edge_id":    rob.edge_id,
                "left_out":   site,
                "beta":       rob.loso_betas.get(site, np.nan),
                "p_value":    rob.loso_pvals.get(site, np.nan),
            })
    return pd.DataFrame(rows)
