"""
Phase 3.4 — Connectome QC

Batch runner and summary CSV writer for atlas-based connectome construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from neurofiber.connectome.atlas_registration import (
    AtlasInfo,
    RegisteredAtlas,
    register_atlas_to_subject,
    load_registered_atlas,
)
from neurofiber.connectome.connectome_builder import (
    ConnectomeRecord,
    build_subject_connectome,
)
from neurofiber.utils.logging import get_logger
from neurofiber.utils.safety import guard_no_raw_write

from neurofiber.tractography.streamline_generation import (
    CLEAN_DTI_SITES_DISPLAY,
    site_to_folder,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_connectome_batch(
    processed_root:  Path,
    raw_root:        Path,
    sites:           list[str],
    atlas_info:      AtlasInfo,
    skip_if_exists:  bool = True,
    registration_level_iters: list[int]   = None,
    registration_sigmas:      list[float] = None,
    registration_factors:     list[int]   = None,
) -> list[ConnectomeRecord]:
    """
    Run atlas registration + connectome construction for all subjects across
    sites. Subject failures do not stop the batch.
    """
    records: list[ConnectomeRecord] = []

    reg_kwargs = {}
    if registration_level_iters is not None:
        reg_kwargs["level_iters"] = registration_level_iters
    if registration_sigmas is not None:
        reg_kwargs["sigmas"] = registration_sigmas
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
            logger.warning("[%s] no dti_1 directories found", folder)
            continue

        logger.info("[%s] processing %d subjects …", folder, len(dti_dirs))

        for dti_dir in dti_dirs:
            subject_id = dti_dir.parent.parent.name
            session_dir = dti_dir.parent
            connectome_dir = session_dir / "connectome"
            atlas_dir = connectome_dir / "atlas"

            try:
                # Atlas registration (per-subject)
                atlas_nii = atlas_dir / "atlas_subject_space.nii.gz"
                if skip_if_exists and atlas_nii.exists():
                    reg_atlas = load_registered_atlas(atlas_dir, atlas_info.label_df)
                    logger.debug(
                        "[%s/%s] atlas already registered — loading from disk",
                        folder, subject_id,
                    )
                else:
                    t1w_path = raw_root / folder / dti_dir.parent.parent.parent.name / subject_id / "session_1" / "anat_1" / "anat.nii.gz"
                    fa_path  = dti_dir / "tensor" / "FA.nii.gz"

                    if not t1w_path.exists():
                        raise FileNotFoundError(f"T1w not found: {t1w_path}")
                    if not fa_path.exists():
                        raise FileNotFoundError(f"FA not found: {fa_path}")

                    logger.info("[%s/%s] registering atlas …", folder, subject_id)
                    reg_atlas = register_atlas_to_subject(
                        t1w_path=t1w_path,
                        fa_path=fa_path,
                        atlas_info=atlas_info,
                        output_dir=atlas_dir,
                        raw_root=None,  # atlas_dir is in processed — safe
                        **reg_kwargs,
                    )

                # Connectome construction
                rec = build_subject_connectome(
                    dti_dir=dti_dir,
                    registered_atlas=reg_atlas,
                    output_dir=connectome_dir,
                    raw_root=raw_root,
                    skip_if_exists=skip_if_exists,
                )
                records.append(rec)

            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s/%s] FAILED: %s", folder, subject_id, exc)
                from neurofiber.connectome.connectome_builder import ConnectomeRecord
                records.append(ConnectomeRecord(
                    site=folder, dataset=dti_dir.parent.parent.parent.name,
                    subject_id=subject_id,
                    node_count=0, edge_count=0, graph_density=0.0, mean_degree=0.0,
                    mean_edge_fa=None, mean_edge_md=None, mean_edge_length=None,
                    streamline_count_total=0, mapped_streamline_count=0,
                    unmapped_streamline_count=0, self_loop_count=0,
                    mapping_success_ratio=0.0,
                    status="failed", error_message=str(exc),
                ))

    return records


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------

def save_summary_csvs(
    records:    list[ConnectomeRecord],
    output_dir: Path,
    raw_root:   Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Write:
      phase3_4_connectome_summary.csv      — one row per subject
      phase3_4_site_connectome_summary.csv — one row per site

    Returns (subject_csv, site_csv).
    """
    if raw_root is not None:
        guard_no_raw_write(output_dir, raw_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.to_summary_row() for r in records])
    subj_csv = output_dir / "phase3_4_connectome_summary.csv"
    df.to_csv(subj_csv, index=False)

    ok = df[df["status"] == "success"]
    site_rows = []
    for site_name, grp in ok.groupby("site"):
        site_rows.append({
            "site":                      site_name,
            "subjects_processed":        len(grp),
            "subjects_failed":           len(df[(df["site"] == site_name) & (df["status"] == "failed")]),
            "mean_edge_count":           round(grp["edge_count"].mean(),            1),
            "mean_graph_density":        round(grp["graph_density"].mean(),         5),
            "mean_edge_fa":              round(grp["mean_edge_fa"].mean(),          4),
            "mean_edge_md":              round(grp["mean_edge_md"].mean(),          6),
            "mean_edge_length":          round(grp["mean_edge_length"].mean(),      2),
            "mean_mapping_success_ratio":round(grp["mapping_success_ratio"].mean(), 4),
            "subjects_review_required":  int(grp["review_required"].sum()),
            "notes": "",
        })
    site_df  = pd.DataFrame(site_rows)
    site_csv = output_dir / "phase3_4_site_connectome_summary.csv"
    site_df.to_csv(site_csv, index=False)

    logger.info(
        "Saved — subject=%s (%d rows)  site=%s (%d rows)",
        subj_csv.name, len(df), site_csv.name, len(site_df),
    )
    return subj_csv, site_csv
