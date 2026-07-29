"""
Shared helpers for ABIDE II site exploration scripts (Phase 1.3+).

Provides: join, CSV builders, site stats extractor, manifest builder, printer.
All functions are pure (no I/O) so they are easy to test.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neurofiber.registry.eth_scanner import SubjectScanResult
from neurofiber.registry.metadata_loader import SubjectMetadata
from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def join_scan_meta(
    scans: list[SubjectScanResult],
    meta: dict[str, SubjectMetadata],
) -> list[dict]:
    rows = []
    for s in scans:
        row = s.to_dict()
        m = meta.get(s.subject_id)
        if m:
            row.update({"diagnosis": m.diagnosis, "age": m.age,
                        "sex": m.sex, "fiq": m.fiq})
        else:
            row.update({"diagnosis": None, "age": None, "sex": None, "fiq": None})
            logger.warning("No metadata match for subject %s", s.subject_id)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def make_validation_summary_df(scans: list[SubjectScanResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "subject_id":         s.subject_id,
        "validation_status":  s.validation_status,
        "anat_exists":        s.anat_exists,
        "dti_exists":         s.dti_exists,
        "rest_exists":        s.rest_exists,
        "missing_modalities": s.missing_modalities,
    } for s in scans])


def make_dti_diagnostic_df(scans: list[SubjectScanResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "subject_id":              s.subject_id,
        "dti_exists":              s.dti_exists,
        "dti_volume_count":        s.dti_volume_count,
        "bval_count":              s.bval_count,
        "bvec_count":              s.bvec_count,
        "unique_bvalues":          s.unique_bvalues,
        "n_b0_volumes":            s.n_b0_volumes,
        "n_dwi_volumes":           s.n_dwi_volumes,
        "dti_consistent":          s.dti_consistent,
        "dti_mismatch_type":       s.dti_mismatch_type,
        "b0_mean_signal":          s.b0_mean_signal,
        "last_volume_mean_signal": s.last_volume_mean_signal,
        "signal_ratio":            s.signal_ratio,
        "recommended_action":      s.recommended_action,
    } for s in scans])


# ---------------------------------------------------------------------------
# Site manifest
# ---------------------------------------------------------------------------

def make_site_manifest(
    site: str,
    dataset: str,
    scans: list[SubjectScanResult],
    meta: dict[str, SubjectMetadata],
) -> dict:
    meta_list = list(meta.values())
    ages     = [m.age for m in meta_list if m.age is not None]
    dx       = Counter(m.diagnosis for m in meta_list)
    sex      = Counter(m.sex for m in meta_list)
    dti_subs = [s for s in scans if s.dti_exists]

    return {
        "site":          site,
        "dataset":       dataset,
        "subject_count": len(scans),
        "demographics": {
            "asd_count":     dx.get("ASD", 0),
            "control_count": dx.get("CONTROL", 0),
            "male_count":    sex.get("M", 0),
            "female_count":  sex.get("F", 0),
            "age_min":   round(float(min(ages)), 2) if ages else None,
            "age_max":   round(float(max(ages)), 2) if ages else None,
            "age_mean":  round(float(np.mean(ages)), 2) if ages else None,
        },
        "modality_presence": {
            m: sum(1 for s in scans if getattr(s, f"{m}_exists"))
            for m in ("anat", "dti", "bval", "bvec", "rest")
        },
        "validation_counts": {
            st: sum(1 for s in scans if s.validation_status == st)
            for st in sorted({s.validation_status for s in scans})
        },
        "dti_info": _dti_info(dti_subs),
        "imaging_profiles": {
            "anat_shapes": dict(Counter(s.anat_shape for s in scans if s.anat_shape)),
            "rest_shapes": dict(Counter(s.rest_shape for s in scans if s.rest_shape)),
        },
    }


def _dti_info(dti_subs: list[SubjectScanResult]) -> dict:
    if not dti_subs:
        return {"dti_subjects": 0, "consistent_count": 0, "mismatch_count": 0,
                "common_volume_count": None, "common_bval_count": None,
                "unique_bvalues": None}
    unique_bv: set[int] = set()
    for s in dti_subs:
        if s.unique_bvalues:
            try:
                unique_bv.update(eval(s.unique_bvalues))  # noqa: S307
            except Exception:
                pass
    return {
        "dti_subjects":         len(dti_subs),
        "consistent_count":     sum(1 for s in dti_subs if s.dti_consistent),
        "mismatch_count":       sum(1 for s in dti_subs if s.dti_consistent is False),
        "common_volume_count":  Counter(s.dti_volume_count for s in dti_subs).most_common(1)[0][0],
        "common_bval_count":    Counter(s.bval_count for s in dti_subs if s.bval_count).most_common(1)[0][0]
                                if any(s.bval_count for s in dti_subs) else None,
        "unique_bvalues":       sorted(unique_bv),
    }


# ---------------------------------------------------------------------------
# Site stats (for comparison table)
# ---------------------------------------------------------------------------

def site_stats_from_scan(
    site: str,
    scans: list[SubjectScanResult],
    meta: dict[str, SubjectMetadata],
) -> dict:
    meta_l = list(meta.values())
    ages   = [m.age for m in meta_l if m.age is not None]
    dx     = Counter(m.diagnosis for m in meta_l)
    sex    = Counter(m.sex for m in meta_l)
    dti_s  = [s for s in scans if s.dti_exists]
    vol_c  = Counter(s.dti_volume_count for s in dti_s if s.dti_volume_count)
    bvc    = Counter(s.bval_count for s in dti_s if s.bval_count)

    unique_bv: set[int] = set()
    for s in dti_s:
        if s.unique_bvalues:
            try:
                unique_bv.update(eval(s.unique_bvalues))  # noqa: S307
            except Exception:
                pass

    anat_shapes = Counter(s.anat_shape for s in scans if s.anat_shape)
    rest_shapes  = Counter(s.rest_shape  for s in scans if s.rest_shape)

    return {
        "site":                 site,
        "subject_count":        len(scans),
        "asd_count":            dx.get("ASD", 0),
        "control_count":        dx.get("CONTROL", 0),
        "age_min":              round(float(min(ages)), 2) if ages else None,
        "age_max":              round(float(max(ages)), 2) if ages else None,
        "age_mean":             round(float(np.mean(ages)), 2) if ages else None,
        "male_count":           sex.get("M", 0),
        "female_count":         sex.get("F", 0),
        "all_modalities_count": sum(1 for s in scans if s.anat_exists and s.rest_exists),
        "valid_dti_count":      sum(1 for s in dti_s if s.dti_consistent is True),
        "dti_mismatch_count":   sum(1 for s in dti_s if s.dti_consistent is False),
        "missing_modality_count": sum(1 for s in scans if s.validation_status == "missing_files"),
        "load_failure_count":   sum(1 for s in scans if s.validation_status == "dti_failed"),
        "common_volume_count":  vol_c.most_common(1)[0][0] if vol_c else None,
        "common_bval_count":    bvc.most_common(1)[0][0] if bvc else None,
        "unique_b_values":      str(sorted(unique_bv)) if unique_bv else None,
        "common_voxel_size":    anat_shapes.most_common(1)[0][0] if anat_shapes else None,
    }


def site_stats_from_index_csv(
    site: str,
    index_csv: Path,
    meta_csv: Optional[Path] = None,
    site_label: Optional[str] = None,
) -> dict:
    """Load site stats from a saved dataset_index CSV + optional metadata CSV."""
    df = pd.read_csv(index_csv)

    # demographics — prefer metadata CSV if provided
    if meta_csv and meta_csv.exists():
        from neurofiber.registry.metadata_loader import load_abide2_metadata
        meta = load_abide2_metadata(meta_csv, site=site_label or site)
        meta_l = list(meta.values())
        ages   = [m.age for m in meta_l if m.age is not None]
        dx     = Counter(m.diagnosis for m in meta_l)
        sex    = Counter(m.sex for m in meta_l)
    else:
        ages = df["age"].dropna().tolist() if "age" in df.columns else []
        dx   = Counter(df["diagnosis"].dropna().tolist()) if "diagnosis" in df.columns else Counter()
        sex  = Counter(df["sex"].dropna().tolist()) if "sex" in df.columns else Counter()

    dti_col = "dti_volume_count" in df.columns
    bval_col = "bval_count" in df.columns

    return {
        "site":                 site,
        "subject_count":        len(df),
        "asd_count":            dx.get("ASD", 0),
        "control_count":        dx.get("CONTROL", 0),
        "age_min":              round(float(min(ages)), 2) if ages else None,
        "age_max":              round(float(max(ages)), 2) if ages else None,
        "age_mean":             round(float(np.mean(ages)), 2) if ages else None,
        "male_count":           sex.get("M", 0),
        "female_count":         sex.get("F", 0),
        "all_modalities_count": int((df["anat_exists"] & df["rest_exists"]).sum()) if "anat_exists" in df.columns else len(df),
        "valid_dti_count":      int((df["validation_status"] == "valid").sum() & df.get("dti_exists", pd.Series([False]*len(df))).any()),
        "dti_mismatch_count":   int((df["validation_status"] == "dti_mismatch").sum()),
        "missing_modality_count": int((df["validation_status"] == "missing_files").sum()),
        "load_failure_count":   int((df["validation_status"] == "dti_failed").sum()),
        "common_volume_count":  int(df["dti_volume_count"].mode()[0]) if dti_col and df["dti_volume_count"].notna().any() else None,
        "common_bval_count":    int(df["bval_count"].mode()[0]) if bval_col and df["bval_count"].notna().any() else None,
        "unique_b_values":      None,
        "common_voxel_size":    None,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_site_summary(
    site: str,
    scans: list[SubjectScanResult],
    meta: dict[str, SubjectMetadata],
) -> None:
    meta_l  = list(meta.values())
    ages    = [m.age for m in meta_l if m.age is not None]
    dx      = Counter(m.diagnosis for m in meta_l)
    sex     = Counter(m.sex for m in meta_l)
    fiq_ok  = sum(1 for m in meta_l if m.fiq is not None)
    dti_s   = [s for s in scans if s.dti_exists]

    all_bvals: set[int] = set()
    for s in dti_s:
        if s.unique_bvalues:
            try:
                all_bvals.update(eval(s.unique_bvalues))  # noqa: S307
            except Exception:
                pass

    anat_shapes = Counter(s.anat_shape for s in scans if s.anat_shape)
    rest_shapes  = Counter(s.rest_shape  for s in scans if s.rest_shape)

    print(f"\n{_SEP}")
    print(f"  PHASE 1.4 — {site} DATASET EXPLORATION SUMMARY")
    print(_SEP)
    print(f"  Total subjects    : {len(scans)}")
    print(f"  ASD               : {dx.get('ASD', 0)}")
    print(f"  CONTROL           : {dx.get('CONTROL', 0)}")
    if ages:
        print(f"  Age min/max/mean  : {min(ages):.2f} / {max(ages):.2f} / {np.mean(ages):.2f}")
    print(f"  Sex  M / F        : {sex.get('M', 0)} / {sex.get('F', 0)}")
    print(f"  FIQ available     : {fiq_ok} / {len(meta_l)}")
    print(_SEP)
    mod_counts = {m: sum(1 for s in scans if getattr(s, f"{m}_exists"))
                  for m in ("anat", "dti", "bval", "bvec", "rest")}
    total = len(scans)
    print("  Modality presence:")
    for m, c in mod_counts.items():
        bar = "#" * c + "-" * (total - c)
        print(f"    {m:4s}  [{bar}]  {c}/{total}")
    print(_SEP)
    print(f"  DTI subjects       : {len(dti_s)}")
    if dti_s:
        valid_dti    = sum(1 for s in dti_s if s.dti_consistent is True)
        mismatch_dti = sum(1 for s in dti_s if s.dti_consistent is False)
        print(f"  Valid DTI          : {valid_dti}")
        print(f"  DTI mismatch       : {mismatch_dti}")
        print(f"  Unique b-values    : {sorted(all_bvals)}")
        print(f"  Recommended action : {_common_recommendation(dti_s)}")
    else:
        print("  *** NO DTI DATA AT THIS SITE ***")
    print(_SEP)
    print("  Anat shape patterns:")
    for shape, cnt in anat_shapes.most_common():
        print(f"    {shape}  ×  {cnt} subjects")
    print("  Rest shape patterns:")
    for shape, cnt in rest_shapes.most_common():
        print(f"    {shape}  ×  {cnt} subjects")
    print(_SEP)
    status_counts = Counter(s.validation_status for s in scans)
    print("  Validation status:")
    for status, cnt in status_counts.most_common():
        print(f"    {status:<25} {cnt}")
    problem = [s for s in scans if s.validation_status not in ("valid",)]
    if problem:
        print("\n  Subjects with validation issues:")
        for s in problem:
            print(f"    {s.subject_id:<10}  status={s.validation_status}  missing={s.missing_modalities}")
    print()


def _common_recommendation(dti_subs: list[SubjectScanResult]) -> str:
    recs = Counter(s.recommended_action for s in dti_subs if s.recommended_action)
    return recs.most_common(1)[0][0] if recs else "none"
