#!/usr/bin/env python3
"""
NeuroFiber — ABIDE II full download + validation + preprocessing pipeline.

Downloads every ABIDE II site from the INDI public S3 bucket, converts the
BIDS layout to the project's legacy naming convention, runs Phase 1 modality
validation, and (for sites with DTI) runs Phase 2 preprocessing.

Raw data is NEVER overwritten or modified.

S3 source:
  s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-{SITE}_{N}/

BIDS → legacy conversion (per subject):
  BIDS:   sub-{id}/ses-{session}/anat/sub-{id}_{session}_run-1_T1w.nii.gz
  Legacy: {id}/session_1/anat_1/anat.nii.gz

  BIDS:   sub-{id}/ses-{session}/dwi/sub-{id}_{session}_run-1_dwi.nii.gz
  Legacy: {id}/session_1/dti_1/dti.nii.gz

  BIDS:   sub-{id}/ses-{session}/dwi/sub-{id}_{session}_run-1_dwi.bval
  Legacy: {id}/session_1/dti_1/dti.bval

  BIDS:   sub-{id}/ses-{session}/dwi/sub-{id}_{session}_run-1_dwi.bvec
  Legacy: {id}/session_1/dti_1/dti.bvec

  BIDS:   sub-{id}/ses-{session}/func/sub-{id}_{session}_task-rest_run-1_bold.nii.gz
  Legacy: {id}/session_1/rest_1/rest.nii.gz

Usage:
    python scripts/download_abide_pipeline.py \\
        --raw-root  data/raw/abide_ii \\
        --data-root data \\
        --workers   8

    # Specific sites only:
    python scripts/download_abide_pipeline.py \\
        --sites KKI_1 NYU_1 ONRC_2 \\
        --raw-root data/raw/abide_ii

    # Dry run (show what would be downloaded):
    python scripts/download_abide_pipeline.py --dry-run

    # Download only, skip preprocessing:
    python scripts/download_abide_pipeline.py --skip-preprocessing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from neurofiber.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_BUCKET  = "fcp-indi"
S3_PREFIX  = "data/Projects/ABIDE2/RawData"
_REGION    = "us-east-1"

# Map: S3 site folder name → (local subfolder abbreviation, local dataset name)
# "local dataset name" is what the folder will be called under raw_root/<abbrev>/
SITE_MAP: dict[str, tuple[str, str]] = {
    "ABIDEII-BNI_1":    ("bni",       "ABIDEII-BNI_1"),
    "ABIDEII-EMC_1":    ("emc",       "ABIDEII-EMC_1"),
    "ABIDEII-ETHZ_1":   ("eth",       "ABIDEII-ETH_1"),   # S3 uses ETHZ, we use ETH
    "ABIDEII-GU_1":     ("gu",        "ABIDEII-GU_1"),
    "ABIDEII-IP_1":     ("ip",        "ABIDEII-IP_1"),
    "ABIDEII-IU_1":     ("iu",        "ABIDEII-IU_1"),
    "ABIDEII-KKI_1":    ("kki",       "ABIDEII-KKI_1"),
    "ABIDEII-KUL_3":    ("kul",       "ABIDEII-KUL_3"),
    "ABIDEII-NYU_1":    ("nyu1",      "ABIDEII-NYU_1"),
    "ABIDEII-NYU_2":    ("nyu2",      "ABIDEII-NYU_2"),
    "ABIDEII-OHSU_1":   ("ohsu",      "ABIDEII-OHSU_1"),
    "ABIDEII-ONRC_2":   ("onrc",      "ABIDEII-ONRC_2"),
    "ABIDEII-SDSU_1":   ("sdsu",      "ABIDEII-SDSU_1"),
    "ABIDEII-TCD_1":    ("tcd",       "ABIDEII-TCD_1"),
    "ABIDEII-UCD_1":    ("ucd",       "ABIDEII-UCD_1"),
    "ABIDEII-UCLA_1":   ("ucla1",     "ABIDEII-UCLA_1"),
    "ABIDEII-UCLA_Long":("ucla_long", "ABIDEII-UCLA_Long"),
    "ABIDEII-UPSM_Long":("upsm_long", "ABIDEII-UPSM_Long"),
    "ABIDEII-USM_1":    ("usm",       "ABIDEII-USM_1"),
}

# Files to download per modality (BIDS suffix → legacy name)
# Key: BIDS filename suffix pattern; Value: legacy filename
_MODALITY_MAP: dict[str, str] = {
    "_T1w.nii.gz":            "anat_1/anat.nii.gz",
    "_dwi.nii.gz":            "dti_1/dti.nii.gz",
    "_dwi.bval":              "dti_1/dti.bval",
    "_dwi.bvec":              "dti_1/dti.bvec",
    "_dwi.bvecs_image":       "dti_1/dti.bvec",   # NYU_1 image-frame bvec naming
    "_bold.nii.gz":           "rest_1/rest.nii.gz",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SiteDownloadResult:
    site_name: str
    local_abbrev: str
    local_dataset_dir: str
    status: str                # "skipped" | "downloaded" | "failed"
    subjects_downloaded: int   = 0
    files_downloaded: int      = 0
    files_skipped: int         = 0
    validation_status: str     = "not_run"
    preprocessing_status: str  = "not_run"
    error_message: Optional[str] = field(default=None)
    elapsed_seconds: float     = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3_client():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name=_REGION,
    )


def _list_s3_subjects(s3, site_s3_prefix: str) -> list[str]:
    """Return all BIDS sub-* prefixes for a site."""
    subjects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=site_s3_prefix + "/",
        Delimiter="/",
    ):
        for cp in page.get("CommonPrefixes", []):
            prefix = cp["Prefix"]
            folder = prefix.rstrip("/").split("/")[-1]
            if folder.startswith("sub-"):
                subjects.append(prefix)
    return sorted(subjects)


def _discover_session(s3, sub_prefix: str) -> Optional[str]:
    """Detect the session folder name (e.g. ses-1, ses-ONRC2baseline)."""
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=sub_prefix,
        Delimiter="/",
    )
    for cp in resp.get("CommonPrefixes", []):
        folder = cp["Prefix"].rstrip("/").split("/")[-1]
        if folder.startswith("ses-"):
            return folder
    return None


def _list_s3_modality_files(s3, sub_ses_prefix: str) -> list[str]:
    """Return all file keys under a subject/session prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=sub_ses_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# BIDS → legacy mapping
# ---------------------------------------------------------------------------

def _bids_key_to_legacy(
    bids_key: str,
    subject_id: str,
    local_dataset_dir: Path,
) -> Optional[Path]:
    """
    Map an S3 BIDS key to a local legacy path.
    Returns None if the file should not be downloaded.
    """
    filename = bids_key.split("/")[-1]
    for bids_suffix, legacy_rel in _MODALITY_MAP.items():
        if filename.endswith(bids_suffix):
            return local_dataset_dir / subject_id / "session_1" / legacy_rel
    return None


# ---------------------------------------------------------------------------
# Download a single file
# ---------------------------------------------------------------------------

def _download_file(
    s3,
    s3_key: str,
    local_path: Path,
    dry_run: bool = False,
) -> bool:
    """Download one S3 file to local_path. Returns True if downloaded, False if skipped."""
    if local_path.exists():
        return False   # already exists → skip
    if dry_run:
        logger.debug("  DRY-RUN would download: %s", local_path)
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
    return True


# ---------------------------------------------------------------------------
# Phenotype: participants.tsv → legacy CSV
# ---------------------------------------------------------------------------

def _download_phenotype(
    s3,
    site_s3_prefix: str,
    local_dataset_dir: Path,
    local_dataset_name: str,
    dry_run: bool = False,
) -> None:
    """Download participants.tsv and convert to legacy CSV format."""
    tsv_key  = f"{site_s3_prefix}/participants.tsv"
    csv_path = local_dataset_dir / f"{local_dataset_name}.csv"

    if csv_path.exists():
        logger.debug("Phenotype CSV already exists: %s", csv_path)
        return

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=tsv_key)
        content = obj["Body"].read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Could not download participants.tsv from %s: %s", tsv_key, exc)
        return

    if dry_run:
        logger.debug("DRY-RUN: would save phenotype CSV → %s", csv_path)
        return

    # Parse TSV; rename columns to legacy uppercase format
    import io
    df = pd.read_csv(io.StringIO(content), sep="\t", na_values=["n/a", "N/A", ""])
    col_renames = {
        "participant_id": "SUB_ID",
        "site_id":        "SITE_ID",
        "dx_group":       "DX_GROUP",
        "age_at_scan":    "AGE_AT_SCAN",
        "age_at_scan ":   "AGE_AT_SCAN",   # trailing space variant
        "sex":            "SEX",
        "fiq":            "FIQ",
        "viq":            "VIQ",
        "piq":            "PIQ",
    }
    df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})

    # Strip "sub-" prefix from participant IDs
    if "SUB_ID" in df.columns:
        df["SUB_ID"] = df["SUB_ID"].astype(str).str.replace(r"^sub-", "", regex=True)
        # Convert to int where possible
        df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce").fillna(df["SUB_ID"])

    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("  Phenotype CSV saved → %s  (%d rows)", csv_path.name, len(df))


# ---------------------------------------------------------------------------
# Download one site
# ---------------------------------------------------------------------------

def download_site(
    site_name: str,
    raw_root: Path,
    workers: int = 4,
    dry_run: bool = False,
) -> SiteDownloadResult:
    """
    Download all subjects for one ABIDE II site from S3.

    Converts BIDS layout to legacy naming on the fly.
    Skips files that already exist.
    """
    t0 = time.time()
    abbrev, local_dataset_name = SITE_MAP[site_name]
    local_dataset_dir = raw_root / abbrev / local_dataset_name

    result = SiteDownloadResult(
        site_name=site_name,
        local_abbrev=abbrev,
        local_dataset_dir=str(local_dataset_dir),
        status="downloading",
    )

    site_s3_prefix = f"{S3_PREFIX}/{site_name}"

    try:
        s3 = _make_s3_client()

        # Phenotype TSV → CSV
        _download_phenotype(s3, site_s3_prefix, local_dataset_dir,
                            local_dataset_name, dry_run=dry_run)

        # List subjects
        subject_prefixes = _list_s3_subjects(s3, site_s3_prefix)
        logger.info("[%s] found %d subjects on S3", site_name, len(subject_prefixes))

        total_downloaded = 0
        total_skipped    = 0
        subjects_done    = 0

        def _process_subject(sub_prefix: str) -> tuple[int, int]:
            """Returns (files_downloaded, files_skipped)."""
            sub_folder  = sub_prefix.rstrip("/").split("/")[-1]
            subject_id  = sub_folder.removeprefix("sub-")
            _s3         = _make_s3_client()   # thread-local client

            session = _discover_session(_s3, sub_prefix)
            if not session:
                logger.warning("  [%s] no session folder found — skipping", subject_id)
                return 0, 0

            sub_ses_prefix = f"{sub_prefix}{session}/"
            keys = _list_s3_modality_files(_s3, sub_ses_prefix)

            dl, sk = 0, 0
            for key in keys:
                local_path = _bids_key_to_legacy(key, subject_id, local_dataset_dir)
                if local_path is None:
                    continue
                if _download_file(_s3, key, local_path, dry_run=dry_run):
                    dl += 1
                else:
                    sk += 1
            return dl, sk

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_subject, sp): sp
                       for sp in subject_prefixes}
            for fut in as_completed(futures):
                dl, sk = fut.result()
                total_downloaded += dl
                total_skipped    += sk
                subjects_done    += 1
                if subjects_done % 10 == 0:
                    logger.info("  [%s] %d/%d subjects processed",
                                site_name, subjects_done, len(subject_prefixes))

        result.subjects_downloaded = len(subject_prefixes)
        result.files_downloaded    = total_downloaded
        result.files_skipped       = total_skipped
        result.status              = "downloaded"
        logger.info("[%s] done — %d files downloaded, %d skipped",
                    site_name, total_downloaded, total_skipped)

    except Exception as exc:
        logger.error("[%s] download failed: %s", site_name, exc)
        result.status        = "failed"
        result.error_message = str(exc)

    result.elapsed_seconds = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Check if a site is already downloaded
# ---------------------------------------------------------------------------

def _is_downloaded(site_name: str, raw_root: Path, min_subjects: int = 1) -> bool:
    """Return True if the local site folder exists and has ≥ min_subjects."""
    abbrev, local_dataset_name = SITE_MAP[site_name]
    local_dir = raw_root / abbrev / local_dataset_name
    if not local_dir.exists():
        return False
    n_subjects = sum(1 for p in local_dir.iterdir()
                     if p.is_dir() and p.name.isdigit())
    return n_subjects >= min_subjects


# ---------------------------------------------------------------------------
# Phase 1 validation
# ---------------------------------------------------------------------------

def _run_validation(
    site_name: str,
    raw_root: Path,
    registry_dir: Path,
) -> str:
    """Run Phase 1 scan + modality validation. Returns status string."""
    abbrev, local_dataset_name = SITE_MAP[site_name]
    site_label  = site_name.split("-")[1].split("_")[0]   # e.g. "BNI"
    dataset_dir = raw_root / abbrev / local_dataset_name
    meta_csv    = dataset_dir / f"{local_dataset_name}.csv"

    try:
        from neurofiber.registry.eth_scanner import scan_site
        from neurofiber.registry.metadata_loader import load_abide2_metadata
        from neurofiber.registry.site_explorer import (
            join_scan_meta, make_validation_summary_df,
            make_dti_diagnostic_df, make_site_manifest,
        )

        scans = scan_site(dataset_dir, site=site_label, dataset="ABIDE_II")
        meta  = load_abide2_metadata(meta_csv, site=site_label) if meta_csv.exists() else {}

        # Save registry files
        registry_dir.mkdir(parents=True, exist_ok=True)
        prefix = site_label.lower()

        joined = join_scan_meta(scans, meta)
        pd.DataFrame(joined).to_csv(
            registry_dir / f"{prefix}_dataset_index.csv", index=False
        )
        make_validation_summary_df(scans).to_csv(
            registry_dir / f"{prefix}_validation_summary.csv", index=False
        )
        make_dti_diagnostic_df(scans).to_csv(
            registry_dir / f"{prefix}_dti_diagnostic_report.csv", index=False
        )
        manifest = make_site_manifest(site_label, "ABIDE_II", scans, meta)
        (registry_dir / f"{prefix}_site_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        n_valid  = sum(1 for s in scans if s.validation_status == "valid")
        n_dti    = sum(1 for s in scans if s.dti_exists)
        logger.info("[%s] validation: %d subjects, %d valid, %d with DTI",
                    site_label, len(scans), n_valid, n_dti)
        return "success"

    except Exception as exc:
        logger.error("[%s] validation failed: %s", site_name, exc)
        return f"failed: {exc}"


# ---------------------------------------------------------------------------
# Phase 2 preprocessing (DTI sites only)
# ---------------------------------------------------------------------------

def _run_preprocessing(
    site_name: str,
    raw_root: Path,
    processed_root: Path,
) -> str:
    """Run Phase 2.1→2.3 DTI preprocessing if DTI is present. Returns status string."""
    abbrev, local_dataset_name = SITE_MAP[site_name]
    dataset_dir = raw_root / abbrev / local_dataset_name
    out_dir     = processed_root / "abide_ii" / abbrev / local_dataset_name

    try:
        from neurofiber.registry.eth_scanner import scan_site
        site_label = site_name.split("-")[1].split("_")[0]
        scans = scan_site(dataset_dir, site=site_label, dataset="ABIDE_II")
        dti_subjects = [s for s in scans if s.dti_exists]

        if not dti_subjects:
            logger.info("[%s] no DTI subjects — skipping Phase 2 pipeline", site_name)
            return "skipped_no_dti"

        logger.info("[%s] running DTI preprocessing on %d subjects",
                    site_name, len(dti_subjects))

        # Phase 2.1 — DTI volume correction
        from neurofiber.preprocessing.dti_correction import run_correction_batch
        corr_records = run_correction_batch(
            dataset_root=dataset_dir,
            output_root=out_dir,
        )
        n_corrected = sum(1 for r in corr_records if r.action == "volume_removed")
        n_manual    = sum(1 for r in corr_records if r.action == "requires_manual_review")
        logger.info("[%s] Phase 2.1: %d corrected, %d manual review, %d failed",
                    site_name, n_corrected, n_manual,
                    sum(1 for r in corr_records if r.status == "failed"))

        # Phase 2.2 — Tensor estimation (only on subjects that were corrected)
        from neurofiber.preprocessing.tensor_estimation import run_tensor_batch
        tensor_records = run_tensor_batch(processed_root=out_dir)
        n_tensors = sum(1 for r in tensor_records if r.status == "success")
        logger.info("[%s] Phase 2.2: %d/%d tensor maps generated",
                    site_name, n_tensors, len(tensor_records))

        # Phase 2.3 — Brain masking
        from neurofiber.preprocessing.brain_masking import run_masking_batch
        mask_records = run_masking_batch(processed_root=out_dir, save_png=True)
        n_masks = sum(1 for r in mask_records if r.status == "success")
        logger.info("[%s] Phase 2.3: %d/%d brain masks created",
                    site_name, n_masks, len(mask_records))

        return "success"

    except Exception as exc:
        logger.error("[%s] preprocessing failed: %s", site_name, exc)
        return f"failed: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ABIDE II sites + run NeuroFiber validation and preprocessing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/abide_ii"),
        help="Local root for raw downloads (default: data/raw/abide_ii).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Project data root for processed output and registry (default: data).",
    )
    p.add_argument(
        "--sites",
        nargs="*",
        metavar="SITE",
        help=(
            "S3 site suffixes to process, e.g. KKI_1 NYU_1 ONRC_2. "
            "Omit to process ALL sites."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel download threads per site (default: 6).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without writing any files.",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Phase 1 modality validation after download.",
    )
    p.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip Phase 2 DTI preprocessing after validation.",
    )
    p.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download sites that appear already present locally.",
    )
    p.add_argument(
        "--list-sites",
        action="store_true",
        help="Print all known sites and their local download status, then exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    raw_root      = args.raw_root
    processed_root = args.data_root / "processed"
    registry_dir  = args.data_root / "registry"

    # --list-sites
    if args.list_sites:
        print(f"\n{'Site':<25} {'Local status':<15} {'Local path'}")
        print("-" * 72)
        for s3_name in sorted(SITE_MAP):
            abbrev, local_name = SITE_MAP[s3_name]
            downloaded = _is_downloaded(s3_name, raw_root)
            status     = "✓ downloaded" if downloaded else "not downloaded"
            local_path = raw_root / abbrev / local_name
            print(f"  {s3_name:<23} {status:<15} {local_path}")
        print()
        return 0

    # Resolve which sites to process
    if args.sites:
        # Accept either "ABIDEII-KKI_1" or just "KKI_1"
        requested: list[str] = []
        for s in args.sites:
            full = s if s.startswith("ABIDEII-") else f"ABIDEII-{s}"
            if full not in SITE_MAP:
                logger.error("Unknown site: %s  (known: %s)", s, ", ".join(SITE_MAP))
                return 1
            requested.append(full)
    else:
        requested = list(SITE_MAP.keys())

    logger.info("Processing %d sites", len(requested))
    results: list[SiteDownloadResult] = []

    for site_name in requested:
        abbrev, local_name = SITE_MAP[site_name]

        # Skip if already downloaded (unless --force-redownload)
        if not args.force_redownload and _is_downloaded(site_name, raw_root):
            logger.info("[%s] already downloaded — skipping download", site_name)
            result = SiteDownloadResult(
                site_name=site_name,
                local_abbrev=abbrev,
                local_dataset_dir=str(raw_root / abbrev / local_name),
                status="skipped",
            )
        else:
            logger.info("[%s] starting download …", site_name)
            result = download_site(
                site_name, raw_root,
                workers=args.workers,
                dry_run=args.dry_run,
            )

        # Phase 1 validation
        if not args.skip_validation and not args.dry_run and result.status in ("downloaded", "skipped"):
            logger.info("[%s] running Phase 1 validation …", site_name)
            result.validation_status = _run_validation(site_name, raw_root, registry_dir)

        # Phase 2 preprocessing (DTI only, only if validation passed)
        if (not args.skip_preprocessing and not args.dry_run
                and result.validation_status in ("success", "not_run")
                and result.status in ("downloaded", "skipped")):
            logger.info("[%s] running Phase 2 preprocessing …", site_name)
            result.preprocessing_status = _run_preprocessing(
                site_name, raw_root, processed_root
            )

        results.append(result)

    # Print summary
    _print_summary(results, args)

    # Save run report
    if not args.dry_run:
        report_path = registry_dir / "download_pipeline_report.json"
        registry_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps([r.to_dict() for r in results], indent=2, default=str)
        )
        logger.info("Run report → %s", report_path)

    n_failed = sum(1 for r in results if r.status == "failed")
    return 1 if n_failed else 0


def _print_summary(results: list[SiteDownloadResult], args) -> None:
    _SEP = "=" * 66
    total      = len(results)
    downloaded = sum(1 for r in results if r.status == "downloaded")
    skipped    = sum(1 for r in results if r.status == "skipped")
    failed     = sum(1 for r in results if r.status == "failed")

    print(f"\n{_SEP}")
    prefix = "  [DRY RUN] " if args.dry_run else "  "
    print(f"{prefix}NEUROFIBER ABIDE II DOWNLOAD + PIPELINE SUMMARY")
    if args.dry_run:
        print("  No files were written. Remove --dry-run to execute.")
    print(_SEP)
    print(f"  Sites processed  : {total}")
    print(f"  Downloaded       : {downloaded}")
    print(f"  Skipped (exists) : {skipped}")
    print(f"  Failed           : {failed}")
    print(_SEP)
    print(f"  {'Site':<24} {'Download':<12} {'Validation':<15} {'Preprocessing'}")
    print(f"  {'-'*24} {'-'*12} {'-'*15} {'-'*14}")
    for r in results:
        print(f"  {r.site_name:<24} {r.status:<12} {r.validation_status:<15} {r.preprocessing_status}")
    print(_SEP)
    if failed:
        print("\n  Failures:")
        for r in results:
            if r.status == "failed":
                print(f"    {r.site_name}  {r.error_message}")
    print()


if __name__ == "__main__":
    sys.exit(main())
