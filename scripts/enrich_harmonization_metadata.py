"""
NeuroFiber Phase 3R.5 — Phenotype Metadata Enrichment

Loads ABIDE-II per-site phenotype CSVs from raw data, normalizes columns,
and joins with harmonization_metadata.csv to produce a fully enriched table.

Phenotype CSV locations:
  data/raw/abide_ii/<site>/ABIDEII-<SITE>/ABIDEII-<SITE>.csv

Column mapping:
  SUB_ID          -> subject_id
  DX_GROUP        -> diagnosis   (1=ASD, 2=CONTROL; -9999/blank=unknown)
  AGE_AT_SCAN     -> age         (years; strip trailing spaces from column name)
  SEX             -> sex         (1=M, 2=F; -9999/blank=unknown)
  FIQ             -> fiq
  SITE_ID         -> site_raw

Output:
  data/processed_v2b/connectome_features/harmonization_metadata_enriched.csv

Columns (original + enriched):
  subject_id, site, dataset,
  age, sex, diagnosis, fiq,
  scanner_info, b_value, direction_count,
  mean_fa, mean_md, density, mean_streamline_length,
  phenotype_source, age_available, sex_available, diagnosis_available

Usage:
  python scripts/enrich_harmonization_metadata.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

log = logging.getLogger("enrich_metadata")

RAW_ROOT  = REPO_ROOT / "data" / "raw" / "abide_ii"
FEAT_DIR  = REPO_ROOT / "data" / "processed_v2b" / "connectome_features"
OUT_PATH  = FEAT_DIR / "harmonization_metadata_enriched.csv"

# Known site folder → phenotype CSV location
_SITE_PHENO_PATHS = {
    "BNI":    RAW_ROOT / "bni"   / "ABIDEII-BNI_1"  / "ABIDEII-BNI_1.csv",
    "IP_1":   RAW_ROOT / "ip"    / "ABIDEII-IP_1"   / "ABIDEII-IP_1.csv",
    "NYU_1":  RAW_ROOT / "nyu1"  / "ABIDEII-NYU_1"  / "ABIDEII-NYU_1.csv",
    "NYU_2":  RAW_ROOT / "nyu2"  / "ABIDEII-NYU_2"  / "ABIDEII-NYU_2.csv",
    "SDSU_1": RAW_ROOT / "sdsu"  / "ABIDEII-SDSU_1" / "ABIDEII-SDSU_1.csv",
    "TCD_1":  RAW_ROOT / "tcd"   / "ABIDEII-TCD_1"  / "ABIDEII-TCD_1.csv",
}

_ENRICHED_FIELDS = [
    "subject_id", "site", "dataset",
    "age", "sex", "diagnosis", "fiq",
    "scanner_info", "b_value", "direction_count",
    "mean_fa", "mean_md", "density", "mean_streamline_length",
    "phenotype_source", "age_available", "sex_available", "diagnosis_available",
]

_MISSING_CODES = {"", "-9999", "-9999.0", "nan", "NaN", "N/A"}


def _safe(val: str) -> str:
    """Return empty string for ABIDE missing codes."""
    return "" if str(val).strip() in _MISSING_CODES else str(val).strip()


def _normalize_col(col: str) -> str:
    """Strip whitespace from column name."""
    return col.strip()


def load_phenotype_csvs(site_list: list[str]) -> dict[str, dict]:
    """
    Load all available site phenotype CSVs.
    Returns {subject_id: {age, sex, diagnosis, fiq, site_raw, phenotype_source}}.
    """
    pheno: dict[str, dict] = {}

    for site in site_list:
        path = _SITE_PHENO_PATHS.get(site)
        if path is None or not path.exists():
            log.warning("[%s] phenotype CSV not found: %s", site, path)
            continue

        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize column names (strip whitespace)
            norm_fields = {_normalize_col(k): k for k in (reader.fieldnames or [])}

            n_loaded = 0
            for raw_row in reader:
                # Re-key with normalized column names
                row = {_normalize_col(k): v for k, v in raw_row.items()}

                sub_id = _safe(row.get("SUB_ID", ""))
                if not sub_id:
                    continue

                # DX_GROUP: 1=ASD, 2=CONTROL
                dx_raw = _safe(row.get("DX_GROUP", ""))
                if dx_raw == "1":
                    diagnosis = "ASD"
                elif dx_raw == "2":
                    diagnosis = "CONTROL"
                else:
                    diagnosis = ""

                # SEX: 1=M, 2=F
                sex_raw = _safe(row.get("SEX", ""))
                if sex_raw == "1":
                    sex = "M"
                elif sex_raw == "2":
                    sex = "F"
                else:
                    sex = ""

                age = _safe(row.get("AGE_AT_SCAN", ""))
                fiq = _safe(row.get("FIQ", ""))

                pheno[sub_id] = {
                    "age":               age,
                    "sex":               sex,
                    "diagnosis":         diagnosis,
                    "fiq":               fiq,
                    "phenotype_source":  path.name,
                }
                n_loaded += 1

        log.info("[%s] loaded %d phenotype rows from %s", site, n_loaded, path.name)

    return pheno


def enrich_metadata(
    meta_path: Path,
    pheno:     dict[str, dict],
) -> tuple[list[dict], dict]:
    """
    Join harmonization_metadata.csv with phenotype lookup.
    Returns (enriched_rows, coverage_report).
    """
    enriched: list[dict] = []
    coverage = {
        "total":             0,
        "age_available":     0,
        "sex_available":     0,
        "diagnosis_available": 0,
        "fiq_available":     0,
        "by_site":           {},
    }

    with open(meta_path) as f:
        for row in csv.DictReader(f):
            sub_id = row.get("subject_id", "").strip()
            site   = row.get("site", "").strip()
            coverage["total"] += 1

            p = pheno.get(sub_id, {})

            age       = p.get("age", "")       or row.get("age", "")
            sex       = p.get("sex", "")       or row.get("sex", "")
            diagnosis = p.get("diagnosis", "") or row.get("diagnosis", "")
            fiq       = p.get("fiq", "")
            source    = p.get("phenotype_source", "not_found")

            if age:
                coverage["age_available"] += 1
            if sex:
                coverage["sex_available"] += 1
            if diagnosis:
                coverage["diagnosis_available"] += 1
            if fiq:
                coverage["fiq_available"] += 1

            site_stats = coverage["by_site"].setdefault(site, {
                "total": 0, "age": 0, "sex": 0, "diagnosis": 0
            })
            site_stats["total"] += 1
            if age:       site_stats["age"]       += 1
            if sex:       site_stats["sex"]       += 1
            if diagnosis: site_stats["diagnosis"] += 1

            enriched.append({
                "subject_id":             sub_id,
                "site":                   site,
                "dataset":                row.get("dataset", ""),
                "age":                    age,
                "sex":                    sex,
                "diagnosis":              diagnosis,
                "fiq":                    fiq,
                "scanner_info":           row.get("scanner_info", ""),
                "b_value":                row.get("b_value", ""),
                "direction_count":        row.get("direction_count", ""),
                "mean_fa":                row.get("mean_fa", ""),
                "mean_md":                row.get("mean_md", ""),
                "density":                row.get("density", ""),
                "mean_streamline_length": row.get("mean_streamline_length", ""),
                "phenotype_source":       source,
                "age_available":          "yes" if age else "no",
                "sex_available":          "yes" if sex else "no",
                "diagnosis_available":    "yes" if diagnosis else "no",
            })

    return enriched, coverage


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich harmonization metadata with ABIDE-II phenotypes.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    meta_path = FEAT_DIR / "harmonization_metadata.csv"
    if not meta_path.exists():
        log.error("harmonization_metadata.csv not found: %s", meta_path)
        sys.exit(1)

    # Load all site phenotype CSVs
    all_sites = list(_SITE_PHENO_PATHS.keys())
    pheno = load_phenotype_csvs(all_sites)
    log.info("Total phenotype records loaded: %d", len(pheno))

    # Enrich
    enriched, coverage = enrich_metadata(meta_path, pheno)

    # Write output
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ENRICHED_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(enriched)

    log.info("Enriched metadata → %s  (%d rows)", OUT_PATH.relative_to(REPO_ROOT), len(enriched))

    # Coverage report
    log.info("=" * 60)
    log.info("METADATA COVERAGE REPORT")
    log.info("=" * 60)
    log.info("Total subjects    : %d", coverage["total"])
    log.info("Age available     : %d / %d  (%.1f%%)",
             coverage["age_available"], coverage["total"],
             100 * coverage["age_available"] / max(coverage["total"], 1))
    log.info("Sex available     : %d / %d  (%.1f%%)",
             coverage["sex_available"], coverage["total"],
             100 * coverage["sex_available"] / max(coverage["total"], 1))
    log.info("Diagnosis avail.  : %d / %d  (%.1f%%)",
             coverage["diagnosis_available"], coverage["total"],
             100 * coverage["diagnosis_available"] / max(coverage["total"], 1))
    log.info("FIQ available     : %d / %d  (%.1f%%)",
             coverage["fiq_available"], coverage["total"],
             100 * coverage["fiq_available"] / max(coverage["total"], 1))
    log.info("-" * 60)
    log.info("Per-site breakdown:")
    for site, stats in sorted(coverage["by_site"].items()):
        n = stats["total"]
        log.info(
            "  [%s]  n=%d  age=%d  sex=%d  diagnosis=%d%s",
            site, n, stats["age"], stats["sex"], stats["diagnosis"],
            "  ← PARTIAL" if stats["diagnosis"] < n else "",
        )

    # Save coverage JSON
    cov_path = FEAT_DIR / "phenotype_coverage_report.json"
    cov_path.write_text(json.dumps(coverage, indent=2))
    log.info("Coverage report → %s", cov_path.relative_to(REPO_ROOT))

    log.info("=" * 60)
    log.info("Next step: re-run Phase 3R.5 harmonization using enriched metadata.")
    log.info("Command: python scripts/run_connectome_harmonization.py")
    log.info("(The harmonization module will auto-detect the enriched file.)")


if __name__ == "__main__":
    main()
