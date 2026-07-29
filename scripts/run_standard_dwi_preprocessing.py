"""
Phase 2R.1 — Standard DWI Preprocessing Runner

Usage:
    python run_standard_dwi_preprocessing.py \\
        --config configs/phase2r_standard_dwi_preprocessing.yaml

    python run_standard_dwi_preprocessing.py \\
        --config configs/phase2r_standard_dwi_preprocessing.yaml \\
        --resume          # skip already-preprocessed subjects

    python run_standard_dwi_preprocessing.py \\
        --config configs/phase2r_standard_dwi_preprocessing.yaml \\
        --subjects 29006 29007  # run specific subjects only

    python run_standard_dwi_preprocessing.py \\
        --config configs/phase2r_standard_dwi_preprocessing.yaml \\
        --preflight       # detect backends + count subjects, don't run

Outputs:
    data/processed_v2/abide_ii/{site}/{dataset}/{subj}/session_1/dti_1/
        dwi_preprocessed.nii.gz
        dwi_preprocessed.bval
        dwi_preprocessed.bvec
        preprocessing_report.json
        qc/
            qc_metrics.json
            noise_map.nii.gz  (if MRtrix3 dwidenoise used)

    data/processed_v2/phase2r_1_standard_dwi_preprocessing_summary.csv
    data/processed_v2/phase2r_1_site_summary.csv

Safety:
    data/processed/ is NEVER touched.
    data/raw/ writes are blocked by guard_no_raw_write().
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

from neurofiber.preprocessing.standard_dwi_preprocessing import (
    detect_backends,
    run_batch_pipeline,
    save_summary_csvs,
    site_to_folder,
    PIPELINE_VERSION,
)

log = logging.getLogger("phase2r_preprocessing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Phase 2R.1 Standard DWI Preprocessing (v{PIPELINE_VERSION})"
    )
    p.add_argument("--config",    required=True, help="YAML config path")
    p.add_argument("--resume",    action="store_true",
                   help="Skip subjects whose preprocessing_report.json already exists")
    p.add_argument("--subjects",  nargs="+",
                   help="Restrict to these subject IDs (for testing or reruns)")
    p.add_argument("--preflight", action="store_true",
                   help="Detect backends and list subjects; do not run pipeline")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def _setup_logging(level_str: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level_str),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def _preflight(cfg: dict, raw_root: Path) -> None:
    """Print backend availability and subject counts; do not run pipeline."""
    backends = detect_backends()
    pv = cfg.get("pipeline_version", PIPELINE_VERSION)
    out_root = cfg["safety"]["output_root"]
    log.info("=" * 60)
    log.info("Phase %s PREFLIGHT  (pipeline_version=%s)", pv, pv)
    log.info("=" * 60)
    log.info("MRtrix3 : %s", backends.mrtrix_version or "NOT FOUND")
    log.info("FSL     : %s", backends.fsl_version    or "NOT FOUND")
    log.info("ANTs    : %s", "found" if backends.ants_available else "NOT FOUND")
    log.info("-" * 60)

    pref = cfg["backend_preference"]
    fall = cfg["fallbacks"]
    denoise_b = backends.denoise_backend(pref["denoise"], fall.get("denoise", "dipy_patch2self"))
    log.info("Step 2 denoise : %s", denoise_b)
    log.info("Step 3 gibbs   : %s", backends.gibbs_backend(pref["gibbs"], fall["gibbs"]))
    log.info("Step 4 eddy    : %s", backends.eddy_backend(pref["eddy"],   fall["eddy"]))
    log.info("Step 5 bias    : %s", backends.bias_backend(pref["bias"],   fall["bias"]))

    pe_dir = cfg.get("phase_encoding", {}).get("default_pe_dir")
    log.info("Phase-enc dir  : %s", pe_dir or "UNKNOWN (eddy will be skipped)")
    log.info("-" * 60)

    total = 0
    for site_display in cfg["included_sites"]:
        folder   = site_to_folder(site_display)
        site_dir = raw_root / folder
        if not site_dir.exists():
            log.warning("[%s] raw directory not found: %s", site_display, site_dir)
            continue
        n = len(sorted(site_dir.rglob("dti.nii.gz")))
        flag = "  ← QC-labelled only" if folder == "ip" else ""
        log.info("[%s] %d subjects%s", site_display, n, flag)
        total += n
    log.info("-" * 60)
    log.info("Total subjects: %d", total)
    log.info("Output root   : %s", out_root)
    log.info("=" * 60)


def main() -> None:
    args = parse_args()

    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    output_root_early = REPO_ROOT / cfg["safety"]["output_root"]
    log_file = output_root_early / f"phase2r_{cfg.get('pipeline_version','1').replace('.','_')}_preprocessing_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_level, log_file)

    raw_root    = REPO_ROOT / cfg["safety"]["raw_root"]
    output_root = REPO_ROOT / cfg["safety"]["output_root"]

    # Safety: never write into v1 processed or raw
    for forbidden in ["data/processed", "data/raw"]:
        forbidden_path = REPO_ROOT / forbidden
        assert str(output_root.resolve()) != str(forbidden_path.resolve()), (
            f"SAFETY: output_root must not point to {forbidden}. "
            "Check safety.output_root in config."
        )

    if args.preflight:
        _preflight(cfg, raw_root)
        return

    log.info("=" * 60)
    pv = cfg.get("pipeline_version", PIPELINE_VERSION)
    log.info("Phase %s Standard DWI Preprocessing  v%s", pv, pv)
    log.info("Raw root   : %s", raw_root)
    log.info("Output root: %s", output_root)
    log.info("Sites      : %s", cfg["included_sites"])
    log.info("Resume     : %s", args.resume)
    log.info("=" * 60)

    records = run_batch_pipeline(
        raw_root=raw_root,
        output_root=output_root,
        sites=cfg["included_sites"],
        ip_qc_only=True,
        skip_if_exists=args.resume,
        config=cfg,
    )

    # Filter to requested subjects if specified
    if args.subjects:
        records = [r for r in records if r.subject_id in args.subjects]

    # Save summary CSVs
    subj_csv, site_csv = save_summary_csvs(records, output_root)

    # Final report
    n_ok   = sum(1 for r in records if r.status == "success")
    n_fail = sum(1 for r in records if r.status == "failed")
    n_warn = sum(r.warning_count for r in records)

    log.info("=" * 60)
    log.info("Phase 2R.1 complete: %d success / %d failed / %d warnings",
             n_ok, n_fail, n_warn)

    if n_fail:
        log.error("Failed subjects:")
        for r in records:
            if r.status == "failed":
                log.error("  [%s/%s] %s", r.site, r.subject_id, r.error_message)

    log.info("Subject CSV  → %s", subj_csv)
    log.info("Site CSV     → %s", site_csv)
    log.info("Run log      → %s", log_file)

    # Print backend summary from first completed subject
    backends_used: dict[str, set] = {
        "denoise": set(), "gibbs": set(), "eddy": set(), "bias": set()
    }
    for r in records:
        if r.status == "success":
            backends_used["denoise"].add(r.denoise_backend)
            backends_used["gibbs"].add(r.gibbs_backend)
            backends_used["eddy"].add(r.eddy_backend)
            backends_used["bias"].add(r.bias_backend)
    log.info("-" * 60)
    log.info("Backends used:")
    for step, vals in backends_used.items():
        log.info("  %-8s %s", step, ", ".join(sorted(vals)))

    skipped_steps = set()
    for r in records:
        skipped_steps.update(r.steps_skipped)
    if skipped_steps:
        log.warning(
            "Steps skipped across cohort: %s — "
            "document as pipeline limitations in Methods section.",
            ", ".join(sorted(skipped_steps))
        )

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
