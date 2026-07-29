"""
NeuroFiber — Phase 2R.1 vs Phase 2.3 Preprocessing QC Comparison

Reads:
  v1: data/processed/abide_ii/<site>/<cohort>/<subj>/session_1/dti_1/tensor/tensor_fit_report.json
      data/processed/abide_ii/<site>/<cohort>/<subj>/session_1/dti_1/correction_report.json
  v2: data/processed_v2/abide_ii/<site>/<cohort>/<subj>/session_1/preprocessing_report.json

Outputs:
  data/processed_v2/phase2r_1_vs_v1_qc_comparison.csv
  research_notes/phase_02R_1_v1_v2_comparison.md

Usage
─────
  python scripts/compare_v1_v2_preprocessing.py

Options:
  --processed-v1   Path to v1 processed root (default: data/processed/abide_ii)
  --processed-v2   Path to v2 processed root (default: data/processed_v2/abide_ii)
  --output-csv     Override output CSV path
  --output-md      Override output MD path
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ai_pipeline"))

log = logging.getLogger("compare_v1_v2")

SITES = ["bni", "ip", "nyu1", "nyu2", "sdsu", "tcd"]
SITE_DISPLAY = {
    "bni":  "BNI",
    "ip":   "IP_1",
    "nyu1": "NYU_1",
    "nyu2": "NYU_2",
    "sdsu": "SDSU_1",
    "tcd":  "TCD_1",
}

CSV_FIELDS = [
    "site",
    "subject_id",
    # v1 fields
    "v1_fa_mean",
    "v1_fa_std",
    "v1_volume_count",
    "v1_status",
    # v2 fields
    "v2_status",
    "v2_denoise_backend",
    "v2_volume_count_original",
    "v2_volume_count_corrected",
    "v2_steps_completed",
    "v2_steps_skipped",
    "v2_warning_count",
    "v2_skipped_gibbs",
    "v2_skipped_eddy",
    "v2_skipped_bias",
    # comparison
    "volume_count_delta",
    "v1_only",
    "v2_only",
]


def _find_subjects(site_root: Path) -> dict[str, Path]:
    """Return {subject_id: session_path} for all subjects under a site root."""
    subjects: dict[str, Path] = {}
    if not site_root.exists():
        return subjects
    for cohort_dir in sorted(site_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        for subj_dir in sorted(cohort_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            session_dir = subj_dir / "session_1"
            if session_dir.is_dir():
                subjects[subj_dir.name] = session_dir
    return subjects


def _load_v1(session_path: Path) -> dict:
    """Load v1 tensor_fit_report and correction_report for a subject."""
    out = {
        "v1_fa_mean":       None,
        "v1_fa_std":        None,
        "v1_volume_count":  None,
        "v1_status":        "missing",
    }
    tensor_path = session_path / "dti_1" / "tensor" / "tensor_fit_report.json"
    correction_path = session_path / "dti_1" / "correction_report.json"

    if tensor_path.exists():
        try:
            d = json.loads(tensor_path.read_text())
            out["v1_fa_mean"]  = d.get("fa_mean")
            out["v1_fa_std"]   = d.get("fa_std")
            out["v1_status"]   = d.get("status", "unknown")
        except Exception:
            out["v1_status"] = "parse_error"

    if correction_path.exists():
        try:
            d = json.loads(correction_path.read_text())
            out["v1_volume_count"] = d.get("corrected_volume_count")
        except Exception:
            pass

    return out


def _load_v2(session_path: Path) -> dict:
    """Load v2 preprocessing_report.json for a subject."""
    out = {
        "v2_status":                "missing",
        "v2_denoise_backend":       None,
        "v2_volume_count_original": None,
        "v2_volume_count_corrected": None,
        "v2_steps_completed":       "",
        "v2_steps_skipped":         "",
        "v2_warning_count":         None,
        "v2_skipped_gibbs":         False,
        "v2_skipped_eddy":          False,
        "v2_skipped_bias":          False,
    }
    report_path = session_path / "dti_1" / "preprocessing_report.json"
    if not report_path.exists():
        return out

    try:
        d = json.loads(report_path.read_text())
        out["v2_status"]                  = d.get("status", "unknown")
        out["v2_denoise_backend"]         = d.get("denoise_backend")
        out["v2_volume_count_original"]   = d.get("original_volume_count")
        out["v2_volume_count_corrected"]  = d.get("corrected_volume_count")
        out["v2_warning_count"]           = d.get("warning_count", 0)

        steps_done = d.get("steps_completed", "")
        steps_skip = d.get("steps_skipped", "")
        if isinstance(steps_done, list):
            steps_done = "|".join(steps_done)
        if isinstance(steps_skip, list):
            steps_skip = "|".join(steps_skip)

        out["v2_steps_completed"] = steps_done
        out["v2_steps_skipped"]   = steps_skip
        out["v2_skipped_gibbs"]   = "step3_gibbs" in steps_skip
        out["v2_skipped_eddy"]    = "step4_eddy"  in steps_skip
        out["v2_skipped_bias"]    = "step5_bias"  in steps_skip
    except Exception:
        out["v2_status"] = "parse_error"

    return out


def build_comparison(
    v1_root: Path,
    v2_root: Path,
) -> tuple[list[dict], dict]:
    """Build per-subject rows and per-site summary."""
    rows: list[dict] = []
    site_stats: dict[str, dict] = {}

    for site in SITES:
        display = SITE_DISPLAY[site]
        v1_subjects = _find_subjects(v1_root / site)
        v2_subjects = _find_subjects(v2_root / site)
        all_subjects = sorted(set(v1_subjects) | set(v2_subjects))

        stats = {
            "site":               display,
            "total_subjects":     len(all_subjects),
            "v1_only":            0,
            "v2_only":            0,
            "both":               0,
            "v2_success":         0,
            "v2_failed":          0,
            "v2_missing":         0,
            "denoise_backends":   defaultdict(int),
            "skipped_gibbs":      0,
            "skipped_eddy":       0,
            "skipped_bias":       0,
        }

        for subj in all_subjects:
            in_v1 = subj in v1_subjects
            in_v2 = subj in v2_subjects

            v1 = _load_v1(v1_subjects[subj]) if in_v1 else {
                "v1_fa_mean": None, "v1_fa_std": None,
                "v1_volume_count": None, "v1_status": "missing",
            }
            v2 = _load_v2(v2_subjects[subj]) if in_v2 else {
                "v2_status": "missing",
                "v2_denoise_backend": None,
                "v2_volume_count_original": None,
                "v2_volume_count_corrected": None,
                "v2_steps_completed": "",
                "v2_steps_skipped": "",
                "v2_warning_count": None,
                "v2_skipped_gibbs": False,
                "v2_skipped_eddy": False,
                "v2_skipped_bias": False,
            }

            # Volume delta: v2_corrected vs v1_corrected
            v1_vol = v1["v1_volume_count"]
            v2_vol = v2["v2_volume_count_corrected"]
            vol_delta = None
            if v1_vol is not None and v2_vol is not None:
                vol_delta = v2_vol - v1_vol

            row = {
                "site":       display,
                "subject_id": subj,
                **v1,
                **v2,
                "volume_count_delta": vol_delta,
                "v1_only": in_v1 and not in_v2,
                "v2_only": not in_v1 and in_v2,
            }
            rows.append(row)

            # Stats
            if in_v1 and in_v2:
                stats["both"] += 1
            elif in_v1:
                stats["v1_only"] += 1
            else:
                stats["v2_only"] += 1

            st = v2["v2_status"]
            if st == "success":
                stats["v2_success"] += 1
            elif st == "missing":
                stats["v2_missing"] += 1
            else:
                stats["v2_failed"] += 1

            if v2["v2_denoise_backend"]:
                stats["denoise_backends"][v2["v2_denoise_backend"]] += 1
            if v2["v2_skipped_gibbs"]:
                stats["skipped_gibbs"] += 1
            if v2["v2_skipped_eddy"]:
                stats["skipped_eddy"] += 1
            if v2["v2_skipped_bias"]:
                stats["skipped_bias"] += 1

        site_stats[site] = stats

    return rows, site_stats


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written → %s  (%d rows)", out_path.relative_to(REPO_ROOT), len(rows))


def write_markdown(rows: list[dict], site_stats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_v2_success = sum(s["v2_success"] for s in site_stats.values())
    total_v2_missing = sum(s["v2_missing"] for s in site_stats.values())
    total_v2_failed  = sum(s["v2_failed"]  for s in site_stats.values())
    total_subjects   = sum(s["total_subjects"] for s in site_stats.values())
    total_skip_gibbs = sum(s["skipped_gibbs"] for s in site_stats.values())
    total_skip_eddy  = sum(s["skipped_eddy"]  for s in site_stats.values())
    total_skip_bias  = sum(s["skipped_bias"]  for s in site_stats.values())

    # Per-site summary table rows
    site_table_rows = []
    for site_key in SITES:
        s = site_stats[site_key]
        backends = ", ".join(
            f"{b}×{n}" for b, n in sorted(s["denoise_backends"].items())
        ) or "—"
        site_table_rows.append(
            f"| {s['site']} | {s['total_subjects']} | {s['v2_success']} | "
            f"{s['v2_failed']} | {s['v2_missing']} | {backends} | "
            f"{s['skipped_gibbs']} | {s['skipped_eddy']} | {s['skipped_bias']} |"
        )

    # Volume delta stats for subjects in both
    deltas = [r["volume_count_delta"] for r in rows
              if r["volume_count_delta"] is not None]
    nonzero_deltas = [d for d in deltas if d != 0]
    delta_summary = (
        f"{len(nonzero_deltas)} subjects had a volume count change "
        f"(v2_corrected − v1_corrected ≠ 0)"
        if nonzero_deltas else
        "All matched subjects have identical volume counts between v1 and v2."
    )

    # FA mean stats for v1 (where available)
    fa_vals = [r["v1_fa_mean"] for r in rows if r["v1_fa_mean"] is not None]
    if fa_vals:
        fa_mean_all = round(sum(fa_vals) / len(fa_vals), 4)
        fa_min      = round(min(fa_vals), 4)
        fa_max      = round(max(fa_vals), 4)
        fa_summary  = f"Mean FA (v1, across all subjects): {fa_mean_all} (range {fa_min}–{fa_max}, N={len(fa_vals)})"
    else:
        fa_summary = "No v1 FA values available yet."

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md = f"""# Phase 2R.1 vs Phase 2.3 (v1) Preprocessing QC Comparison

**Generated:** {now}
**Source CSV:** `data/processed_v2/phase2r_1_vs_v1_qc_comparison.csv`

---

## Summary

| Metric | Value |
|--------|-------|
| Total subjects (all DTI sites) | {total_subjects} |
| v2 success | {total_v2_success} |
| v2 failed | {total_v2_failed} |
| v2 missing (not yet run) | {total_v2_missing} |
| Subjects with Gibbs skipped | {total_skip_gibbs} |
| Subjects with eddy skipped | {total_skip_eddy} |
| Subjects with bias skipped | {total_skip_bias} |

{fa_summary}

{delta_summary}

> **Gate:** Do not proceed to tensor estimation until v2 success count matches total expected subjects per site and warning_count ≤ 1 per subject (IP_1 warnings expected and accepted).

---

## Per-Site Summary

| Site | Total | v2 Success | v2 Failed | v2 Missing | Denoise Backend | Gibbs Skipped | Eddy Skipped | Bias Skipped |
|------|-------|-----------|-----------|------------|----------------|--------------|-------------|-------------|
{chr(10).join(site_table_rows)}

---

## Preprocessing Steps Coverage

### Step Skip Counts

- **Gibbs ringing correction** (step3): {total_skip_gibbs}/{total_subjects} skipped
  - Expected when MRtrix3 (`mrdegibbs`) is not installed
- **Eddy/motion correction** (step4): {total_skip_eddy}/{total_subjects} skipped
  - Expected when `pe_dir=null` in config (phase-encoding direction not yet confirmed per site)
- **Bias-field correction** (step5): {total_skip_bias}/{total_subjects} skipped
  - Expected when MRtrix3 (`dwibiascorrect`) is not installed

### Interpretation

Skips for Gibbs, eddy, and bias correction are expected in the current environment
(MRtrix3 and FSL not yet installed; `pe_dir` not yet confirmed). These steps are
recorded explicitly in each subject's `preprocessing_report.json` — nothing is silently
omitted. The MP-PCA denoising step (step2) is the primary noise-reduction contribution
of Phase 2R.1.

---

## v1 vs v2 Volume Count Comparison

Differences arise when step1 (volume correction) strips a trailing non-DWI volume that
v1 retained. A delta of −1 is expected and correct. A delta > 1 or positive delta
warrants investigation.

| Delta | Count |
|-------|-------|
{_vol_delta_table(rows)}

---

## v1 FA Mean by Site (Phase 2.3 reference)

These are v1 values computed on uncorrected/minimally preprocessed data.
After Phase 2R.1 tensor estimation (Phase 2R.2), compare new FA means — denoising
should reduce noise floor and modestly increase white-matter FA coherence.

| Site | N (v1) | FA Mean | FA Std |
|------|--------|---------|--------|
{_fa_by_site_table(rows)}

---

## Next Steps

1. Confirm Phase 2R.1 is complete (v2 success == expected subjects per site).
2. Review warning logs for any unexpected skips or data issues.
3. Install MRtrix3 ≥ 3.0.4 and FSL to unlock Gibbs, eddy, and bias correction.
4. Confirm phase-encoding direction per site; set `pe_dir` in config.
5. Re-run Phase 2R.1 with MRtrix3/FSL available — compare FA mean shift.
6. **Only after verification above:** proceed to Phase 2R.2 (tensor estimation).
"""
    out_path.write_text(md)
    log.info("Markdown written → %s", out_path.relative_to(REPO_ROOT))


def _vol_delta_table(rows: list[dict]) -> str:
    from collections import Counter
    counts = Counter(
        r["volume_count_delta"]
        for r in rows if r["volume_count_delta"] is not None
    )
    if not counts:
        return "| (no matched subjects yet) | — |"
    lines = []
    for delta in sorted(counts):
        lines.append(f"| {delta:+d} | {counts[delta]} |")
    return "\n".join(lines)


def _fa_by_site_table(rows: list[dict]) -> str:
    from collections import defaultdict
    site_fa: dict[str, list[float]] = defaultdict(list)
    site_fa_std: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["v1_fa_mean"] is not None:
            site_fa[r["site"]].append(r["v1_fa_mean"])
        if r["v1_fa_std"] is not None:
            site_fa_std[r["site"]].append(r["v1_fa_std"])

    lines = []
    for site_key in SITES:
        disp = SITE_DISPLAY[site_key]
        vals = site_fa[disp]
        stds = site_fa_std[disp]
        if vals:
            mean_fa  = round(sum(vals) / len(vals), 4)
            mean_std = round(sum(stds) / len(stds), 4) if stds else "—"
            lines.append(f"| {disp} | {len(vals)} | {mean_fa} | {mean_std} |")
        else:
            lines.append(f"| {disp} | 0 | — | — |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare v1 vs v2 preprocessing outputs.")
    p.add_argument("--processed-v1", default=str(REPO_ROOT / "data/processed/abide_ii"))
    p.add_argument("--processed-v2", default=str(REPO_ROOT / "data/processed_v2/abide_ii"))
    p.add_argument("--output-csv",   default=str(REPO_ROOT / "data/processed_v2/phase2r_1_vs_v1_qc_comparison.csv"))
    p.add_argument("--output-md",    default=str(REPO_ROOT / "research_notes/phase_02R_1_v1_v2_comparison.md"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    v1_root = Path(args.processed_v1)
    v2_root = Path(args.processed_v2)
    out_csv = Path(args.output_csv)
    out_md  = Path(args.output_md)

    log.info("v1 root: %s", v1_root)
    log.info("v2 root: %s", v2_root)

    rows, site_stats = build_comparison(v1_root, v2_root)

    log.info("Total rows: %d", len(rows))
    for site_key in SITES:
        s = site_stats[site_key]
        log.info(
            "[%s]  total=%d  v2_success=%d  v2_missing=%d  v2_failed=%d  "
            "gibbs_skipped=%d  eddy_skipped=%d",
            s["site"], s["total_subjects"], s["v2_success"],
            s["v2_missing"], s["v2_failed"],
            s["skipped_gibbs"], s["skipped_eddy"],
        )

    write_csv(rows, out_csv)
    write_markdown(rows, site_stats, out_md)

    log.info("Done.")


if __name__ == "__main__":
    main()
