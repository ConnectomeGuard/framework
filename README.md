# ConnectomeGuard

## About the ConnectomeGuard framework

**ConnectomeGuard** is a validation framework for multi-site diffusion-MRI connectomics. It
wraps a connectome-construction pipeline and screens it for the failure modes that silently
invalidate downstream analyses, through **four sequential validation gates**:

1. **Cohort curation** — systematic modality inventory and site-level QC before any subject is processed.
2. **Preprocessing validation** — detection of denoising/preprocessing failures that depend on acquisition parameters.
3. **Tractography completeness** — quantification of edge recovery as a function of seed count (seed saturation).
4. **Harmonization evaluation** — paired linear/nonlinear (Logistic Regression + Random Forest) assessment of batch correction.

The framework's central finding is the **Zero-Fingerprint Effect**: site-wise z-scoring of
sparse streamline-count connectomes maps structural zeros to site-specific constants,
generating perfect-accuracy acquisition-site fingerprints that are invisible to the standard
linear-classifier harmonization check — and are therefore easy to miss.

The **pipeline** is the framework's reference implementation: an end-to-end diffusion-MRI
workflow (preprocessing → tensor estimation → tractography → connectome construction →
biomarker analysis) at whose stages the four gates run.

---

## Why it's useful for researchers

Multi-site diffusion-MRI studies tend to fail *quietly*: a cohort may be missing the modality
you assume it has, a denoiser can corrupt anisotropy at certain acquisition parameters,
tractography can undersample the edge set, and a harmonization step can *look* successful while
encoding acquisition site into every zero-valued edge. Any one of these silently invalidates
downstream biomarker analysis. ConnectomeGuard makes these failure modes visible **before** you
draw conclusions.

- **Avoid false confidence in harmonization.** The field's standard check — "a linear classifier can no longer predict site" — can pass even when site is fully recoverable by a nonlinear model (the Zero-Fingerprint Effect). The paired linear/nonlinear evaluation catches leakage the standard check misses.
- **Interpret null results correctly.** After validation, a null association means *underpowered but valid*, not *undetected pipeline failure* — a different, defensible conclusion. The framework also reports effect sizes and power estimates for planning adequately-powered studies.
- **Reproducible curation.** Every subject/site exclusion is emitted with a specific, documented, reproducible reason, so another researcher reaches the same analytical cohort *by construction* — directly targeting a common source of non-replicability.
- **Pipeline- and diagnosis-agnostic gates.** The gates are drop-in: the directional-variance denoising check needs no ground-truth FA and works for any denoiser; the seed-saturation protocol works for any tractography algorithm; the dual-classifier harmonization evaluation works for any batch-correction method. They apply to any multi-site tractography dataset and any condition (autism, schizophrenia, aging, TBI, normative).
- **Open and citable.** Fully open-source (MIT) with an archived release and dataset, so results are reproducible and individual gates can be adopted piecemeal or as a whole.

---

## Pipeline (framework component)

The reference pipeline processes raw multi-site DTI data (ABIDE-II, `canonical_v1`)
through nine stages:

1. Skull stripping and motion correction
2. Tensor estimation (FA, MD)
3. CSA-ODF fiber-orientation estimation (DiPy)
4. EuDX deterministic tractography (100k seeds)
5. Atlas registration (Schaefer-100, subject space)
6. ConnectomeGuard QC gates (Zero-Fingerprint, harmonization)

External validation on independent cohorts (e.g., HCP Young Adult) is planned for a
later phase of the program (see the [Roadmap](ROADMAP.md)).

---

## Roadmap

Phase-by-phase program — research → publication → open-source release (full plan: **[ROADMAP.md](ROADMAP.md)**):

| Paper | Focus | Status |
|-------|-------|--------|
| 1 | ConnectomeGuard validation framework + Zero-Fingerprint Effect (**this repo**) | 🟡 Under submission |
| 2 | Advanced algorithms + HCP external validation | ⬜ Future project |
| 3 | Biomarker discovery (adequately powered) | ⬜ Planned |

---

## Structure

```
neurofiber/
├── preprocessing/   ← DTI preprocessing (eddy, topup wrappers)
├── tensor/          ← FA/MD tensor estimation
├── tractography/    ← EuDX wrapper + bootstrap
├── connectome/      ← Connectome matrix construction
├── harmonization/   ← Within-site z-scoring (Zero-Fingerprint)
├── biomarkers/      ← Age-signal metrics
├── registry/        ← Subject/dataset registry
└── utils/           ← IO, logging, QC utilities

configs/             ← Pipeline YAML configurations
scripts/             ← per-stage pipeline + validation-gate run scripts (see Usage)
```

---

## Usage

### ABIDE-II pipeline (canonical_v1)

The ABIDE-II analysis runs as a sequence of per-stage scripts over the five
analytical sites (`BNI NYU_1 NYU_2 SDSU_1 TCD_1`). Most stage scripts accept
`--sites`, `--resume`, and `--dry-run`.

```bash
# 0. Audit and validate the dataset before processing
python scripts/validate_dataset.py --site BNI
python scripts/run_phase40a_phenotype_audit.py

# 1. DWI preprocessing (denoising validation, brain masking, eddy)
python scripts/run_standard_dwi_preprocessing.py \
    --config configs/phase2r_standard_dwi_preprocessing.yaml --resume

# 2. Tensor estimation (FA, MD, AD, RD)
python scripts/run_tensor_estimation_v2.py --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1

# 3. Fiber orientation (CSA) and deterministic tractography
python scripts/run_fod_preparation_v2b.py        --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1
python scripts/run_streamline_generation_v2b.py  --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1

# 4. Connectome construction (Schaefer-100) and QC
python scripts/run_connectome_builder_v2b.py     --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1
python scripts/run_connectome_qc_harmonization.py

# 5. ConnectomeGuard validation gates
python scripts/run_seed_saturation_analysis.py   # tractography completeness
python scripts/run_combat_evaluation.py          # harmonization / Zero-Fingerprint

# 6. Biomarker discovery (ComBat-corrected; edge + network-pair)
python scripts/run_biomarker_discovery.py --perms 1000
python scripts/run_phase42_network_aggregation.py
```

Single-site or single-subject runs use the same scripts with a narrowed
`--sites`/`--subjects` argument (e.g. `--sites BNI`).

External validation on independent cohorts is planned for a future phase of the
program (see the [Roadmap](ROADMAP.md)).

---

## Requirements

- Python >= 3.9
- DiPy >= 1.7.0
- nibabel >= 5.0
- numpy, scipy, scikit-learn, matplotlib

---

## Documentation

- [FAQ](FAQ.md) — common questions about the framework, validation gates, scope, and usage
- [User Guide](docs/user_guide.md) — framework stages, configuration, validation gates, and analysis results
- [Citation metadata](CITATION.cff) — machine-readable citation (CFF format)

## Citation

If you use ConnectomeGuard, please cite:

M. M. Islam, M. S. Uddin, and M. H. Ali, "ConnectomeGuard: A validation framework for multi-site diffusion MRI connectomics (v1.0-canonical)," Zenodo, 2025. doi: [10.5281/zenodo.20812181](https://doi.org/10.5281/zenodo.20812181)

```bibtex
@software{islam_connectomeguard_2025,
  author    = {Islam, Md. Maksudul and Uddin, Mohammad Shorif and Ali, Mohammad Hanif},
  title     = {ConnectomeGuard: A validation framework for multi-site diffusion MRI connectomics},
  version   = {v1.0-canonical},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20812181}
}
```

## License

MIT License
