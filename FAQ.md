# ConnectomeGuard — Frequently Asked Questions

Common questions about the ConnectomeGuard validation framework and its reference pipeline.
For stage-by-stage detail see the [User Guide](docs/user_guide.md); for the program plan see the
[Roadmap](ROADMAP.md).

---

## About the framework

### What is ConnectomeGuard?
ConnectomeGuard is a **validation framework for multi-site diffusion-MRI connectomics**. It wraps a
connectome-construction pipeline and screens it for the failure modes that silently invalidate
downstream analyses, through four sequential validation gates (see below). It ships with a full
reference pipeline (preprocessing → tractography → connectome construction → validation →
biomarker analysis) applied to ABIDE-II (N = 229, five sites).

### What problem does it solve?
Multi-site diffusion-MRI studies tend to fail *quietly*. A cohort may be missing a modality you
assumed it had; a denoiser can corrupt anisotropy at certain acquisition parameters; tractography
can undersample the edge set; and a harmonization step can *look* successful while encoding
acquisition site into every zero-valued edge. Any one of these silently invalidates biomarker
analysis. ConnectomeGuard makes these failures **visible before you draw conclusions**.

### What is the Zero-Fingerprint Effect?
It is the framework's central finding. Structural connectomes built from streamline counts are
sparse — most edges are zero. When site-wise z-scoring is applied, every structural zero maps to a
site-specific constant (`−mean_site / sd_site`), because site means and standard deviations differ.
The result is a **perfect-accuracy acquisition-site fingerprint** that a nonlinear classifier
recovers from as few as ten edges — while a *linear* classifier returns near-chance accuracy,
exactly mimicking successful harmonization. Because the field's standard check is a linear
classifier, the leakage is easy to miss. The effect is structural, not a sampling artifact, and it
survives ComBat and other site-wise affine corrections.

### What are the four validation gates?
1. **Cohort curation** — systematic modality inventory and site-level QC before any subject is processed.
2. **Preprocessing validation** — detection of denoising/preprocessing failures that depend on acquisition parameters.
3. **Tractography completeness** — quantification of edge recovery as a function of seed count (seed saturation).
4. **Harmonization evaluation** — paired linear + nonlinear (Logistic Regression + Random Forest) assessment of batch correction, which is what catches the Zero-Fingerprint Effect.

---

## Scope and applicability

### Is this only for autism / ABIDE-II?
No. ABIDE-II (N = 229) is the reference dataset used to develop and demonstrate the framework, but
the gates are **diagnosis- and dataset-agnostic**. They apply to any multi-site tractography dataset
and any condition — autism, schizophrenia, aging, TBI, or normative cohorts.

### Do I have to run the whole pipeline, or can I use individual gates?
Either. The gates are designed as **drop-in checks**:
- the denoising-validation check needs no ground-truth FA and works for any denoiser;
- the seed-saturation protocol works for any tractography algorithm;
- the dual-classifier harmonization evaluation works for any batch-correction method.

You can adopt one gate against your own pipeline, or run the whole reference pipeline end to end.

### What does the reference pipeline actually run?
The ABIDE-II reference pipeline (mode `canonical_v1`) runs, per subject: skull-stripping and motion
correction → tensor estimation (FA, MD) → fiber-orientation estimation → EuDX deterministic
tractography → atlas registration (Schaefer-100, subject space) → connectome construction → the
ConnectomeGuard validation gates → biomarker analysis. It also includes an **HCP Young-Adult
external-validation** mode used in the program's later phase.

### How is the analytical cohort decided?
By **reproducible curation**: every subject or site exclusion is emitted with a specific,
documented, machine-checkable reason. Another researcher reaches the same analytical cohort *by
construction*, which directly targets a common source of non-replicability.

### How should I interpret a null result after validation?
As **underpowered but valid**, not *undetected pipeline failure* — a different and defensible
conclusion. The framework reports effect sizes and power estimates so you can plan an adequately
powered study rather than over-interpret a null.

---

## Using it

### What are the requirements?
- Python ≥ 3.9
- DiPy ≥ 1.7.0, nibabel ≥ 5.0, numpy, scipy, scikit-learn
- boto3 and AWS credentials with ConnectomeDB HCP Open-Access permissions (only if you run HCP external validation)

### How do I run it?
The pipeline is a sequence of per-stage scripts under `scripts/`, most accepting `--sites`,
`--resume`, and `--dry-run`. A typical ABIDE-II run starts with `scripts/validate_dataset.py` and
proceeds through preprocessing, tensor estimation, tractography, connectome construction, the
validation gates (`run_seed_saturation_analysis.py`, `run_combat_evaluation.py`), and biomarker
discovery. See the [README Usage section](README.md#usage) and the [User Guide](docs/user_guide.md)
for the full command sequence.

### Can I run a single site or a single subject?
Yes — the same scripts accept a narrowed `--sites` / `--subjects` argument (e.g. `--sites BNI`).

### Where are the results and the data?
Pipeline results are written under `results/`; HCP external-validation results under `results/hcp/`.
The archived code and dataset are on Zenodo (see Citation).

---

## Project, licensing, and citation

### How does this relate to the "advanced algorithms" and later papers?
This repository is Phase 1 of the ConnectomeGuard program — the validation framework and the
Zero-Fingerprint Effect. Advanced connectome-construction algorithms and the HCP external-validation
analysis are part of **forthcoming publications** and are intentionally not documented here. See the
[Roadmap](ROADMAP.md).

### Is it open source?
Yes — **MIT License**. You may adopt individual gates or the whole framework, commercially or
academically, with attribution.

### How do I cite it?
> M. M. Islam and M. S. Uddin, "ConnectomeGuard: A validation framework for multi-site diffusion MRI
> connectomics (v1.0-canonical)," Zenodo, 2025. doi:
> [10.5281/zenodo.20812181](https://doi.org/10.5281/zenodo.20812181)

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

### How do I report a bug or ask a question?
Open an issue on the repository. Please include your Python/DiPy versions, the stage script and
command you ran, and the relevant log output so the problem can be reproduced.
