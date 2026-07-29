# ConnectomeGuard — Research & Publication Roadmap

A phase-by-phase research program on reliable, reproducible multi-site
diffusion-MRI connectomics, organized as **three papers**. Each phase moves
through **research → publication → open-source release**. **This repository is
Phase 1** (the ConnectomeGuard validation framework); later phases are
summarized here and released as each paper is published.

**Legend:** ✅ Complete · 🟡 In progress · ⬜ Planned / future

## Summary

| Paper | Focus | Research | Publication | Open-source release |
|-------|-------|----------|-------------|---------------------|
| **1** | ConnectomeGuard validation framework + the Zero-Fingerprint Effect | ✅ complete | 🟡 under submission | ✅ this repository |
| **2** | Advanced connectome-construction algorithms + HCP external validation | 🟡 in progress | ⬜ planned | ⬜ future (separate repo) |
| **3** | Biomarker discovery (adequately powered) | ⬜ planned | ⬜ planned | ⬜ future |

---

## Phase 1 — ConnectomeGuard validation framework + Zero-Fingerprint Effect ✅ (this repository)
- Multi-site DTI preprocessing and deterministic tractography pipeline.
- **Four validation gates:** cohort curation, preprocessing validation, tractography completeness (seed saturation), and harmonization evaluation.
- **The Zero-Fingerprint Effect** — a formal characterization of how site-wise z-scoring of sparse count connectomes leaks acquisition-site identity, together with a linear/nonlinear evaluation criterion that detects it.
- Demonstrated on ABIDE-II (N = 229, 5 sites).
- **Research:** complete · **Publication:** under submission · **Open source:** this repository.

## Phase 2 — Advanced algorithms + HCP external validation ⬜ (future project)
- Next-generation tractography and confidence-weighting algorithms (in active development).
- External validation on Human Connectome Project Young Adult data as a normative, diagnosis-independent baseline and a generalization test of the framework.
- **Research:** in progress · **Publication:** planned · **Open source:** to be released as a **separate repository** together with the paper. Methodological details and code are **withheld pending publication**.

## Phase 3 — Biomarker discovery ⬜ (planned)
- Adequately-powered biomarker discovery (N ≈ 500–800, per the power/effect-size estimates established in Phase 1), applying the validated framework to autism and — because the framework is diagnosis-agnostic — to other conditions (e.g., schizophrenia, aging, traumatic brain injury).
- **Research:** planned · **Publication:** planned · **Open source:** future.

---

*This roadmap is a high-level plan and will evolve; later-phase details and code are released as each paper is published.*
