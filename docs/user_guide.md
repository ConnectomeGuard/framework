# ConnectomeGuard — User Guide

This guide covers environment setup, ABIDE-II data acquisition, running the
end-to-end connectome pipeline, and applying the four ConnectomeGuard validation
gates. It corresponds to **Paper 1** of the ConnectomeGuard program (ABIDE-II,
N = 229; see the [Roadmap](../ROADMAP.md)).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Data Acquisition — ABIDE-II](#3-data-acquisition--abide-ii)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Validation Gates](#5-validation-gates)
6. [QC and Monitoring](#6-qc-and-monitoring)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### Software

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.9 | All pipeline scripts |
| DiPy | ≥ 1.7.0 | FOD estimation, EuDX tractography |
| nibabel | ≥ 5.0 | NIfTI I/O |
| numpy | ≥ 1.24 | Numerical operations |
| scipy | ≥ 1.10 | Statistical tests |
| scikit-learn | ≥ 1.2 | Site-classification QC (harmonization gate) |
| neuroCombat | ≥ 0.2.12 | ComBat harmonization |
| matplotlib | ≥ 3.7 | Figures |

> **Note:** FSL and MRtrix3 are not required for the core pipeline. DiPy performs all
> DWI processing steps.

### Hardware

- RAM: ≥ 16 GB recommended (FOD estimation loads the full DWI volume).
- Storage: ≥ 100 GB free for the full ABIDE-II cohort.
- CPU: multi-core recommended (FOD estimation uses all available cores).

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/ConnectomeGuard/framework.git
cd framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install dipy nibabel numpy scipy scikit-learn neuroCombat matplotlib

# Verify
python -c "import dipy; print('DiPy', dipy.__version__)"
```

---

## 3. Data Acquisition — ABIDE-II

ABIDE-II data is publicly available from NITRC. Diffusion (DTI) data requires a free
NITRC account.

### Step 1: Register at NITRC

Go to [nitrc.org/projects/abide_ii](https://www.nitrc.org/projects/abide_ii) and
request access to the diffusion (DTI) data package.

### Step 2: Download the DTI-eligible sites

Of the 19 ABIDE-II acquisition cohorts, five acquired usable DTI and form the
analytical cohort (N = 229). Download these sites:

| Site | Institution | N | ASD / Control |
|------|-------------|---|---------------|
| BNI    | Barrow Neurological Institute | 58 | 29 / 29 |
| NYU_1  | NYU Langone (cohort 1) | 55 | 33 / 22 |
| NYU_2  | NYU Langone (cohort 2) | 19 | 19 / 0 |
| SDSU_1 | San Diego State University | 57 | 33 / 24 |
| TCD_1  | Trinity College Dublin | 40 | 20 / 20 |

> **Excluded at the cohort-curation gate:** 13 cohorts acquired no DTI at the protocol
> level, and site IP_1 was excluded for an implausible whole-brain mean FA
> (0.667, 25.8σ above the analytical-site mean). See the full inventory in
> Supplementary Table S1 of the paper.

### Step 3: Organize raw data

```
data/raw/abide2/
├── BNI/
│   ├── 29006/
│   │   ├── dwi.nii.gz
│   │   ├── dwi.bval
│   │   └── dwi.bvec
│   └── ...
├── NYU_1/
├── NYU_2/
├── SDSU_1/
└── TCD_1/
```

### Step 4: Audit the dataset before processing

```bash
python scripts/validate_dataset.py --site BNI
python scripts/run_phase40a_phenotype_audit.py
```

This emits a per-subject audit (modality availability, DTI bval/bvec consistency,
phenotype completeness) with a documented, reproducible reason for every
inclusion/exclusion — the first ConnectomeGuard gate.

---

## 4. Running the Pipeline

The analysis runs as a sequence of per-stage scripts over the five analytical sites
(`BNI NYU_1 NYU_2 SDSU_1 TCD_1`). Most stage scripts accept `--sites`, `--resume`,
and `--dry-run`.

```bash
# 1. DWI preprocessing (denoising validation, brain masking, eddy)
python scripts/run_standard_dwi_preprocessing.py \
    --config configs/phase2r_standard_dwi_preprocessing.yaml --resume

# 2. Tensor estimation (FA, MD, AD, RD)
python scripts/run_tensor_estimation_v2.py --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1

# 3. Fiber orientation (CSA) and deterministic tractography
python scripts/run_fod_preparation_v2b.py       --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1
python scripts/run_streamline_generation_v2b.py --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1

# 4. Connectome construction (Schaefer-100) and QC
python scripts/run_connectome_builder_v2b.py    --sites BNI NYU_1 NYU_2 SDSU_1 TCD_1
python scripts/run_connectome_qc_harmonization.py
```

Single-site or single-subject runs use the same scripts with a narrowed
`--sites`/`--subjects` argument (e.g. `--sites BNI`).

### Configuration

Tractography and atlas parameters are set in the stage YAML configs under `configs/`.
The canonical settings are:

```yaml
tractography:
  n_seeds: 100000       # 100k seeds (canonical)
  step_size: 0.5        # mm
  max_angle: 30.0       # degrees

atlas:
  name: schaefer_100
  n_parcels: 100
```

---

## 5. Validation Gates

The four ConnectomeGuard gates run at the corresponding pipeline stages:

| Gate | Script | What it checks |
|------|--------|----------------|
| 1. Cohort curation | `scripts/validate_dataset.py`, `run_phase40a_phenotype_audit.py` | Modality/DTI availability and phenotype completeness per subject, with a reproducible exclusion reason. |
| 2. Preprocessing validation | `run_standard_dwi_preprocessing.py` | Directional-variance criterion that flags denoising failures (e.g., Patch2Self) without needing ground-truth FA. |
| 3. Tractography completeness | `scripts/run_seed_saturation_analysis.py` | Edge recovery as a function of seed count (seed saturation). |
| 4. Harmonization evaluation | `scripts/run_combat_evaluation.py` | Paired Logistic Regression + Random Forest site-classification, detecting the Zero-Fingerprint Effect that a linear-only check misses. |

```bash
# Gate 3 — tractography completeness (seed saturation)
python scripts/run_seed_saturation_analysis.py

# Gate 4 — harmonization / Zero-Fingerprint evaluation
python scripts/run_combat_evaluation.py
```

Once the gates pass, biomarker discovery runs on the ComBat-corrected features at the
edge and network-pair level:

```bash
python scripts/run_biomarker_discovery.py --perms 1000
python scripts/run_phase42_network_aggregation.py
```

---

## 6. QC and Monitoring

Each stage writes a QC report (JSON) and a completion marker; re-running a stage skips
already-completed subjects. To summarize connectome QC across the cohort:

```bash
python scripts/run_connectome_qc_harmonization.py
```

Key QC quantities to check:

- **Connectome density** per subject and per site (count connectomes are ~95%
  zero-inflated by construction).
- **Site-classification balanced accuracy** (Random Forest vs. Logistic Regression):
  a large RF–LR gap on count edges is the Zero-Fingerprint signature.
- **Age-signal preservation** (MD–age correlation) before and after harmonization.

> **Edge counts use upper-triangle unique pairs** (`np.triu(matrix, k=1) > 0`). Counting
> the full symmetric matrix double-counts every edge.

---

## 7. Troubleshooting

### EuDX edge-list vs. full matrix

EuDX outputs a **3-column edge-list** (`parcel_a, parcel_b, count`), 1-indexed — not a
full 100×100 matrix. Load it correctly:

```python
import numpy as np

def load_edgelist(path, n=100):
    m = np.zeros((n, n))
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis]
    for row in data:
        if np.any(np.isnan(row)):
            continue
        i, j, v = int(row[0]) - 1, int(row[1]) - 1, row[2]   # 1-indexed
        if 0 <= i < n and 0 <= j < n:
            m[i, j] += v
            m[j, i] += v
    return m
```

### Missing brain mask

If a brain mask is unavailable, derive one from the FA map:

```python
import nibabel as nib
import numpy as np

fa   = nib.load("FA.nii.gz")
mask = (fa.get_fdata() > 0.05).astype(np.uint8)   # typical range 0.03–0.10
nib.save(nib.Nifti1Image(mask, fa.affine), "brain_mask.nii.gz")
```

### Reproducibility

All analyses that require randomness use fixed seeds (`numpy` `default_rng(42)`,
scikit-learn `random_state=42`, permutation tests `np.random.seed(42)`). The full
computational environment is recorded in Supplementary Table S8 of the paper.
