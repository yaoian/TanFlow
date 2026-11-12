# TanFlow: Closure-Preserving Velocity Flow Matching for Trajectory Gap Repair

[![ICDE 2027](https://img.shields.io/badge/ICDE%202027-Under%20Review-blue.svg)](#)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24163449.svg)](https://doi.org/10.5281/zenodo.24163449)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

This repository provides the official implementation, reproduction scripts, and theoretical proofs for **TanFlow** (under review at IEEE ICDE 2027).

TanFlow formulates trajectory gap recovery as flow matching on the codimension-2 affine velocity displacement manifold $\mathcal{M}_{\Delta p}$, ensuring exact endpoint closure without open-loop drift while recovering interior motion under 8-NFE deterministic Heun integration.

Full proofs of Propositions 1–3 and the Prefix-Integration Affine Bijection Lemma are provided in [`theory/proofs.md`](theory/proofs.md).

---

## 1. Environment & Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- PyYAML >= 6.0

Install dependencies via:
```bash
pip install -r requirements.txt
```
For exact pinned reproduction versions, use:
```bash
pip install -r requirements-freeze.txt
```

Hardware: Model training and throughput benchmarks target an NVIDIA GPU (e.g., RTX 3090, 24 GB VRAM) using `--device cuda:0 --batch-size 256`. CPU mode is supported for quick tests and verification (`--device cpu`).

---

## 2. Quick Start (Smoke Demo)

To verify the training and evaluation pipeline on synthetic data in seconds:

```bash
# Run Table 1 training and evaluation demo
bash scripts/reproduce_table1.sh --demo

# Run all ablation arms demo
bash scripts/reproduce_ablation.sh --demo
```

Alternatively, run the step-by-step CLI demo from the repository root:
```bash
# 1. Generate synthetic data (train/val/test = 20/7/9 windows)
mkdir -p results/demo
python3 -m scripts.release_run demo-data --output results/demo/data.npz

# 2. Run a 4-step smoke training; writes best.pt and complete.json
python3 -m scripts.release_run train \
    --data results/demo/data.npz \
    --output results/demo/run \
    --dataset atc --arm TanFlow-Full --gap 13 --seed 0 \
    --steps 4 --eval-every 2 --batch-size 8 --smoke

# 3. Evaluate the validation-best checkpoint
python3 -m scripts.release_run evaluate \
    --data results/demo/data.npz \
    --checkpoint results/demo/run/best.pt \
    --output results/demo/eval --allow-smoke
```

All three output paths must be new; remove `results/demo` before rerunning the demo.

---

## 3. Data Format, Schema & Download

### Benchmark Datasets Download

The preprocessed benchmark datasets (`atc.npz`, `indoor.npz`, `sdd.npz`, `edinburgh.npz`, `zara1.npz`, `zara2.npz`) are hosted on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24163449.svg)](https://doi.org/10.5281/zenodo.24163449)

To download and extract all benchmarks automatically into `data/`:
```bash
# Automated one-command download via helper script
bash scripts/download_data.sh
```

Or manually extract `tanflow_benchmarks.tar.gz` downloaded from [Zenodo (10.5281/zenodo.24163449)](https://doi.org/10.5281/zenodo.24163449):
```bash
tar -xzvf tanflow_benchmarks.tar.gz -C data/
```

Expected layout:
```text
TanFlow/
└── data/
    ├── atc.npz
    ├── indoor.npz
    ├── sdd.npz
    ├── edinburgh.npz
    ├── zara1.npz
    └── zara2.npz
```

`scripts/release_run.py` consumes one normalized NPZ per benchmark. Training opens only `train` and `val`; `test` is opened only by `evaluate`.

### Core arrays

| Key | Type / shape | Contract |
|---|---|---|
| `metadata` | 0-D JSON string array | JSON object described below |
| `train`, `val` | `float32 [N, 25, 3]` | Finite `(x, y, t)` windows; `t` strictly increasing along axis 1 |
| `test` | `float32 [N, 25, 3]` | Same contract; required for `evaluate` and never read by training |
| `<split>_entity` | 1-D string array, length `N` | Optional window-to-entity identity aligned row-for-row with `<split>`; `train_entity` and `val_entity` are required when `metadata.entity_disjoint = true` |

Windows are fixed at 25 points. `B_normalized` is the required `metadata.coordinates` tag, not a separate NPZ array. The centered missing gap `M ∈ {5, 9, 13, 17, 21}` is applied at load time; its missing index range is `[(25-M)//2, (25-M)//2 + M - 1]` inclusive (for example, indices 6–18 for `M = 13`). No mask or missing-value array is stored in the NPZ.

### `metadata` fields

Store the object as `np.array(json.dumps(meta))`; the loader uses `np.load(..., allow_pickle=False)`.

| Field | Contract |
|---|---|
| `dataset` | CLI key from the table below; for `train`, must equal `--dataset` |
| `coordinates` | Literal `"B_normalized"` |
| `normalization_fit` | Literal `"full_train"` |
| `mean_xy` | Two finite floats |
| `xy_scale` | Two finite positive floats, equal (isotropic) |
| `time_scale` | Finite positive float |
| `entity_disjoint` | Optional boolean. If `true`, every pair of splits loaded together must provide `<split>_entity`, with no shared entity (training loads `train` and `val` together) |

Normalization is frozen from the full training split and reused unchanged at evaluation: `x_norm = (x - mean_xy[0]) / xy_scale[0]`, `y_norm = (y - mean_xy[1]) / xy_scale[1]`, `t_norm = t / time_scale`. Extra metadata keys are ignored.

### Split semantics

- `train`: model updates and source/calibrator fitting.
- `val`: validation ADE and checkpoint selection (`best.pt`).
- `test`: held-out evaluation only. Training never reads it; checkpoint-based `evaluate` verifies the checkpoint and input hashes before opening it.

### Dataset CLI keys

Paper names are not accepted by the CLI. Use the exact, case-sensitive key below, and set `metadata.dataset` to the same value. In particular, ICMTD maps to `indoor`; there is no automatic alias resolution.

| Paper benchmark | `--dataset` / `metadata.dataset` |
|---|---|
| ATC | `atc` |
| ICMTD (Indoor concourse) | `indoor` |
| SDD | `sdd` |
| Edinburgh | `edinburgh` |
| Zara1 | `zara1` |
| Zara2 | `zara2` |
| WorldTrace (transfer) | `worldtrace_chengdu`, `worldtrace_guangzhou`, `worldtrace_shanghai` |

The accepted keys are exactly the `DATASETS` tuple in `scripts/multigap_data.py`.

Standalone `train` requires both `--data <path>` and `--dataset <key>`. Standalone `evaluate` accepts `--data <path>` but no `--dataset`; the dataset key remains in `metadata.dataset`. The reproduction wrappers accept `--data <key>=<path>` and forward the key to training as `--dataset`.

---

## 4. Reproducing Paper Experiments

### Table 1: Main Benchmarks
The paper evaluates six primary benchmarks: **ATC**, **ICMTD** (Indoor concourse), **SDD**, **Edinburgh**, **Zara1**, and **Zara2** across centered missing gaps $M \in \{5, 13, 21\}$.

```bash
# Run formal Table 1 benchmark (30,000 updates; 60,000 on SDD, batch size 256)
bash scripts/reproduce_table1.sh \
    --data atc=/path/to/atc.npz \
    --device cuda:0 \
    --output results/table1
```

### Table 2: Ablation Studies
Run controlled ablation variants (Prior formulations, Source distributions, Network architectures, etc.):
```bash
bash scripts/reproduce_ablation.sh \
    --data atc=/path/to/atc.npz \
    --device cuda:0 \
    --output results/ablation
```

### Downstream Spatiotemporal Queries (Table 4)
Evaluate range, $k\text{NN}$, and stay-duration queries on repaired trajectories:
```bash
# Print query protocol specification
python3 -m scripts.release_run query-protocol

# Evaluate query workloads on prediction archives
python3 -m scripts.release_run evaluate-queries \
    --predictions results/table1/atc_mid13_seed0_evaluation/mid13_predictions.npz \
    --positions /path/to/positions.npy \
    --times /path/to/times.npy \
    --workload /path/to/workload.json \
    --output results/query_eval
```

---

## 5. Model Architecture & Paper Model Variants

### 8-Channel Model Stem
Each interval $i \in \{0, \dots, N-1\}$ ($N=24$) feeds an 8-channel vector into `FlowTCN(width=64, in_channels=8)`:
$$X_{\tau,i} = \bigl[\underbrace{\alpha_i z_{\tau,i,x},\, \alpha_i z_{\tau,i,y}}_{\text{2-ch active state}},\; \underbrace{\Delta t_i,\, \alpha_i \rho_{\mathrm{sup},i},\, \alpha_i}_{\text{3-ch temporal/mask}},\; \underbrace{v^{\mathrm{obs}}_{i,x},\, v^{\mathrm{obs}}_{i,y},\, m_{\mathrm{obs},i}}_{\text{3-ch bilateral context } B(c)}\bigr]^\top \in \mathbb{R}^8$$
where Channel 4 is reserved ($\rho_{\mathrm{sup},i} \equiv 0$). The full model has 117,122 parameters.

### Paper Model Variants
The following are the only public model variants, exactly matching the main model and Table 2 ablations:

| Paper Variant (Table 2) | CLI Variant Name | Description |
|---|---|---|
| **TanFlow (Full)** | `TanFlow-Full` | Full 8-channel Flow-TCN with bilateral kinematic-prior center $V_K(c)$, bilateral observation context $B(c)$, tangent-space Gaussian Schur source, and stage-wise projection. |
| **w/o Stage-Wise Projection (Terminal Only)** | `Terminal-Only` | Trains an unprojected field, integrates without stage-wise tangent projection, and applies one terminal closure projection at $\tau=1$. |
| **w/o Any Projection (Unconstrained Flow)** | `Unconstrained-Flow` | Loads the Terminal Only checkpoint and integrates without stage-wise or terminal projection. This variant is evaluation-only. |
| **w/o Kinematic Prior (Chord Source)** | `Chord-Source` | Replaces the kinematic-prior center with the linear chord center $V_{\mathrm{chord}}(c)$. |
| **w/o Tangent Noise (Point Source, $\sigma=0$)** | `Point-Source` | Uses the degenerate point source with $\sigma=0$ (rank 0), without tangent-space noise. |
| **w/o Flow Transport (1-Step Direct TCN)** | `Direct-TCN` | Replaces four-step flow-matching transport with single-step direct residual regression. |
| **TanFlow (Transformer Backbone)** | `Transformer` | Replaces the Flow-TCN backbone with an equal-capacity Transformer encoder. |

---

## 6. Unit Tests

Run the test suite via pytest to verify model architectures, parameter counts, affine projections, and causal scopes:

```bash
pytest
```

---

## 7. Citation & License

This project is licensed under the MIT License - see [`LICENSE`](LICENSE) for details.

```bibtex
@misc{tanflow2027,
  title        = {TanFlow: Closure-Preserving Velocity Flow Matching for Trajectory Gap Repair},
  author       = {TanFlow Authors},
  year         = {2027},
  howpublished = {Manuscript under review at IEEE International Conference on Data Engineering (ICDE 2027)}
}
```
