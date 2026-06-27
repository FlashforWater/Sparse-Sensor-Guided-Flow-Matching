# S3FM: Informative-Source Guided Flow Matching

Sparse-sensor spatiotemporal reconstruction with Flow Matching, measurement
guidance, and observation-informed low-bandwidth source initialization.

This repository contains a research implementation of **S3FM** for reconstructing
complete spatiotemporal fields from sparse sensor observations. The current main
benchmark is the 1D Kuramoto-Sivashinsky equation (KSE).

## Overview

Given sparse observations

```text
y = H(X) + noise
```

S3FM reconstructs the full field `X` by integrating a guided Flow Matching ODE.
The core idea is to avoid starting reconstruction from pure Gaussian noise. The
main S3FM-Info variant starts from an observation-informed low-bandwidth source:

```text
Z0 = q_phi(y, H) + eta
```

where `q_phi(y,H)` predicts a low-bandwidth approximation to the target field.
The flow prior then fills in the missing details, while measurement guidance
keeps the reconstruction consistent with the observed sensors.

## Current Result Snapshot

The current KSE server suite evaluates:

```text
4 observation fractions x 3 mask seeds x 4 NFE budgets = 48 matched-NFE comparisons
```

Summary from the completed run:

```text
completed cases: 12/12
M4b learned-source pass cases: 12/12
PF-ODE attribution pass cases: 12/12
matched-NFE rows passed: 48/48
```

Overall normalized RMSE:

| Method | Mean nRMSE | Median nRMSE |
|---|---:|---:|
| S3FM learned guided | 0.0227 | 0.0119 |
| S3FM learned unguided | 0.0254 | 0.0143 |
| Gaussian-source guided FM | 0.3077 | 0.2459 |
| validation-tuned S3GM-PF-ODE | 68.1348 | 0.5328 |

Note: the PF-ODE mean is affected by a single unstable outlier; the median and
per-case pass/fail gates are the safer headline diagnostics.

## What This Supports

The completed KSE experiments support the following bounded conclusion:

> On in-distribution KSE sparse-sensor reconstruction, S3FM with learned
> low-bandwidth source initialization and measurement guidance consistently
> outperforms Gaussian-source guided Flow Matching and a validation-tuned S3GM
> probability-flow-ODE baseline at matched NFE.

The conclusion is intentionally scoped. The learned-source variant includes a
supervised source initializer `q_phi(y,H) -> S(X)`. It is not data leakage
because it is trained only on training trajectories and predicts only a
low-bandwidth projection, not the full target field. Still, it is a learned
component and should not be described as a purely unconditional prior.

## Repository Layout

```text
.
├── README.md
├── S3FM_项目总览.md
├── S3FM_论文草稿.md
├── S3FM_论文结果分析.md
├── S3FM_复现实验记录.md
├── S3FM_本地泛化小实验.md
└── s3fm/
    ├── configs/
    ├── scripts/
    ├── s3fm/
    │   ├── data/
    │   ├── diffusion/
    │   ├── flow/
    │   ├── guidance/
    │   ├── measurements/
    │   └── models/
    └── tests/
```

Important documents:

| File | Purpose |
|---|---|
| `S3FM_项目总览.md` | Chinese project overview and method explanation |
| `S3FM_论文草稿.md` | Paper draft |
| `S3FM_论文结果分析.md` | Server result analysis |
| `S3FM_复现实验记录.md` | Reproducibility record |
| `S3FM_本地泛化小实验.md` | Local warm/marginal source probe |
| `s3fm/PAPER_EXPERIMENTS.md` | Paper-scale experiment plan |
| `s3fm/SERVER_RUN.md` | Server run guide |

## Installation

```bash
cd s3fm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Run a deterministic smoke test:

```bash
python -m s3fm.smoke --seed 0 --out experiments/smoke_run
```

## Training

Paper-scale model training:

```bash
cd s3fm
bash scripts/train_paper_models.sh
```

This trains:

```text
experiments/paper/kse_prior/final.pt
experiments/paper/kse_prior_info/final.pt
experiments/paper/kse_score_prior/best.pt
experiments/paper/kse_source_inference/best.pt
```

For long server runs, use `screen`, `tmux`, or `nohup`.

## Evaluation

Run the paper evaluation suite after training:

```bash
cd s3fm
bash scripts/run_paper_suite.sh
```

The suite evaluates S3FM against Gaussian-source guided FM and the
validation-tuned S3GM-PF-ODE baseline over observation fractions, mask seeds,
and NFE budgets.

Main outputs:

```text
results/paper_server_suite/README.md
results/paper_server_suite/aggregate_by_mask.csv
results/paper_server_suite/commands.jsonl
```

## Additional Controls

Warm-source mechanism check:

```bash
bash scripts/run_warm_source_suite.sh
```

Direct supervised reconstruction baseline:

```bash
bash scripts/train_direct_reconstruction.sh
bash scripts/run_direct_reconstruction_eval.sh
```

These controls are important for separating:

```text
learned source initialization
observation-informed non-learned source construction
measurement guidance
direct y,H -> X supervised reconstruction
```

## Generated Artifacts

Generated outputs should not be committed:

```text
s3fm/experiments/
s3fm/results/
server_downloads/
s3fm_server_upload/
```

The repository `.gitignore` excludes checkpoints, experiment outputs, local
virtual environments, PDFs, and local cache directories.

## Citation / Status

This is an active research project. The current results support the KSE
same-distribution setting; cross-system generalization and direct reconstruction
comparisons are being added as follow-up controls.
