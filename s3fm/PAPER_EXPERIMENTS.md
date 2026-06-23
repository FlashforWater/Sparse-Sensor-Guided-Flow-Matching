# Paper-Scale Experiment Plan

This document defines the paper-scale KSE training and evaluation run. It is
separate from the pilot configs so the smaller local checks remain reproducible.

## 1. Paper-Scale Data And Training Defaults

The paper configs use:

```text
system: KSE
Nx: 64
dt: 0.25
n_steps: 1600
warmup_steps: 400
n_trajectories: 128
window_length: 20
stride: 4
split: 70% train / 15% val / 15% test by trajectory
```

The split is trajectory-level, so train, validation, and test windows do not
come from the same trajectories.

Models trained for the paper run:

```text
experiments/paper/kse_prior/final.pt                 S3FM-Gauss control
experiments/paper/kse_prior_info/final.pt            S3FM-Info prior
experiments/paper/kse_score_prior/best.pt            S3GM-PF-ODE attribution baseline
experiments/paper/kse_source_inference/best.pt       q_phi(y,H) source inference
```

The source-inference model is trained across the full paper sparsity set:

```text
observed_fractions: 0.05, 0.1, 0.15, 0.3
```

## 2. Train Paper Models On Server

From the `s3fm/` project root:

```bash
DRY_RUN=1 DEVICE=cuda EXPERIMENT_ROOT=experiments/paper \
bash scripts/train_paper_models.sh
```

If the printed commands look correct, launch training:

```bash
nohup env DEVICE=cuda EXPERIMENT_ROOT=experiments/paper \
bash scripts/train_paper_models.sh > train_paper_models.log 2>&1 &
```

Monitor:

```bash
tail -f train_paper_models.log
```

If the server is constrained, reduce only training steps first:

```bash
nohup env DEVICE=cuda EXPERIMENT_ROOT=experiments/paper \
GAUSS_STEPS=20000 INFO_STEPS=20000 SCORE_STEPS=20000 SOURCE_STEPS=20000 \
bash scripts/train_paper_models.sh > train_paper_models.log 2>&1 &
```

## 3. Run Paper Evaluation Suite

After all four checkpoints exist, run:

```bash
nohup env DEVICE=cuda \
OUT=results/paper_server_suite \
WINDOWS=256 \
MASK_SEEDS=0,1,2 \
SOURCE_SEEDS=0,1,2 \
OBS_FRACTIONS=0.05,0.1,0.15,0.3 \
NFE=10,20,50,100 \
bash scripts/run_paper_suite.sh > paper_server_suite.log 2>&1 &
```

Monitor:

```bash
tail -f paper_server_suite.log
```

The suite is resume-friendly. Completed sub-runs are skipped unless
`OVERWRITE=1` is passed.

## 4. Outputs To Use In The Paper Draft

Primary aggregate files:

```text
results/paper_server_suite/README.md
results/paper_server_suite/aggregate_by_mask.csv
results/paper_server_suite/commands.jsonl
```

Important per-case files:

```text
results/paper_server_suite/obs_<fraction>/mask_<seed>/m4b_learned/summary.csv
results/paper_server_suite/obs_<fraction>/mask_<seed>/m4b_learned/transport_metrics.json
results/paper_server_suite/obs_<fraction>/mask_<seed>/pf_val_sweep/best_pf_by_nfe.csv
results/paper_server_suite/obs_<fraction>/mask_<seed>/pf_test_val_tuned/summary.csv
```

## 5. Paper-Level Success Criteria

The result is strong enough for the KSE main table if:

- all expected cases complete;
- learned-source S3FM beats Gaussian-source S3FM at matched NFE;
- learned-source S3FM beats its unguided variant at matched NFE;
- learned-source S3FM beats validation-tuned S3GM-PF-ODE at matched NFE;
- the above holds across mask seeds and observation fractions;
- transport diagnostics show learned/informative sources are target-short
  relative to Gaussian or marginal sources.

If these conditions fail at difficult sparsity such as `0.05`, report the
failure honestly and present the valid sparsity range.
