# Server Run Guide

This guide is for running the S3FM attribution suite on a CUDA server. Run all
commands from this directory, the `s3fm/` project root. For the full
paper-scale train-then-evaluate workflow, see `PAPER_EXPERIMENTS.md`.

## 1. Preflight

Activate the project environment, then confirm CUDA and checkpoints:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

ls experiments/kse_prior/final.pt
ls experiments/kse_prior_info/final.pt
ls experiments/kse_score_prior/best.pt
ls experiments/kse_source_inference/best.pt
```

The four default checkpoints are:

```text
experiments/kse_prior/final.pt
experiments/kse_prior_info/final.pt
experiments/kse_score_prior/best.pt
experiments/kse_source_inference/best.pt
```

If the server stores checkpoints elsewhere, override them with `GAUSS_CKPT`,
`INFO_CKPT`, `SCORE_CKPT`, and `SOURCE_INFERENCE_CKPT`.

## 2. Optional Dry Run

Use this to verify command structure without launching the expensive suite:

```bash
DEVICE=cuda \
OUT=results/server_suite_dryrun \
WINDOWS=128 \
MASK_SEEDS=0,1,2 \
SOURCE_SEEDS=0,1,2 \
OBS_FRACTIONS=0.15 \
NFE=10,20,50,100 \
DRY_RUN=1 \
bash scripts/run_server_suite.sh
```

Dry-run output is only for command inspection. It does not replace the real
suite.

## 3. Recommended Formal Run

This is the first formal server run:

```bash
DEVICE=cuda \
OUT=results/server_suite \
WINDOWS=128 \
MASK_SEEDS=0,1,2 \
SOURCE_SEEDS=0,1,2 \
OBS_FRACTIONS=0.15 \
NFE=10,20,50,100 \
bash scripts/run_server_suite.sh
```

For a disconnected server session, use `nohup`:

```bash
nohup env DEVICE=cuda OUT=results/server_suite WINDOWS=128 MASK_SEEDS=0,1,2 SOURCE_SEEDS=0,1,2 OBS_FRACTIONS=0.15 NFE=10,20,50,100 bash scripts/run_server_suite.sh > server_suite.log 2>&1 &
```

Monitor progress:

```bash
tail -f server_suite.log
```

## 4. Full Observation-Fraction Sweep

Run this after the single-fraction suite looks correct:

```bash
nohup env DEVICE=cuda \
OUT=results/server_suite_full \
WINDOWS=128 \
MASK_SEEDS=0,1,2 \
SOURCE_SEEDS=0,1,2 \
OBS_FRACTIONS=0.05,0.1,0.15,0.3 \
NFE=10,20,50,100 \
bash scripts/run_server_suite.sh > server_suite_full.log 2>&1 &
```

This is about four times the work of the `OBS_FRACTIONS=0.15` run.

## 5. What The Suite Runs

For each observed fraction and mask seed, `s3fm.run_server_suite` runs:

1. `m4b_learned`: learned-source S3FM test ablation.
2. `pf_val_sweep`: S3GM-PF-ODE lambda sweep on validation.
3. `pf_test_val_tuned`: S3GM-PF-ODE test with validation-selected lambdas.
4. Aggregation into a compact CSV and README.

The runner is resume-friendly. If a sub-run already has its required outputs,
it is skipped unless `--overwrite` is passed.

The shell launcher also accepts these optional environment flags:

```text
DRY_RUN=1     print commands without running the suite
OVERWRITE=1   re-run completed sub-runs
SKIP_M4B=1    skip the learned-source S3FM ablation
SKIP_PF=1     skip the PF-ODE validation/test baseline
```

## 6. Outputs To Check

For the recommended run:

```text
results/server_suite/README.md
results/server_suite/aggregate_by_mask.csv
results/server_suite/commands.jsonl
```

For the full sweep:

```text
results/server_suite_full/README.md
results/server_suite_full/aggregate_by_mask.csv
results/server_suite_full/commands.jsonl
```

The directory layout is:

```text
results/server_suite/
  obs_0p15/
    mask_0/
      m4b_learned/
      pf_val_sweep/
      pf_test_val_tuned/
    mask_1/
    mask_2/
  aggregate_by_mask.csv
  README.md
  commands.jsonl
```

## 7. Success Criteria

The run is usable for the paper draft when:

- `README.md` reports the expected number of completed cases.
- `M4b learned-source pass cases` is all cases.
- `val-tuned PF attribution pass cases` is all cases.
- `aggregate_by_mask.csv` has one row for every
  `observed_fraction x mask_seed x NFE`.
- In `aggregate_by_mask.csv`, these comparisons hold at matched NFE:
  - `s3fm_learned_guided_nrmse <= gauss_guided_nrmse`
  - `s3fm_learned_guided_nrmse <= s3fm_learned_unguided_nrmse`
  - `s3fm_learned_guided_nrmse <= pf_val_tuned_guided_nrmse`

## 8. Checkpoint Override Example

```bash
GAUSS_CKPT=/path/to/kse_prior/final.pt \
INFO_CKPT=/path/to/kse_prior_info/final.pt \
SCORE_CKPT=/path/to/kse_score_prior/best.pt \
SOURCE_INFERENCE_CKPT=/path/to/kse_source_inference/best.pt \
DEVICE=cuda \
OUT=results/server_suite \
bash scripts/run_server_suite.sh
```
