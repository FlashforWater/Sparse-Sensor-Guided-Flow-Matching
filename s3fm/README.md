# S3FM

Sparse-sensor-guided flow matching for spatiotemporal dynamics reconstruction.

Research project. The authoritative design docs live one level up
(`../S3FM_项目总览.md` for the overview, `../S3FM_AI_IMPLEMENTATION_SPEC.md`
for the implementation spec). This repository is the code.

## Status

Milestone-gated build. Current: **M0 (reproducibility foundation) complete.**

| Milestone | What | Gate |
|---|---|---|
| M0 ✅ | seeds, config, logging, smoke test | same seed → identical run |
| M1 | KSE data pipeline + measurement operators | norm round-trip < 1e-6, no leakage |
| M2 | linear flow-matching prior | endpoint oracle test, tiny-overfit |
| M3 | unguided ODE sampling + stats | plausible KSE samples at 10/20/50/100 NFE |
| M4 | single-window g_cov-G guidance (S3FM-Gauss) | guidance lowers residual & nRMSE |

(M4b informative source and beyond come after M4 plumbing is verified.)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the M0 gate

```bash
pytest -q                                  # all tests
python -m s3fm.smoke --seed 0 --out experiments/smoke_run   # deterministic smoke
```

Each run writes a `resolved_config.yaml` and `metrics.jsonl` into its output
directory. Two runs with the same seed produce identical metrics (modulo
wall-clock time).

## Conventions (non-negotiable)

- Physical frame index is `frame_idx` / `n`; flow-integration time is
  `flow_time` / `s`. **Never share one variable for both.**
- NFE counts every call to the velocity model, not solver steps.
- Source sample `x_source` / `Z0`; clean target `x_target` / `Z1`;
  intermediate `x_s`; endpoint estimate `x1_hat`.
