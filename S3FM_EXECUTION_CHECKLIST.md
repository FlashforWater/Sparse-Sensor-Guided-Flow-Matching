# S3FM Execution Checklist

This checklist tracks implementation progress. Detailed definitions and acceptance criteria are in `S3FM_CORE_SPEC.md`.

## Research gate

- [ ] Confirm intended target: scientific-ML/application paper or general ML methods paper.
- [ ] Confirm the implementation repository and writable workspace.
- [ ] Confirm access to KSE data or a reproducible KSE generator.
- [ ] Confirm access to an S3GM baseline or enough information to reproduce it.

## Milestone 0: repository audit

- [ ] Inspect repository structure and git status.
- [ ] Record environment and accelerator information.
- [ ] Locate reusable datasets, normalization, models, and metrics.
- [ ] Record the current tensor convention.
- [ ] Run one existing baseline or document the blocker.
- [ ] Produce an audit note.

## Milestone 1: KSE data

- [ ] Create leakage-free train/validation/test splits.
- [ ] Verify physical time step and spatial resolution.
- [ ] Implement normalization and exact inverse normalization.
- [ ] Implement sparse/regular measurement operators.
- [ ] Verify `y = H(X_ref)` with synthetic alignment tests.
- [ ] Reproduce a reference KSE visualization and metric.

## Milestone 2: unconditional flow prior

- [ ] Implement Gaussian source sampling.
- [ ] Implement linear conditional flow path.
- [ ] Implement velocity-matching loss.
- [ ] Train a small smoke-test model.
- [ ] Train the pilot KSE velocity model.
- [ ] Implement unguided Euler sampling.
- [ ] Verify endpoint identity with oracle velocity.
- [ ] Evaluate unguided sample statistics.
- [ ] Save checkpoint, config, normalization, and source revision.

## Milestone 3: single-window guidance

- [ ] Implement endpoint estimate `X1_hat = Xs + (1-s)v`.
- [ ] Implement observation energy.
- [ ] Implement `g_cov-G` without detaching the model.
- [ ] Add one-dimensional guidance-sign test.
- [ ] Confirm one guided step reduces energy.
- [ ] Reconstruct a single KSE window.
- [ ] Sweep guidance strength.
- [ ] Sweep NFE `{10, 20, 50, 100}`.
- [ ] Log residuals, norms, runtime, NFE, and memory.

## Milestone 4: long observed sequence

- [ ] Implement window splitting and global index map.
- [ ] Add exact assembly tests.
- [ ] Implement normalized overlap energy.
- [ ] Apply stop-gradient to the earlier-window anchor.
- [ ] Generate all observed windows jointly.
- [ ] Verify sequence energy reduces overlap seams.
- [ ] Verify exact final target length.

## Milestone 5: autoregressive future

- [ ] Implement initial-frame energy.
- [ ] Keep autoregressive energy separate from observation/sequence energy.
- [ ] Roll the latest `T_init` frames into the next window.
- [ ] Forecast beyond the observed interval.
- [ ] Report pointwise and statistical error growth.
- [ ] Generate multiple futures from different source seeds.

## Milestone 6: baseline comparison

- [ ] Run matched S3GM baseline.
- [ ] Confirm identical data, observations, metrics, and hardware.
- [ ] Count true model evaluations for every solver.
- [ ] Measure synchronized wall-clock inference time.
- [ ] Run at least three seeds.
- [ ] Create quality-versus-NFE plot.
- [ ] Create quality-versus-runtime plot.
- [ ] Create observation-residual plot.
- [ ] Create overlap-discontinuity plot.

## Milestone 7: stronger tasks

- [ ] Fourier KSE observation.
- [ ] Observation noise sweep.
- [ ] Kolmogorov flow reconstruction and energy spectrum.
- [ ] ERA5 missing-variable reconstruction.
- [ ] Nonlinear velocity-magnitude observation.
- [ ] Time-averaged observation.

## Milestone 8: additional novelty

- [ ] Specify residual-adaptive guidance before final experiments.
- [ ] Specify adaptive-NFE rule before final experiments.
- [ ] Add fixed-compute ablation.
- [ ] Verify improvement across more than one sensor setting.
- [ ] Add failure-case analysis.

## Paper artifacts

- [ ] Final method diagram.
- [ ] Algorithm pseudocode.
- [ ] Main speed-accuracy table.
- [ ] Main runtime table.
- [ ] KSE reconstruction figure.
- [ ] Long-sequence seam comparison.
- [ ] Uncertainty figure.
- [ ] Ablation table.
- [ ] Limitations section.
- [ ] Reproducibility appendix.

## Claim gate

- [ ] Every result includes dataset, `H`, noise, solver, NFE, hardware, and seeds.
- [ ] No cross-system zero-shot claim.
- [ ] No physics-consistency claim without a measured physics residual.
- [ ] No speedup claim without synchronized matched timing.
- [ ] No “same accuracy” claim without a predefined tolerance.

## First prompt to give an implementation AI

```text
Read S3FM_CORE_SPEC.md and perform only Milestone 0.

Audit the repository before editing anything. Identify reusable S3GM data generation, normalization, Video U-Net, score-model, sampler, and evaluation components. Record the current tensor convention and distinguish physical time from generative time. Run the smallest existing test or baseline that does not require new downloads. Do not implement Flow Matching yet.

Deliver an audit report containing:
1. repository map;
2. reusable modules;
3. missing components;
4. environment and dependency risks;
5. proposed minimal file-level implementation plan for Milestones 1-3;
6. exact commands used and their results.
```
