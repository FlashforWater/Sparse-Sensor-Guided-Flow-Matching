# S3FM Start Here

This is the short entry point for implementing the S3FM research project with AI assistance.

The authoritative specification is:

- [`S3FM_AI_IMPLEMENTATION_SPEC.md`](./S3FM_AI_IMPLEMENTATION_SPEC.md)

If this file and the full specification disagree, follow the full specification and update this file.

## Current Status

- Research direction defined.
- **Headline contribution: S3FM-Info — transport from an informative (non-Gaussian) source under general dependent-coupling flow guidance.** Gaussian-source S3FM is the controlled baseline.
- Main guidance selected: `g_cov-G`.
- First benchmark selected: KSE.
- First target: quality near S3GM with ≤ 50 ODE NFE, **and** beating a probability-flow-ODE-of-S3GM baseline at matched NFE.
- No implementation or experimental result should be assumed yet.
- Current milestone: **M0 - repository and reproducibility foundation**.

## Fixed Initial Decisions

```text
Source (headline, S3FM-Info): informative field S(X)+eta (coarse-blur / spectral-truncation)
Source (control, S3FM-Gauss): independent standard Gaussian
Coupling: dependent for S3FM-Info (P ~= 1 regime), independent for S3FM-Gauss
Flow path: linear affine path
Primary model: spatiotemporal Video U-Net velocity model
Primary guidance: g_cov-G
First measurement operator: sparse mask
First dataset: KSE
First comparison: S3FM-Info vs S3FM-Gauss vs S3GM-PF-ODE at 10/20/50/100 NFE, vs S3GM at 1,000 steps
```

> Note: the Gaussian-source + linear-path + independent-coupling setting is diffusion-equivalent up to schedule, so it is a baseline, not the contribution. Build it first for plumbing (M0–M4), then introduce the informative source at M4b.

## Execution Order

Do not skip gates.

```text
M0 Reproducibility and configuration
  ↓ gate: deterministic smoke test
M1 KSE data pipeline
  ↓ gate: split and normalization tests
M2 Unconditional flow prior
  ↓ gate: oracle endpoint and tiny-overfit tests
M3 Unguided distribution evaluation
  ↓ gate: statistically plausible samples
M4 Single-window sparse guidance (S3FM-Gauss)
  ↓ gate: guidance improves residual and nRMSE
M4b Informative-source prior (S3FM-Info) — headline
  ↓ gate: beats S3FM-Gauss at matched NFE; measured shorter transport
M5 Parallel overlapping windows
  ↓ gate: reduced seam error
M6 Autoregressive forecasting
  ↓ gate: stable horizon/error plots
M7 Speed-quality benchmark (incl. S3GM-PF-ODE attribution baseline)
  ↓ gate: measured accuracy-speed result, gain not from SDE→ODE alone
M8 Ablations
  ↓
M9 Additional systems
```

## Prompt for the First AI Session

```text
Read docs/S3FM_START_HERE.md and docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

Work only on M0: repository and reproducibility foundation.

First inspect the repository, existing environment, and user changes. Propose the smallest code structure consistent with the existing project. Implement deterministic seed handling, configuration loading with resolved-config saving, and lightweight experiment-result logging. Add a CPU smoke test proving that two runs with the same seed match within numerical tolerance.

Do not implement KSE generation, neural networks, flow matching, guidance, overlapping windows, or experiments yet.

Preserve unrelated changes. Run focused tests. End with the completion-report format from Section 16 of the full specification.
```

## Prompt for M1 After M0 Passes

```text
Read docs/S3FM_START_HERE.md and docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

M0 has passed. Work only on M1: KSE data pipeline.

Inspect any existing KSE files or generation code before creating new implementations. Establish immutable train/validation/test splits at trajectory level, reversible normalization, fixed physical-time metadata, and window extraction. Add tests for split leakage, dimensions, frame indexing, and normalization round-trip error below 1e-6. Produce a small visualization or numerical inspection of clean samples.

Do not train a model or implement guidance. End with the completion-report format from Section 16.
```

## Prompt for M2 After M1 Passes

```text
Read docs/S3FM_START_HERE.md and docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

M0 and M1 have passed. Work only on M2: unconditional linear-path flow-matching prior.

Implement the linear path Zs=(1-s)Z0+sZ1, velocity target Z1-Z0, endpoint estimate X1_hat=Zs+(1-s)v_theta(Zs,s), and flow-matching loss. Use the existing project architecture where possible. Add analytical tests for the path, derivative, and endpoint. First overfit a tiny batch; do not launch an expensive full training run without showing the tiny-overfit result and estimated compute cost.

Do not implement sensor guidance or long-sequence windows. End with the completion-report format from Section 16.
```

## Rule for Every Later AI Session

Copy the relevant milestone and acceptance criteria from the full specification into the prompt. Ask the AI to stop after that milestone and report:

```text
Milestone
Outcome
Files changed
Mathematical conventions
Tests and results
Configuration
Metrics/runtime
Known risks
Next permitted step
```

## The Three Most Important Correctness Checks

1. **Endpoint:** with oracle velocity, `X1_hat` must recover the clean target.
2. **Guidance direction:** a small guidance-only step must reduce the defined energy.
3. **Time semantics:** physical frames and flow integration time must never share the same variable.

