# S3FM: Sparse-Sensor-Guided Flow Matching

## AI-Executable Research and Implementation Specification

**Document status:** Working specification, version 0.2  
**This document is the single source of truth.** It supersedes `IMPLEMENTATION/S3FM_CORE_SPEC.md` (now a pointer stub). If any other S3FM document disagrees with this one, this document wins and the other must be updated.

**Headline contribution (what makes this more than a sampler swap):** S3FM reconstructs spatiotemporal dynamics from sparse measurements by transporting from an **informative, non-Gaussian source distribution** (a cheap coarse/interpolated field) to the clean field, under **general flow-matching guidance that is valid for dependent couplings**. The short transport path this induces is what enables few-step (≤50 NFE) reconstruction; the operator-agnostic guidance is what preserves S3GM's zero-shot flexibility across measurement operators `H`.

**Primary goal:** Demonstrate that an informative-source guided flow-matching ODE reconstructs full spatiotemporal fields from sparse, noisy, incomplete, or nonlinear measurements at an order of magnitude fewer function evaluations than S3GM, *and* that the speedup is attributable to the method rather than merely to switching from an SDE to an ODE solver.

**First required result:** On the KSE benchmark, approach the reconstruction quality of the 1,000-step S3GM sampler with at most 50 ODE function evaluations, while beating a matched probability-flow-ODE-of-S3GM baseline at the same NFE budget.

> Caution on framing: with a **standard-Gaussian source + linear affine path + independent coupling**, flow matching is mathematically equivalent to a diffusion model up to noise schedule (see *On the Guidance of Flow Matching*, and §18 decision log). That configuration is therefore a **controlled baseline (S3FM-Gauss)**, not the contribution. The contribution lives in the informative-source / dependent-coupling regime (§6.0).

---

## 1. Instructions for Any AI Agent Using This Document

Before changing code, the agent must:

1. Read this entire document.
2. Inspect the existing repository and preserve unrelated user changes.
3. Identify the current milestone in Section 11.
4. Implement only that milestone unless the user explicitly expands the scope.
5. Add or update tests for every mathematical operator introduced.
6. Run the smallest relevant tests before expensive experiments.
7. Report exact commands, configuration, metrics, runtime, and unresolved risks.

The agent must not:

- claim speedups or accuracy improvements before measuring them;
- silently change the data split, normalization, grid, physical time step, or metric definitions;
- confuse physical time with flow integration time;
- train a conditional model on `(measurement, full field)` pairs unless explicitly requested;
- treat a new dynamical system as zero-shot generalization;
- combine the parallel-window equation and autoregressive equation into one update;
- use the old diffusion weights as flow-matching weights without a separately justified conversion method.
- **Additionally:** never attribute a measured speedup to "flow matching" without the probability-flow-ODE-of-S3GM baseline (§11 M7); never present the standard-Gaussian-source configuration as the contribution (it is the `S3FM-Gauss` controlled baseline, §6.0); never use a different source distribution at inference than the prior was trained to transport from (§6.0 train/test source consistency).

---

## 2. Research Question and Hypothesis

### 2.1 Research question

Can a pretrained flow-matching prior that transports from an **informative source distribution** reconstruct and forecast full spatiotemporal dynamics from sparse, noisy, incomplete, or nonlinear measurements substantially faster than S3GM, without retraining for each measurement operator `H`, *and with the speed advantage demonstrably caused by the shorter transport path rather than by ODE-vs-SDE integration alone*?

### 2.2 Primary hypothesis

A flow-matching prior whose **source distribution carries information about the target** (e.g. a coarse, blurred, or interpolated field) needs a much shorter and straighter transport than a Gaussian-source diffusion model, so the same operator-agnostic covariance-gradient guidance reaches S3GM-level reconstruction quality in roughly 20–100 ODE function evaluations instead of 1,000 SDE steps. Crucially, this advantage should persist *over a matched probability-flow-ODE-of-S3GM baseline at equal NFE*, isolating the contribution of the informative source from the contribution of deterministic integration.

### 2.3 Secondary hypotheses

1. An informative-source prior (S3FM-Info) attains a given reconstruction quality at strictly fewer NFE than the Gaussian-source prior (S3FM-Gauss), and the gap widens as the NFE budget shrinks.
2. General flow-matching guidance valid for **dependent couplings** (the reverse-coupling ratio `P`, *On the Guidance of Flow Matching* §3.1) lets the informative source and target be paired non-independently (e.g. each coarse field paired with its own fine field) without breaking the guidance correctness, under the stated `P ≈ 1` scope.
3. Parallel overlapping-window guidance can generate sequences longer than the training window without visible temporal seams.
4. Autoregressive endpoint guidance can extend predictions beyond the observed interval.
5. The speed advantage remains meaningful after including gradient computation through the velocity network.
6. Statistical features may remain accurate even when chaotic pointwise trajectories diverge.

### 2.4 Non-goals for the first paper version

- A universal foundation model across unrelated dynamical systems.
- Exact enforcement of governing PDEs.
- A theoretical proof of posterior exactness for arbitrary nonlinear `H`, or a new proof handling `P ≠ 1` dependent couplings (we operate in the `P ≈ 1` regime and cite the existing framework).
- Real-time performance before a correct KSE implementation exists.
- Reproducing every dataset from S3GM before the KSE pilot succeeds.

---

## 3. Source Papers

1. **S3GM:** *Learning spatiotemporal dynamics with a pretrained generative model*.
   - Reusable ideas: unconditional spatiotemporal prior, observation consistency, sequence consistency, parallel overlapping windows, autoregressive continuation, uncertainty from repeated sampling.
   - Main limitation addressed here: 1,000-step guided SDE sampling.

2. **Flow guidance:** *On the Guidance of Flow Matching*.
   - Reusable ideas: general guidance vector field (their Eq. 1) valid for **arbitrary source distribution and dependent coupling**, the endpoint estimate, `g_cov-G`, `g_cov-A`, and inverse-problem-specific guidance.
   - **Key enabling result for this project:** the guidance vector field carries a reverse-coupling ratio `P = π'(x0|x1)/π(x0|x1)`; `P = 1` exactly under independent coupling and is a reasonable approximation for mini-batch-OT-style dependent couplings. This is what theoretically licenses an informative, dependently-coupled source (§6.0) while reusing the same closed-form guidance. We operate inside the `P ≈ 1` scope and do not claim new theory for `P ≠ 1`.
   - Initial method choice for this project: `g_cov-G` as the main method, `g_cov-A` as a speed-oriented ablation, and inverse-specific guidance only for linear `H`.

### 3.1 Novelty boundary (what is and is not a contribution)

Replacing a diffusion sampler with flow matching is **not**, by itself, a methodological contribution — with a Gaussian source and linear path the two are equivalent up to schedule. A defensible paper must therefore demonstrate at least:

1. **Informative-source spatiotemporal flow prior:** transporting from a cheap structured field (coarse/blurred/interpolated) rather than Gaussian noise, with an explicit, measured shorter-transport advantage at low NFE.
2. **Operator-agnostic guidance under (possibly dependent) couplings** for spatiotemporal inverse problems, reusing the general flow-guidance framework.
3. **Long-sequence reconstruction** via jointly guided overlapping flow windows.
4. **A speed-accuracy study that isolates causation** — beating the probability-flow-ODE-of-S3GM baseline at matched NFE, not only the 1,000-step SDE.
5. At least one additional contribution, preferably **residual-adaptive guidance or adaptive NFE allocation**.

Do not claim a new general theory of flow guidance unless new proofs are actually developed. Do not claim cross-system zero-shot generalization.

---

## 4. Notation Contract

This notation is mandatory in code, comments, plots, and documentation.

| Symbol | Meaning |
|---|---|
| `t_phys` or `n` | Physical time or physical frame index |
| `s` | Flow integration time, increasing from `0` to `1` |
| `i` | Window index |
| `X` | A clean full spatiotemporal sample |
| `Z0` | Source sample. **Gaussian noise in S3FM-Gauss; an informative structured field (e.g. `S(X)`) in S3FM-Info** (§6.0) |
| `Z1` | Target clean sample; during training `Z1 = X` |
| `S(.)` | Source-construction operator producing the informative source from a clean field (e.g. coarse-blur-downsample-upsample); used in S3FM-Info |
| `pi(Z0, Z1)` | Coupling (joint law of source and target pairs); independent in S3FM-Gauss, dependent in S3FM-Info |
| `Zs` | Intermediate flow state at flow time `s` |
| `v_theta(Zs, s)` | Learned flow velocity |
| `X1_hat` | Estimated clean endpoint from the current flow state |
| `y` | Measurements from the current real or simulated sensors |
| `H` | Known differentiable measurement operator |
| `T` | Number of physical frames in one model window |
| `T_total` | Desired total physical sequence length |
| `m` | Number of overlapping physical frames between adjacent windows |
| `T_init` | Number of initial frames constraining an autoregressive window |
| `NFE` | Number of velocity-network function evaluations during ODE sampling |

Never use the same symbol `t` for both physical time and flow time in implementation documents.

---

## 5. Problem Formulation

The unknown clean sequence is:

```text
X in R^[T_total, C, spatial dimensions]
```

Measurements follow:

```text
y = H(X) + measurement_noise
```

`H` may represent:

- sparse point sampling;
- regular downsampling;
- missing physical variables;
- Fourier-space measurement;
- time averaging;
- velocity magnitude `sqrt(u^2 + v^2)`;
- combinations of the above.

The desired posterior is conceptually:

```text
p(X | y) proportional to p_prior(X) * exp(-J_obs(X))
```

For Gaussian measurement noise:

```text
J_obs(X) = 0.5 / sigma_y^2 * ||y - H(X)||_2^2
```

In code, default to a **mean squared residual**, not a raw sum, so guidance scale does not change only because the number of observations changes. If reproducing a paper formula that uses a sum, make the reduction configurable and record it.

---

## 6. Method Specification

### 6.0 Source distribution and coupling design (headline contribution)

S3FM is defined in two variants. **Both must be implemented**, because the second is the contribution and the first is its control.

**S3FM-Gauss (controlled baseline).** Standard Gaussian source, independent coupling, linear affine path. This is diffusion-equivalent up to noise schedule (§18). It exists to (a) bring up all plumbing on the simplest setting and (b) quantify how much the informative source actually buys at each NFE budget.

```text
Z0 ~ Normal(0, I)        # pure noise, carries no information about X
pi(Z0, Z1) = p(Z0) p(Z1) # independent coupling, so P = 1 exactly
```

**S3FM-Info (the contribution).** The source is a cheap, structured approximation of the target produced by a deterministic, information-destroying operator `S`:

```text
Z1 = X ~ p_data
Z0 = S(X) + eta          # eta is small residual noise to keep the source non-degenerate
pi(Z0, Z1)               # dependent coupling: each source is paired with ITS OWN target
```

`S` is a fixed, cheap map that discards exactly the information guidance must restore. Candidate constructions for KSE, in increasing realism:

1. **Coarse-blur source:** spatially downsample by the same factor as the measurement operator, then upsample back (e.g. blur + decimate + interpolate). The source is "what a naive interpolation of a generic sparse layout would give."
2. **Spectral-truncation source:** keep the lowest `k` Fourier modes of `X`, zero the rest. The flow then learns to synthesize fine-scale structure.
3. **Low-fidelity surrogate source (later systems only):** a fast reduced-order or coarse-grid solve. Reserved for Kolmogorov/ERA5; not required for the KSE pilot.

The source operator `S` for the prior **must not depend on the specific inference-time `H`** — otherwise the operator-agnostic story collapses and the model is just a conditional `(y → X)` regressor in disguise. `S` defines a generic, fixed corruption family; the inference-time `y, H` then steer within the prior. This separation is the entire point and must be stated explicitly in the paper.

**Why this gives few-step sampling (the mechanism we will measure).** Under the linear path `Zs = (1-s)Z0 + sZ1` with `Z0 = S(X)+eta`, the per-sample transport displacement is `Z1 - Z0 = X - S(X) - eta`, i.e. only the *missing detail*, not the entire field. Smaller and lower-curvature displacements need fewer Euler/RK steps for a given error. We will **measure** this, not assume it (see path-length / straightness diagnostic in §12 and the NFE sweep in M3/M4b).

**Coupling validity (`P ≈ 1` scope).** The general guidance of *On the Guidance of Flow Matching* (Eq. 1) carries a reverse-coupling ratio `P = π'(x0|x1)/π(x0|x1)`. S3FM-Info uses a **deterministic** source map, so `π(x0|x1)` is sharply peaked at `x0 = S(x1)`; the guided distribution reweights `x1` but keeps the same conditional source map, giving `P ≈ 1`. We adopt the same `P = 1` approximation the source paper uses for dependent (mini-batch-OT) couplings, state it as an explicit assumption, and flag any regime where it is questionable. No new `P ≠ 1` theory is claimed.

**Train/test source consistency (non-negotiable).** The velocity field is only valid on the path family it was trained on. Therefore:

- The **same** `S` (and the same residual-noise scale for `eta`) used in pretraining must be used to draw `Z0` at inference.
- At inference we do **not** have `X`, so the inference source must be constructed from quantities available at test time in a way that matches the *distribution* of training sources. Acceptable constructions, to be chosen and frozen per experiment:
  - **Marginal-source sampling:** draw `Z0` from the empirical/parametric law of training sources `S(X)+eta` (information-agnostic; safest, breaks no assumption).
  - **Warm-start-from-y (a deliberate, separately-ablated coupling choice):** build `Z0` by mapping the measurement back to field space with a *fixed, generic* reconstruction (e.g. interpolate-then-blur to match `S`'s statistics). This injects observation information into the source. It can shorten transport further, but it changes the inference coupling and **must be reported as a distinct method (`S3FM-Info-warm`)**, with the marginal-source variant kept as the clean comparison.
- A unit test must assert that inference-time `Z0` and training-time `Z0` share matching summary statistics (per-channel mean/std and energy spectrum) within tolerance.

**Scope discipline.** S3FM-Info is the headline. S3FM-Gauss is mandatory as control. Until S3FM-Gauss plumbing passes M0–M4, do not block progress on `S` design; the source operator can be introduced at M4b (§11) once single-window Gaussian guidance works.

### 6.1 Unconditional flow-matching pretraining

For S3FM-Gauss, use an independent Gaussian source and an affine linear path:

```text
Z0 ~ Normal(0, I)
Z1 = X ~ p_data
s  ~ Uniform(0, 1)
Zs = (1 - s) * Z0 + s * Z1
target_velocity = Z1 - Z0
```

For S3FM-Info, replace only the source draw and coupling (the path, loss, and endpoint formula are unchanged):

```text
Z1 = X ~ p_data
Z0 = S(X) + eta          # dependent coupling; same S used at inference (§6.0)
s  ~ Uniform(0, 1)
Zs = (1 - s) * Z0 + s * Z1
target_velocity = Z1 - Z0
```

Train (identical objective for both variants):

```text
L_FM = mean_square(v_theta(Zs, s) - target_velocity)
```

Required model input and output shapes:

```text
input Zs:  [batch, physical_time, channels, *spatial]
input s:   [batch] or broadcast-compatible embedding
output:    exactly the same shape as Zs
```

The first model should reuse the spatiotemporal inductive biases of Video U-Net:

- spatial convolutions;
- temporal attention or temporal mixing;
- skip connections;
- continuous flow-time embedding.

Do not add sensor information during prior pretraining.

### 6.2 Clean endpoint estimate

For the linear path:

```text
X1_hat = Zs + (1 - s) * v_theta(Zs, s)
```

This plays the role of the Tweedie clean estimate in S3GM.

Required unit test:

- With an oracle velocity `v = Z1 - Z0` evaluated on the exact linear path, `X1_hat` must equal `Z1` up to numerical precision for several values of `s`.

### 6.3 Main guidance: covariance-gradient guidance

Define a positive guidance strength `lambda_s >= 0` and use the explicit descent convention:

```text
g_cov_G = -lambda_s * grad_Zs J(X1_hat)
guided_velocity = v_theta(Zs, s) + g_cov_G
```

`grad_Zs` differentiates through both the endpoint expression and `v_theta`. This requires a vector-Jacobian product through the velocity network.

Reference autograd pattern (do **not** detach `velocity` or `x1_hat` in the main method):

```python
zs = zs.requires_grad_(True)
velocity = model(zs, flow_time)
x1_hat = zs + (1.0 - flow_time) * velocity        # source-agnostic endpoint (§6.2)
energy = energy_fn(x1_hat)                          # scalar, normalized
grad = torch.autograd.grad(energy, zs, create_graph=False)[0]
guidance = -lambda_s * grad                         # explicit descent; sign is unit-tested
guided_velocity = velocity + guidance
```

Important implementation rule:

- Do not infer the guidance sign from memory or from a differently oriented time convention.
- The local guidance unit test must verify that a sufficiently small Euler step reduces `J` when the base velocity is disabled.
- The endpoint estimate `X1_hat = Zs + (1-s) v_theta` is derived from the linear path alone, so it is **identical for S3FM-Gauss and S3FM-Info**; no source-specific endpoint code is needed.

### 6.4 Fast ablation: endpoint-gradient guidance

Use:

```text
g_cov_A = -lambda_s * grad_X1_hat J(X1_hat)
```

Treat this gradient as a vector in state space without differentiating through `v_theta`.

This is cheaper but more approximate. It is an ablation, not the initial main claim.

### 6.5 Observation-dependent long-sequence reconstruction

Split the observed sequence into `B` windows of length `T`, overlap `m`, and stride `T - m`.

Example with `T = 5`, `m = 2`:

```text
window 1: frames 1  2  3  4  5
window 2: frames          4  5  6  7  8
window 3: frames                   7  8  9 10 11
```

For window `i`, compute `X1_hat[i]`. Assemble an estimated global sequence for measurement evaluation.

Observation energy:

```text
J_obs = mean_square(y - H(global_X1_hat)) / (2 * sigma_y^2)
```

Sequence-consistency energy:

```text
J_seq = mean over adjacent windows and overlap entries of
        ||overlap(X1_hat[i + 1]) - stop_grad(overlap(X1_hat[i]))||^2
```

Combined energy for Equation-A, the parallel observed stage:

```text
J_parallel = alpha * J_obs + beta * J_seq
```

All windows are integrated together as one joint ODE state. This stage uses `alpha` and `beta`; it does not use `gamma`.

### 6.6 Observation-independent autoregressive forecasting

After measurements end, forecast one window at a time.

Given the last `T_init` frames from the previous completed sequence:

```text
J_init = mean_square(
    first_T_init_frames(X1_hat),
    given_initial_frames
)
```

Autoregressive energy for Equation-B:

```text
J_AR = gamma * J_init
```

Generate the window, append only its new frames, then use its last `T_init` frames to condition the next window.

This stage uses `gamma`; it does not use `alpha`, `beta`, or measurement `y` unless the experiment explicitly performs online data assimilation.

### 6.7 Uncertainty estimation

The ODE is deterministic for a fixed initial source sample, but uncertainty is retained by sampling different `Z0` values:

```text
Z0^(k) ~ Normal(0, I)
X_hat^(k) = GuidedFlow(Z0^(k), y, H)
```

Report mean prediction and standard deviation across independent source samples.

---

## 7. Measurement Operator Interface

Every operator must provide metadata and deterministic behavior.

Recommended interface:

```python
class MeasurementOperator:
    def forward(self, full_state):
        """Map full clean state to measurement-shaped tensor."""

    def metadata(self):
        """Return locations, physical times, variables, units, and normalization."""

    @property
    def is_linear(self):
        return False
```

Linear operators should optionally implement:

```python
def adjoint(self, residual):
    """Apply H^T for tests or inverse-specific guidance."""
```

Initial operators, in required order:

1. `IdentityOperator` for debugging.
2. `MaskOperator` for random sparse KSE observations.
3. `RegularDownsampleOperator`.
4. `FourierSubsampleOperator`.
5. `TemporalAverageOperator`.
6. `VelocityMagnitudeOperator` for nonlinear cylinder measurements.

Each `y` must carry or reference the exact `H` metadata used to produce it. A tensor of values without sensor positions, physical times, observed variables, and measurement type is invalid input.

---

## 8. Guidance Strength and Numerical Stability

Start with a constant positive `lambda_s = lambda0` for debugging. Then compare schedules.

Required schedules for ablation:

1. Constant.
2. Paper-inspired decay toward `s = 1`.
3. A bounded residual-adaptive schedule, introduced only after the constant baseline works.

Safety mechanisms must be configurable and logged:

- gradient norm clipping;
- maximum guidance-to-prior velocity ratio;
- minimum and maximum `lambda_s`;
- ODE solver tolerance;
- fixed-step versus adaptive-step integration.

Always log at each sampled flow time:

```text
||v_theta||
||g||
J_obs
J_seq or J_init
measurement residual
```

Do not hide unstable runs by reporting only successful seeds.

### 8.1 NFE accounting (mandatory for every speed claim)

NFE counts **every call to `v_theta`**, not solver steps. Begin with a fixed-step Euler solver because it makes signs, gradients, and NFE transparent; add fixed-step midpoint/RK4 only afterward. Do not begin with an adaptive black-box solver — it hides NFE and complicates inference-time autograd.

| Solver | Steps | NFE (forward) |
|---|---|---|
| Euler | 50 | 50 |
| Midpoint | 50 | 100 |
| RK4 | 50 | 200 |

Never compare "50 RK4 steps" with "50 Euler steps" as if model cost were equal. Report both nominal steps and true NFE.

**Cost of `g_cov-G`.** Each guided NFE additionally backpropagates through `v_theta` (a VJP), so its wall-clock cost per evaluation is roughly 2–3× an unguided forward call. `g_cov-A` does not backprop through `v_theta` and costs ≈1× plus an endpoint-gradient. Runtime tables (§20) must reflect this; an "NFE-matched" comparison is **not** automatically a wall-clock-matched comparison, and both must be reported.

---

## 9. Recommended Code Organization

```text
src/
  data/
    kse.py
    normalization.py
  models/
    video_unet.py
    time_embedding.py
  flow/
    sources.py        # S(.) source operators + inference-source samplers (S3FM-Info, §6.0)
    paths.py
    losses.py
    endpoint.py
    ode_sampler.py
  guidance/
    energies.py
    cov_g.py
    cov_a.py
    schedules.py
  measurements/
    base.py
    identity.py
    mask.py
    downsample.py
    fourier.py
    temporal.py
    nonlinear.py
  sequences/
    windows.py
    assemble.py
    autoregressive.py
  metrics/
    reconstruction.py
    statistics.py
    runtime.py
  train_flow.py
  sample_guided.py
  evaluate.py
configs/
  kse_flow.yaml
  kse_guided_mask.yaml
tests/
  test_flow_path.py
  test_endpoint.py
  test_measurements.py
  test_guidance_direction.py
  test_windows.py
  test_sampler_smoke.py
experiments/
  README.md
  results.csv
```

Adapt this structure to the existing repository rather than duplicating established abstractions.

---

## 10. Minimum Configuration Fields

Every experiment must save a resolved configuration containing:

```yaml
seed: 0
dataset:
  name: kse
  train_split: null
  val_split: null
  test_split: null
  physical_dt: null
  spatial_resolution: null
  window_length: null
  normalization: null

flow:
  source: standard_gaussian
  path: linear
  model: video_unet
  checkpoint: null

sampler:
  solver: euler_or_selected_solver
  nfe: 50
  adaptive: false

guidance:
  method: cov_g
  lambda0: null
  schedule: constant
  gradient_clip: null
  max_guidance_prior_ratio: null

measurement:
  operator: mask
  observed_fraction: null
  noise_std: null

sequence:
  overlap: null
  initial_frames: null

evaluation:
  num_samples: 3
  metrics:
    - nrmse
    - observation_residual
    - two_point_correlation
    - runtime
```

No experiment result is valid without its resolved configuration and model checkpoint identifier.

---

## 11. Milestones and Acceptance Criteria

### M0. Repository and reproducibility foundation

- [ ] Inspect the repository and dependency environment.
- [ ] Add deterministic seed handling.
- [ ] Add configuration loading and resolved-config saving.
- [ ] Add experiment result logging to CSV or JSONL.
- [ ] Confirm CPU smoke tests can run without a GPU.

**Acceptance:** Two identical smoke runs with the same seed produce identical losses and samples within numerical tolerance.

### M1. KSE data pipeline

- [ ] Load or generate KSE trajectories.
- [ ] Freeze train/validation/test splits.
- [ ] Implement reversible normalization.
- [ ] Construct windows of exactly `T` physical frames.
- [ ] Visualize several clean samples and verify axes.

**Acceptance:** Normalization round-trip error is below `1e-6`; no trajectory leaks across splits; sample dimensions are documented.

### M2. Unconditional flow-matching prior

- [ ] Implement linear conditional path.
- [ ] Implement flow-matching loss.
- [ ] Train a small overfitting model on a tiny batch.
- [ ] Train the first full KSE prior.
- [ ] Save checkpoint and training curves.

**Acceptance:** Tiny-batch loss decreases substantially; endpoint oracle test passes; unguided samples are finite and non-constant.

### M3. Unguided prior evaluation

- [ ] Sample with NFE `10, 20, 50, 100`.
- [ ] Compare marginal value distribution and two-point correlation with test data.
- [ ] Measure wall-clock runtime and peak memory.

**Acceptance:** At least one NFE setting produces statistically plausible KSE samples before measurement guidance is added.

### M4. Single-window sparse observation guidance

- [ ] Implement `IdentityOperator` and `MaskOperator`.
- [ ] Implement `J_obs`.
- [ ] Implement `g_cov-G`.
- [ ] Add guidance-direction unit test.
- [ ] Reconstruct one KSE window from known sparse observations.
- [ ] Sweep a small set of `lambda0` values.

**Acceptance:** Guided reconstruction has lower measurement residual and lower mean nRMSE than unguided sampling on the same observations and seeds.

> M2–M4 use the **S3FM-Gauss** prior (Gaussian source). This brings up all plumbing on the diffusion-equivalent control before introducing the informative source at M4b.

### M4b. Informative-source flow prior — headline contribution

This is the core scientific milestone, not an optional extension. Do it after M4 plumbing passes.

- [ ] Implement `S(.)` source operators in `flow/sources.py` (start with coarse-blur and spectral-truncation for KSE).
- [ ] Implement the matched inference-source sampler (marginal-source variant first; `warm-start-from-y` only as a separately-labelled second method).
- [ ] Add source determinism, train/test source-consistency, and transport-shortening tests (§13.13–13.15).
- [ ] Train the **S3FM-Info** prior with the dependent coupling `Z0 = S(X)+eta` (same loss, path, endpoint as S3FM-Gauss).
- [ ] Re-run the M4 single-window guided reconstruction with the informative prior.
- [ ] Measure mean transport displacement `||Z1 - Z0||` and path curvature for both priors.

**Acceptance:** (a) endpoint and consistency tests pass for the informative source; (b) at matched NFE, S3FM-Info reaches a lower (or equal) reconstruction error than S3FM-Gauss, with the gap growing as NFE shrinks; (c) the measured transport displacement/curvature is smaller for the informative source, evidencing the mechanism. If (b) fails, report it honestly — it would falsify the primary hypothesis and must be surfaced, not buried.

### M5. Parallel long-sequence reconstruction

- [ ] Implement overlapping window decomposition.
- [ ] Implement stop-gradient sequence consistency.
- [ ] Integrate all observed windows jointly.
- [ ] Assemble the final long sequence.
- [ ] Compare with and without `J_seq`.

**Acceptance:** Sequence consistency reduces overlap disagreement without materially increasing observation residual. No duplicated or missing physical frame indices.

### M6. Autoregressive future forecasting

- [ ] Implement `J_init`.
- [ ] Generate future windows sequentially.
- [ ] Append only new frames.
- [ ] Record pointwise error and statistical error versus forecast horizon.

**Acceptance:** The first generated frames match the supplied initial frames; long-horizon outputs remain finite; error accumulation is plotted rather than hidden.

### M7. Main speed-quality benchmark

- [ ] Compare S3GM at 1,000 steps with S3FM at NFE `10, 20, 50, 100`.
- [ ] **Attribution baseline (mandatory):** integrate the *same S3GM score model* as a probability-flow ODE with the *same guidance* at NFE `10, 20, 50, 100` (`S3GM-PF-ODE`). This separates "ODE vs SDE" from "the S3FM method."
- [ ] Report both S3FM-Gauss and S3FM-Info at every NFE, so the informative-source gain is visible against the diffusion-equivalent control.
- [ ] Use identical test cases, observations, normalization, and hardware.
- [ ] Include gradient-computation time in S3FM runtime; report NFE-matched **and** wall-clock-matched comparisons (§8.1).
- [ ] Repeat with at least three seeds.

**Primary success criterion:** at NFE ≤ 50, S3FM-Info reaches a predeclared quality threshold near S3GM@1,000 **and beats `S3GM-PF-ODE` at the same NFE**, with a meaningful measured wall-clock speedup. If S3GM-PF-ODE already matches S3FM at low NFE, the honest conclusion is that the gain is from deterministic integration, not from this method — state that outcome plainly if it occurs.

The exact quality threshold must be fixed before the final benchmark. A reasonable pilot threshold is no more than 10% relative degradation in nRMSE and statistical-feature error, but it must be reconsidered after reproducing the S3GM baseline.

### M8. Guidance and source ablations

- [ ] `g_cov-G` versus `g_cov-A`.
- [ ] **Source distribution:** Gaussian vs coarse-blur vs spectral-truncation vs warm-start-from-y, at matched NFE.
- [ ] **Coupling sensitivity:** vary the residual-noise scale `eta` (which controls how dependent the coupling is) and check guidance stability / the `P ≈ 1` assumption.
- [ ] Constant versus scheduled guidance.
- [ ] Different observation sparsity and noise.
- [ ] Different overlap sizes.
- [ ] Fixed-step versus adaptive ODE solver.

**Acceptance:** Each ablation changes exactly one primary factor and uses the same evaluation set.

### M9. Additional systems

Proceed only after M7 succeeds.

Recommended order:

1. Kolmogorov flow.
2. ERA5 incomplete-variable reconstruction.
3. Cylinder flow with nonlinear velocity-magnitude observations.

Each new system requires its own unconditional prior. Do not describe this as zero-shot transfer across systems.

---

## 12. Experiment Matrix

Minimum KSE matrix:

| Factor | Values |
|---|---|
| Method | S3GM (1,000-step SDE), **S3GM-PF-ODE (attribution baseline)**, unguided FM, `g_cov-A`, `g_cov-G` |
| Source / coupling | **Gaussian (S3FM-Gauss control), coarse-blur, spectral-truncation, warm-start-from-y** |
| NFE | 10, 20, 50, 100; S3GM reference at 1,000; S3GM-PF-ODE at 10, 20, 50, 100 |
| Observation | random mask, regular downsampling, Fourier subsampling, initial frames |
| Sparsity/downsampling | at least 3 levels, including one difficult level |
| Noise | 0 and at least one nonzero level |
| Sequence consistency | off/on |
| Seeds | at least 3 |

Required metrics:

- normalized RMSE;
- observation residual in measurement space;
- spatial two-point correlation error for KSE;
- overlap disagreement;
- wall-clock sampling time;
- NFE;
- peak memory if available;
- mean and standard deviation across seeds.
- **mean transport displacement `||Z1 - Z0||` and discrete ODE path curvature** (to evidence the shorter-transport mechanism and compare sources);
- **train/test source-distribution shift diagnostic** for S3FM-Info (per-channel mean/std and spectrum distance between training and inference sources).

For turbulence, additionally report kinetic energy spectrum error.

---

## 13. Mandatory Unit and Integration Tests

1. **Linear path test:** `Zs` equals the analytical interpolation (both sources).
2. **Velocity target test:** analytical derivative equals `Z1 - Z0` (both sources).
3. **Endpoint test:** oracle velocity recovers `Z1`, verified for Gaussian **and** informative `Z0` (the endpoint formula is source-agnostic).
4. **Identity measurement test:** `H(X) = X` exactly.
5. **Mask measurement test:** observed entries and shapes are exact.
6. **Adjoint test for linear H:** `<Hx, y> = <x, H^T y>` within tolerance.
7. **Guidance sign test:** a small guidance-only step decreases energy.
8. **Zero-guidance equivalence:** `lambda0 = 0` matches unguided ODE output.
9. **Window indexing test:** decomposition and assembly preserve every global frame.
10. **Overlap loss test:** identical overlaps give zero loss; perturbed overlaps give positive loss.
11. **Autoregressive append test:** no duplicated output frames after removing overlap.
12. **Gradient finiteness test:** all guidance gradients are finite for representative inputs.
13. **Source determinism test (S3FM-Info):** `S(X)` is deterministic given a seed for `eta`, and re-applying `S` is idempotent enough to be reproducible.
14. **Train/test source-consistency test (S3FM-Info):** inference-time `Z0` and training-time `S(X)+eta` match in per-channel mean/std and energy spectrum within a declared tolerance; the test fails loudly on a deliberately mismatched source.
15. **Transport-shortening sanity test:** mean `||Z1 - Z0||` and discrete path curvature are strictly smaller for the informative source than for the Gaussian source on the same targets.

---

## 14. Common Failure Modes

### Mathematical failures

- Wrong guidance sign due to reversing diffusion and flow time conventions.
- Applying `H` to `Zs` instead of the estimated clean endpoint.
- Comparing measurements and generated values in different normalization spaces.
- Treating physical frame index as flow time.
- Using `g_cov-A` while claiming `g_cov-G`.
- Backpropagating into both sides of the stop-gradient overlap target.
- Letting the source operator `S` depend on the inference `H`, turning the "prior" into a disguised conditional regressor.
- Assuming the guidance is exact when the deterministic-source coupling makes `P ≠ 1`.

### Numerical failures

- Guidance norm dominates prior velocity.
- Endpoint estimate is poor near the source and causes unstable early guidance.
- Strong guidance bends the ODE path and removes the expected NFE advantage.
- Adaptive solver performs many hidden evaluations; report actual NFE, not nominal steps.
- Autoregressive prediction feeds accumulated error back as truth.
- Inference source drawn from a different distribution than training source, so `v_theta` is queried off-path at `s = 0`.

### Scientific failures

- Comparing methods with different preprocessing or data splits.
- **Attributing a speedup to "flow matching" without the `S3GM-PF-ODE` baseline** (it may be ODE-vs-SDE).
- **Claiming the informative source helps without the matched-NFE Gaussian-source ablation and a measured transport-shortening.**
- Reporting only measurement residual while ignoring full-field error.
- Reporting only nRMSE for chaotic systems while ignoring statistical features.
- Calling a plausible unobserved variable a measured recovery.
- Claiming real-time performance based only on theoretical step-count reduction.
- Claiming cross-system zero-shot generalization.

---

## 15. AI Task Prompt Template

Use this template for each implementation session:

```text
Read docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

Current milestone: M[number and name].

Your task:
[one bounded task]

Required behavior:
1. Inspect existing code before editing.
2. Preserve unrelated changes.
3. State the exact mathematical convention you will implement.
4. Add or update focused tests.
5. Run the smallest relevant test suite.
6. Do not start later milestones.
7. Report changed files, tests, results, assumptions, and remaining risks.

Acceptance criteria:
[copy the relevant criteria from Section 11]
```

Example for M4:

```text
Read docs/S3FM_AI_IMPLEMENTATION_SPEC.md completely.

Current milestone: M4, single-window sparse observation guidance.

Implement MaskOperator, normalized Gaussian observation energy, and g_cov-G for one KSE window. Use the explicit convention g = -lambda * grad_Zs J(X1_hat). Add tests proving that the mask is correct, lambda=0 reproduces unguided sampling, gradients are finite, and a small guidance-only step reduces J. Do not implement overlapping windows or autoregressive generation yet.
```

---

## 16. AI Completion Report Template

Every AI implementation response should end with:

```text
Milestone:

Outcome:

Files changed:

Mathematical conventions:

Tests executed and results:

Experiment configuration:

Measured metrics/runtime:

Known limitations or risks:

Next permitted step:
```

---

## 17. Paper Structure

1. **Introduction**
   - Sparse-sensor reconstruction is important.
   - S3GM is flexible but slow (1,000-step SDE).
   - Flow matching enables transport from an informative source, but spatiotemporal posterior guidance from such sources is unestablished; a Gaussian-source flow is merely a re-scheduled diffusion, so the source is where the leverage is.

2. **Related Work**
   - Spatiotemporal reconstruction.
   - Score-based inverse problems and S3GM.
   - Flow matching, general flow guidance, and couplings.
   - Scientific generative modeling.

3. **Problem Setup**
   - `y = H(X) + noise`.
   - Zero-shot refers to new measurement operators for a fixed dynamical system.

4. **Method**
   - Informative-source spatiotemporal flow prior and its dependent coupling (`P ≈ 1` scope).
   - Endpoint observation guidance for general flow matching.
   - Parallel sequence consistency.
   - Autoregressive continuation.
   - Solver, NFE accounting, and the shorter-transport mechanism.

5. **Experiments**
   - Accuracy, statistics, uncertainty, runtime, NFE.
   - Linear and nonlinear measurements.
   - Ablations.

6. **Limitations**
   - Approximate guidance.
   - Prior shift.
   - Strong-guidance path curvature.
   - Autoregressive error accumulation.
   - Per-system pretraining.

7. **Conclusion**

Do not write numerical claims into the abstract until M7 is complete.

---

## 18. Decision Log

Record all consequential decisions here.

| Date | Decision | Reason | Status |
|---|---|---|---|
| 2026-06-22 | Use independent Gaussian source and linear affine path for first implementation | Simplest setting; aligns with `g_cov-G` assumptions and clean endpoint formula | Superseded as the *contribution* (now the S3FM-Gauss control); retained as plumbing baseline |
| 2026-06-22 | Use `g_cov-G` as main method | Supports differentiable nonlinear `H` and is closest to S3GM/DPS guidance | Active |
| 2026-06-22 | Use `g_cov-A` only as initial speed ablation | Cheaper but more approximate | Active |
| 2026-06-22 | Begin with KSE | Lowest-cost controlled benchmark with pointwise and statistical metrics | Active |
| 2026-06-22 | Require <= 50 NFE as first performance target | Creates a concrete speed objective relative to 1,000-step S3GM | Provisional |
| 2026-06-22 | **Elevate informative-source / dependent-coupling flow (S3FM-Info) to the headline contribution; demote Gaussian-source to controlled baseline** | A Gaussian-source linear path is diffusion-equivalent up to schedule, so it cannot carry the novelty; an informative source shortens transport (the real few-step mechanism) and exploits flow matching's unique non-Gaussian-source capability | Active |
| 2026-06-22 | **Add `S3GM-PF-ODE` (S3GM score model under a probability-flow ODE + same guidance) as a mandatory attribution baseline** | Without it, any speedup is mis-attributed to "flow matching" when it may be due to ODE-vs-SDE integration | Active |
| 2026-06-22 | **Operate in the `P ≈ 1` reverse-coupling-ratio regime; treat `P ≠ 1` correction as out of scope** | The source paper's guidance assumes `P ≈ 1`; a deterministic source map keeps us inside this regime; deriving the `P ≠ 1` correction would be a separate theoretical contribution we do not claim here | Active |
| 2026-06-22 | **Consolidate specs: this file is the single source of truth; `S3FM_CORE_SPEC.md` becomes a pointer stub; `S3FM_START_HERE.md` and the checklist point here** | Two specs previously both claimed authority, risking drift | Active |

---

## 19. Definition of Project Success

The project is successful only if experiments demonstrate all of the following:

1. The flow prior generates statistically plausible unconditioned dynamics.
2. Guided samples match sparse measurements and improve full-field reconstruction over unguided samples.
3. **The informative-source prior (S3FM-Info) beats the Gaussian-source control (S3FM-Gauss) at matched NFE, with a measured reduction in transport displacement/curvature explaining the gain.**
4. **At NFE ≤ 50, S3FM beats the `S3GM-PF-ODE` attribution baseline at the same NFE — i.e. the advantage is not merely SDE→ODE.**
5. Long sequences do not develop unacceptable overlap seams.
6. Autoregressive forecasts remain numerically stable and report error accumulation honestly.
7. Actual wall-clock runtime, including guidance gradients, improves meaningfully over S3GM.
8. Accuracy-speed tradeoffs are reported across NFE rather than at one favorable setting.
9. Claims remain limited to the dynamical systems and measurement conditions tested.

The desired headline result is:

> By transporting from an informative source rather than Gaussian noise, a training-free, measurement-operator-flexible guided flow reconstructs spatiotemporal dynamics with accuracy close to S3GM using an order of magnitude fewer function evaluations and a verified wall-clock speedup — and this advantage holds against a probability-flow-ODE version of S3GM at matched NFE, confirming the gain comes from the method, not just from deterministic integration.

