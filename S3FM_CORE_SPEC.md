# S3FM Core Research and Implementation Specification

> Working title: **Sparse-Sensor-Guided Flow Matching for Efficient Spatiotemporal Dynamics Reconstruction**  
> Working acronym: **S3FM**  
> Status: implementation specification, before pilot experiments  
> Primary pilot system: Kuramoto-Sivashinsky equation (KSE)

## 1. Purpose of this document

This document is the source of truth for implementing and evaluating S3FM with the help of AI coding agents.

An implementation agent must read this document before changing code. If existing code and this document disagree, the agent must report the discrepancy instead of silently changing the research design.

The immediate objective is:

> On KSE, reconstruct full spatiotemporal fields from sparse measurements using at most 50 ODE function evaluations while approaching the reconstruction accuracy and statistical fidelity of the 1,000-step S3GM sampler.

This is a research hypothesis, not an established result. Do not state it as achieved until experiments verify it.

## 2. Research motivation

S3GM separates two kinds of information:

1. A pretrained generative prior that represents plausible spatiotemporal dynamics.
2. Inference-time guidance from a measurement pair `(y, H)`.

This makes S3GM flexible across sensor layouts and measurement types, but its reverse-SDE sampler uses 1,000 generating steps and is expensive for high-dimensional fields.

S3FM preserves the same separation while replacing score-based SDE sampling with a flow-matching ODE:

```text
S3GM: pretrained score prior + measurement-guided reverse SDE
S3FM: pretrained velocity prior + measurement-guided forward ODE
```

The intended advantage is fewer model evaluations and lower wall-clock inference time without retraining for each new differentiable measurement operator `H`.

## 3. Novelty boundary

The project combines and extends ideas from:

- S3GM: sparse-sensor posterior reconstruction, overlapping-window consistency, and autoregressive long-horizon generation.
- On the Guidance of Flow Matching: general flow-matching guidance, especially covariance-gradient guidance.

Simply replacing a diffusion sampler with flow matching is not, by itself, a strong methodological contribution. A publishable paper should demonstrate at least the following:

1. A working operator-agnostic flow-guidance formulation for spatiotemporal inverse problems.
2. Long-sequence reconstruction using jointly guided overlapping flow windows.
3. A convincing speed-accuracy study against S3GM.
4. At least one additional contribution, preferably residual-adaptive guidance or adaptive NFE allocation.

Candidate additional contribution:

> Adapt the guidance strength and/or ODE step allocation using normalized observation residual and overlap residual, concentrating computation where constraints are not yet satisfied.

Do not claim a new general theory of flow guidance unless new proofs are actually developed.

## 4. Non-negotiable notation

Notation must remain consistent across code, comments, figures, and the paper.

| Symbol | Meaning | Code name |
|---|---|---|
| `t_phys` or `n` | Physical time/frame index of the dynamical system | `frame_idx` |
| `s in [0,1]` | Flow-generation time | `flow_time` |
| `X_0` | Source sample, normally Gaussian noise | `x_source` |
| `X_1` | Clean target spatiotemporal sample | `x_target` |
| `X_s` | Intermediate flow state | `x_s` |
| `v_theta(X_s,s)` | Learned flow velocity | `velocity` |
| `X1_hat` | Estimated clean endpoint from `X_s` | `x1_hat` |
| `y` | Measurements from the current case | `observation` |
| `H` | Known differentiable measurement operator | `measurement_operator` |
| `T` | Number of physical frames in one model window | `window_length` |
| `T_prime` | Desired total number of physical frames | `target_length` |
| `m` | Number of overlapping frames between windows | `overlap` |
| `B` | Number of windows | `num_windows` |
| `T_init` | Initial frames used for autoregressive continuation | `num_initial_frames` |
| `J_obs` | Observation-consistency energy | `observation_energy` |
| `J_seq` | Parallel-window sequence-consistency energy | `sequence_energy` |
| `J_init` | Autoregressive initial-frame energy | `initial_energy` |
| `NFE` | Number of velocity-model evaluations | `nfe` |

Never use plain `t` for both physical time and flow time in the same implementation or explanation.

## 5. Problem definition

The unknown clean spatiotemporal field is:

```math
X = \{x^{(n)}\}_{n=1}^{T'},
```

where each physical snapshot may contain multiple quantities of interest (QoIs).

Measurements follow:

```math
y = H(X) + \epsilon.
```

`H` may select sparse space-time points, remove entire QoI channels, transform to Fourier space, compute a velocity magnitude, or compute a time average.

The desired posterior is conceptually:

```math
p(X \mid y) \propto p_\theta(X)\,p(y\mid X).
```

S3FM represents `p_theta(X)` through an unconditional flow-matching model and imposes `p(y|X)` through inference-time energy guidance.

## 6. Base flow-matching model

### 6.1 Initial implementation choice

Use the simplest setting first:

- Source distribution: independent standard Gaussian.
- Coupling: independent pairing of noise and clean samples.
- Conditional path: linear interpolation.
- Parameterization: velocity prediction.

```math
X_0 \sim \mathcal N(0,I), \qquad X_1 \sim p_{data}(X),
```

```math
X_s = (1-s)X_0 + sX_1,
```

```math
u_s(X_s\mid X_0,X_1)=X_1-X_0.
```

Train `v_theta` with conditional flow matching:

```math
\mathcal L_{FM}
=
\mathbb E\left[
\|v_\theta(X_s,s)-(X_1-X_0)\|_2^2
\right].
```

### 6.2 Model interface

The velocity model must satisfy:

```python
velocity = model(x_s, flow_time)
assert velocity.shape == x_s.shape
```

Recommended tensor convention:

```text
[batch, physical_time, channels, spatial_dimensions...]
```

For KSE:

```text
[batch, T, C, Nx]
```

For two-dimensional fields:

```text
[batch, T, C, Ny, Nx]
```

If an existing Video U-Net expects a different convention, isolate the permutation in a wrapper. Do not scatter dimension permutations through training and sampling code.

### 6.3 Clean endpoint estimate

For the linear path:

```math
\hat X_1(X_s,s)=X_s+(1-s)v_\theta(X_s,s).
```

This is the flow-matching analogue of the Tweedie clean estimate used by S3GM.

At `s -> 1`, `X1_hat` should approach the final clean sample.

## 7. Guidance design

### 7.1 Main guidance method

The primary method is covariance-gradient guidance, denoted `g_cov-G` in *On the Guidance of Flow Matching*.

For implementation clarity, use an explicitly descending energy direction:

```math
g_s(X_s) = -w(s)\nabla_{X_s}J(\hat X_1(X_s,s)).
```

The guided ODE is:

```math
\frac{dX_s}{ds}=v_\theta(X_s,s)+g_s(X_s).
```

The sign must be tested numerically. A small guided step must reduce the selected energy in a toy problem. Never trust the sign from visual inspection of a paper equation alone because flow-time conventions differ.

### 7.2 Important autograd distinction

For `g_cov-G`, the gradient must pass through the endpoint estimator and through `v_theta`:

```text
X_s -> v_theta(X_s,s) -> X1_hat -> J -> gradient with respect to X_s
```

Conceptual PyTorch pattern:

```python
x_s = x_s.requires_grad_(True)
velocity = model(x_s, flow_time)
x1_hat = x_s + (1.0 - flow_time) * velocity
energy = energy_fn(x1_hat)
guidance = -weight(flow_time) * torch.autograd.grad(
    energy, x_s, create_graph=False
)[0]
guided_velocity = velocity + guidance
```

Do not detach `velocity` or `x1_hat` in the main method.

### 7.3 Fast approximation

Implement `g_cov-A` as an ablation after `g_cov-G` works:

```math
g_s^{cov-A}=-w_A(s)\nabla_{\hat X_1}J(\hat X_1).
```

This does not backpropagate through `v_theta`. It is faster but expected to be less accurate.

### 7.4 Methods not selected for the first version

- Do not start with `g_MC`: its variance is unsuitable for high-dimensional spatiotemporal inverse problems.
- Do not start with a learned guidance network `g_phi`: it adds training complexity and weakens the operator-agnostic story.
- Consider `g_sim-inv-A` only for linear measurement operators after the main method works.

## 8. Energy functions

All energies must be scalar, differentiable, normalized, and logged separately.

### 8.1 Observation energy

```math
J_{obs}(\hat X_1)
=
\frac{1}{2\sigma_y^2}
\|y-H(\hat X_1)\|_2^2.
```

In code, also log the normalized residual:

```math
r_{obs}
=
\frac{\|y-H(\hat X_1)\|_2}
{\|y\|_2+\varepsilon}.
```

### 8.2 Parallel-window sequence energy

Suppose windows have length `T` and overlap `m`.

Example:

```text
window 1: frames 1 2 3 4 5
window 2:          4 5 6 7 8
```

The same physical frames have two generated versions. Penalize their mismatch:

```math
J_{seq}
=
\sum_{i=1}^{B-1}
\left\|
\hat X_{1,head}^{(i+1)}
-
\operatorname{sg}(\hat X_{1,tail}^{(i)})
\right\|_2^2.
```

`sg` is stop-gradient. The earlier window is the anchor for this energy term.

Normalize by the number of overlapping elements so that changing spatial resolution or overlap does not silently change the guidance scale.

### 8.3 Autoregressive initial-frame energy

For a future window conditioned on the previous window's final frames:

```math
J_{init}
=
\sum_{n=1}^{T_{init}}
\|\hat x_1^{(n)}-x_{given}^{(n)}\|_2^2.
```

Normalize by the number of constrained elements.

### 8.4 Piecewise use of energies

Do not place all energies in every generation phase.

Observed long-sequence reconstruction uses:

```math
J = \lambda_{obs}J_{obs}+\lambda_{seq}J_{seq}.
```

Observation-free autoregressive continuation uses:

```math
J = \lambda_{init}J_{init}.
```

This mirrors the separation between equations (5) and (6) in S3GM.

## 9. Measurement-operator interface

Every measurement operator must implement a common interface:

```python
class MeasurementOperator:
    def forward(self, full_field, metadata):
        """Map full spatiotemporal field to measurement space."""

    def validate(self, full_field, observation, metadata):
        """Check shapes, units, coordinates, times, and channels."""
```

The metadata must explicitly identify:

- physical frame indices or timestamps;
- sensor coordinates;
- measured QoI/channel;
- measurement units and normalization;
- noise scale;
- any interpolation rule.

Initial operators:

1. `MaskSamplingOperator`: sparse space-time point selection.
2. `RegularDownsampleOperator`: regular grid downsampling.
3. `FourierObservationOperator`: selected Fourier coefficients.
4. `VelocityMagnitudeOperator`: `sqrt(u^2 + v^2 + eps)`.
5. `TimeAverageOperator`: average or integral over selected physical frames.

For every operator, add a synthetic test where `y = H(X_true)` and verify exact alignment.

## 10. Window manager

The window manager must be independent of the generative model.

Required functions:

```python
windows, index_map = split_windows(full_sequence, window_length=T, overlap=m)
assembled = assemble_windows(windows, index_map, reduction="anchor_or_average")
loss = overlap_energy(windows, index_map, stop_gradient_anchor=True)
```

For window `i`, the global starting frame is:

```math
start_i=i(T-m).
```

The stride is `T-m`.

Required unit tests:

- splitting followed by assembly recovers the original sequence;
- identical overlaps yield zero sequence energy;
- changing one overlap element produces a positive energy;
- the global frame indices of both copies of an overlap are identical;
- no physical frame is skipped;
- the final target length is correct for non-divisible lengths.

## 11. ODE integration and NFE accounting

### 11.1 First implementation

Begin with a fixed-step Euler solver because it makes gradients, signs, and NFE transparent.

Then add fixed-step midpoint or RK4.

Do not begin with an adaptive black-box ODE solver. It can hide NFE, invoke the vector field many times, and complicate inference-time autograd.

### 11.2 NFE definition

Count every call to `v_theta`.

- Euler with 50 steps: 50 NFE.
- Midpoint with 50 steps: 100 NFE.
- RK4 with 50 steps: 200 NFE.

Report both solver steps and actual NFE. Never compare “50 RK4 steps” with “50 Euler steps” as if their model cost were equal.

### 11.3 Solver termination

The initial experiments use a fixed interval `s=0 -> 1` and a fixed NFE budget.

Adaptive termination based on residual is a later research contribution, not part of the MVP.

## 12. Training algorithm

```text
Input: complete training sequences X1

repeat:
    sample clean window X1
    sample Gaussian source X0 with the same shape
    sample flow time s ~ Uniform(0, 1)
    construct Xs = (1-s)X0 + sX1
    target_velocity = X1 - X0
    predicted_velocity = v_theta(Xs, s)
    loss = mean_square(predicted_velocity, target_velocity)
    update theta

save:
    model weights
    optimizer state
    EMA weights if used
    normalization statistics
    exact configuration and source revision
```

Validation must include both velocity loss and unguided sample quality. Low flow-matching loss alone is not sufficient.

## 13. Guided reconstruction algorithm

```text
Input:
    frozen velocity model v_theta
    observation y
    differentiable measurement operator H
    target physical length T_prime
    window length T and overlap m
    ODE NFE budget

initialize B Gaussian source windows Xs=0^(1...B)

for each ODE evaluation from s=0 to s=1:
    evaluate velocity for every window
    estimate each clean endpoint:
        X1_hat^(i) = Xs^(i) + (1-s) * velocity^(i)
    assemble the endpoint windows into one global sequence
    compute J_obs from y and H(global_endpoint)
    compute J_seq from matching overlapping endpoint frames
    compute guidance gradient with respect to all current Xs windows
    guided_velocity = base_velocity - guidance_weight(s) * gradient
    update all windows using the selected ODE solver

at s=1:
    compute final endpoints
    assemble the output sequence
    return reconstruction and diagnostics
```

Diagnostics must include trajectories of:

- `J_obs`;
- `J_seq`;
- normalized observation residual;
- guidance norm;
- base velocity norm;
- ratio `||guidance|| / (||velocity|| + eps)`;
- NFE;
- wall-clock time;
- peak accelerator memory.

## 14. Autoregressive future-generation algorithm

```text
Input:
    final T_init frames of the observed/reconstructed segment
    desired future horizon

while horizon is not reached:
    initialize one Gaussian source window
    define J_init using the given T_init frames
    solve the guided ODE from s=0 to s=1
    retain the newly generated non-overlapping frames
    use the latest T_init generated frames as the next condition

return concatenated future sequence
```

Log pointwise error and statistical error separately. In chaotic systems, pointwise trajectories may diverge while statistical features remain meaningful.

## 15. Repository architecture

Suggested structure:

```text
s3fm/
  configs/
    kse_flow_prior.yaml
    kse_guided_reconstruction.yaml
  data/
    kse.py
  models/
    video_unet_velocity.py
    time_embedding.py
  flow/
    paths.py
    training.py
    endpoint.py
    solvers.py
  guidance/
    energies.py
    cov_g.py
    cov_a.py
    schedules.py
  measurements/
    base.py
    mask.py
    downsample.py
    fourier.py
    velocity_magnitude.py
    time_average.py
  sequences/
    windows.py
    autoregressive.py
  evaluation/
    metrics.py
    runtime.py
    plots.py
  tests/
    test_paths.py
    test_endpoint.py
    test_measurements.py
    test_windows.py
    test_guidance_sign.py
    test_solvers.py
  train_flow.py
  reconstruct.py
  forecast.py
```

Adapt this to the existing repository rather than duplicating equivalent modules.

## 16. Configuration schema

Minimum reproducible configuration:

```yaml
experiment:
  name: kse_sparse_reconstruction_cov_g
  seed: 0

data:
  system: kse
  window_length: 20
  physical_dt: 0.5
  normalization: channel_standardization

flow:
  source: standard_gaussian
  path: linear
  parameterization: velocity
  ema: true

guidance:
  method: cov_g
  observation_weight: 1.0
  sequence_weight: 0.1
  schedule: constant_then_decay
  max_guidance_to_velocity_ratio: null

measurement:
  operator: regular_downsample
  spatial_factor: 8
  noise_std: 0.0

sequence:
  target_length: 100
  overlap: 2

solver:
  name: euler
  steps: 50

evaluation:
  num_posterior_samples: 3
  save_trajectories: true
```

The exact initial values are pilot settings, not final paper settings.

## 17. Correctness tests before scientific experiments

An AI agent must not launch expensive training until these tests pass.

### 17.1 Linear path identity

For a known pair `(X0, X1)` and oracle velocity `X1-X0`, verify that Euler integration recovers `X1` up to numerical error.

### 17.2 Endpoint identity

For oracle linear velocity, verify:

```math
X_s+(1-s)(X_1-X_0)=X_1.
```

### 17.3 Guidance sign

Use a one-dimensional identity measurement:

```math
H(x)=x, \qquad J(x)=\tfrac12(x-y)^2.
```

Verify one small guided update lowers `J`.

### 17.4 Measurement alignment

Construct a field with unique values at every `(frame, channel, coordinate)` and confirm every operator extracts the intended values.

### 17.5 Window indexing

Use integer frame labels and verify overlaps refer to the same global frames.

### 17.6 No-guidance equivalence

Setting all guidance weights to zero must reproduce the unguided solver output exactly for the same source noise and solver.

## 18. Milestones and acceptance criteria

### Milestone 0: environment and repository audit

Tasks:

- inspect existing code, data, checkpoints, environment, and git status;
- identify reusable S3GM data pipelines and Video U-Net components;
- document tensor conventions and normalization;
- run existing tests or a minimal baseline command.

Done when:

- an audit note identifies reusable modules and conflicts;
- no user changes were overwritten;
- one existing baseline can be executed or its blocker is documented.

### Milestone 1: KSE dataset and deterministic baseline

Tasks:

- load or generate KSE trajectories;
- create train/validation/test splits without leakage;
- reproduce reference visualizations and metrics;
- implement measurement operators.

Done when:

- a held-out complete sequence can be loaded;
- `y = H(X_ref)` is verified;
- normalization and inverse normalization round-trip correctly.

### Milestone 2: unconditional flow prior

Tasks:

- implement linear conditional flow matching;
- train the velocity model;
- implement unguided ODE generation;
- evaluate sample statistics.

Done when:

- training is stable;
- unguided samples have plausible KSE structure;
- two-point correlation is meaningfully close to the test distribution;
- checkpoints and exact configs are saved.

### Milestone 3: single-window sparse reconstruction

Tasks:

- implement `g_cov-G`;
- reconstruct one model-length window using a linear sparse `H`;
- sweep guidance weight and NFE;
- compare with unguided flow.

Done when:

- guidance consistently lowers observation residual;
- reconstruction improves over unguided flow;
- no NaNs or exploding guidance occur;
- runtime and NFE are correctly reported.

### Milestone 4: long observed sequence

Tasks:

- implement overlapping parallel windows;
- add `J_seq` with stop-gradient anchors;
- reconstruct a sequence longer than the training window.

Done when:

- overlap discontinuity is measurably lower with `J_seq`;
- assembled sequence has the exact requested length;
- observation residual and sequence residual both converge.

### Milestone 5: autoregressive future prediction

Tasks:

- implement `J_init` and rolling windows;
- forecast beyond the observation interval;
- evaluate error growth and statistical fidelity.

Done when:

- every future window begins consistently with the previous tail;
- long-horizon nRMSE and two-point correlation are reported;
- uncertainty is estimated with multiple source-noise seeds.

### Milestone 6: S3GM comparison

Tasks:

- reproduce or run the S3GM baseline under matched data and hardware;
- compare quality at matched NFE and matched wall time where possible;
- run NFE values `{10, 20, 50, 100}` for S3FM.

Done when:

- the primary speed-accuracy plot exists;
- timing includes warm-up and synchronized accelerator measurements;
- claims are supported by at least three seeds.

### Milestone 7: stronger measurements and systems

Order:

1. Fourier KSE measurements.
2. Kolmogorov flow.
3. ERA5 missing-variable reconstruction.
4. Nonlinear velocity-magnitude/cylinder-flow measurements.

Do not expand before the KSE speed-accuracy result is convincing.

### Milestone 8: adaptive guidance contribution

Candidate design:

```math
w(s)=w_{base}(s)\,f(r_{obs}(s),r_{seq}(s)).
```

Possible behaviors:

- increase guidance when normalized residual stagnates;
- cap guidance relative to the base velocity norm;
- allocate extra ODE evaluations to intervals with high residual reduction potential.

Done when:

- the rule is specified before final evaluation;
- it improves the Pareto frontier, not only one selected case;
- ablations separate schedule adaptation from increased compute.

## 19. Experimental matrix

### 19.1 Primary KSE matrix

| Axis | Values |
|---|---|
| Guidance | none, cov-A, cov-G |
| NFE | 10, 20, 50, 100 |
| Spatial downsampling | 4x, 8x, 16x, 32x, 64x |
| Observation noise | 0%, 5%, 10% |
| Sequence energy | off, on |
| Seeds | at least 3 |

### 19.2 Required metrics

- nRMSE;
- normalized observation residual;
- spatial two-point correlation error;
- overlap discontinuity;
- wall-clock time;
- NFE;
- peak memory;
- posterior-sample standard deviation.

### 19.3 Primary comparison rule

Use the same:

- train/validation/test trajectories;
- physical resolution and normalization;
- observation `y` and operator `H`;
- target length;
- accelerator;
- number of posterior samples;
- metric implementation.

## 20. Runtime measurement protocol

For GPU timing:

1. Warm up the model.
2. Synchronize before starting the timer.
3. Synchronize before stopping the timer.
4. Separate data-loading time from model inference time.
5. Report batch size and number of windows.
6. Report solver and true NFE.
7. Report mean and standard deviation over repeated runs.

Do not compare cached S3FM inference with uncached S3GM inference.

## 21. Risks and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Wrong guidance sign | observation residual rises | mandatory one-dimensional sign test |
| Guidance dominates prior | plausible structure collapses | log norm ratio; cap or schedule guidance |
| Guidance too weak | result ignores `y` | normalize energy; sweep weight |
| Flow bends strongly | low-NFE result degrades | increase NFE; improve schedule; adaptive allocation |
| Window indexing bug | seams or duplicated frames | integer-label unit tests |
| Prior distribution mismatch | plausible but wrong reconstruction | OOD tests and uncertainty reporting |
| Nonlinear `H` unstable | gradients explode or vanish | stable operator implementation; gradient clipping; continuation in weight |
| Autoregressive drift | long-horizon error grows rapidly | more initial frames; periodic re-observation; report statistical metrics |
| Unfair speed claim | hidden NFE or solver cost | strict runtime protocol |
| Incremental novelty | results look like a sampler swap | adaptive guidance or scalable preconditioning contribution |

## 22. Claims discipline

Allowed before experiments:

- “S3FM is designed to reduce the number of sampling evaluations.”
- “The method preserves operator-agnostic inference-time guidance in principle.”

Not allowed before experiments:

- “S3FM is faster than S3GM.”
- “S3FM achieves the same accuracy in 50 steps.”
- “The method works for arbitrary nonlinear sensors.”
- “The method is physically consistent.”

After experiments, every numerical claim must identify dataset, sensor operator, NFE, solver, hardware, and number of seeds.

## 23. Paper outline

1. **Introduction**
   - Sparse-sensor reconstruction is important.
   - S3GM is flexible but slow.
   - Flow matching offers efficient transport, but spatiotemporal posterior guidance remains underexplored.

2. **Related Work**
   - Spatiotemporal reconstruction.
   - Score-based inverse problems and S3GM.
   - Flow matching and flow guidance.

3. **Method**
   - Problem formulation.
   - Spatiotemporal flow prior.
   - Sensor-guided vector field.
   - Parallel window consistency.
   - Autoregressive continuation.
   - Optional adaptive guidance.

4. **Experiments**
   - KSE pilot and speed-accuracy frontier.
   - Turbulence and real-world data.
   - Linear, nonlinear, noisy, and missing-variable observations.
   - Ablations.

5. **Limitations**
   - Approximate guidance.
   - Prior mismatch.
   - Long-horizon error.
   - Remaining inference-time gradients.

6. **Conclusion**

## 24. AI-agent execution protocol

Every implementation request to an AI agent should include:

1. The current milestone.
2. The relevant sections of this specification.
3. The repository path and environment constraints.
4. The exact files allowed to change.
5. The expected tests and acceptance criteria.
6. A requirement to inspect existing code before editing.
7. A requirement to preserve unrelated user changes.

The agent must:

- report assumptions before major implementation choices;
- prefer small, testable changes;
- run targeted tests after each module;
- record exact commands and configs;
- distinguish verified results from expectations;
- stop and report if tensor semantics, normalization, or measurement alignment are ambiguous.

The agent must not:

- silently invent dataset details;
- change physical-time spacing to make results look better;
- compare methods using different observations;
- call solver steps “NFE” without accounting for stages;
- detach the flow model in `g_cov-G`;
- use all three energies in all phases;
- claim zero-shot generalization across unrelated dynamical systems.

## 25. Reusable AI prompt template

```text
You are implementing Milestone <N> of S3FM.

Read S3FM_CORE_SPEC.md first. Treat it as the source of truth.

Objective:
<one bounded objective>

Before editing:
1. Inspect the repository and existing tests.
2. State the tensor convention, normalization, and relevant interfaces.
3. Identify reusable code and any conflict with the specification.

Implementation constraints:
- Preserve unrelated user changes.
- Do not change scientific definitions without reporting the reason.
- Keep physical time separate from flow time.
- Count true NFE.
- Add or update targeted tests.

Acceptance criteria:
<copy from the milestone>

After implementation, report:
- files changed;
- tests run and results;
- exact command/config for reproduction;
- unresolved risks or assumptions.
```

## 26. Immediate next action

Begin only with Milestone 0: audit the intended implementation repository and identify whether S3GM data generation, normalization, Video U-Net, and evaluation code are already available.

Do not start model training until the audit and correctness-test plan are complete.
