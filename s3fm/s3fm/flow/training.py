"""Flow-matching training step and loop.

The training step implements the spec's algorithm for S3FM-Gauss:

    sample clean window Z1
    sample Gaussian source Z0 ~ N(0, I), same shape
    sample s ~ Uniform(0, 1)
    Zs = (1 - s) Z0 + s Z1
    target = Z1 - Z0
    loss = mean_square(v_theta(Zs, s), target)

`train_step` is pure (model + batch -> loss) so it is easy to test and reuse for
both tiny-overfit and full training. The loop is device-agnostic (CPU / MPS /
CUDA) via a passed-in device, so the same code runs on a laptop or a cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paths import flow_matching_loss, interpolate_path, sample_flow_time


def gaussian_source_like(x_target: torch.Tensor, generator=None) -> torch.Tensor:
    """Z0 ~ N(0, I) with the same shape/device/dtype as the clean target."""
    return torch.randn(
        x_target.shape, device=x_target.device, dtype=x_target.dtype, generator=generator
    )


def train_step(
    model: torch.nn.Module,
    x_target: torch.Tensor,
    generator=None,
) -> torch.Tensor:
    """One flow-matching loss evaluation on a batch of clean windows.

    Does NOT call backward/optimizer.step — the caller owns the optimization
    loop. Returns the scalar loss.
    """
    z1 = x_target
    z0 = gaussian_source_like(z1, generator=generator)
    s = sample_flow_time(z1.shape[0], device=z1.device, generator=generator)
    zs = interpolate_path(z0, z1, s)
    pred = model(zs, s)
    return flow_matching_loss(pred, z0, z1)


@dataclass
class TrainResult:
    losses: list[float]
    final_loss: float


def overfit(
    model: torch.nn.Module,
    x_target: torch.Tensor,
    steps: int,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    seed: int = 0,
    log_every: int = 0,
    fix_source: bool = False,
) -> TrainResult:
    """Repeatedly fit a *fixed* small batch to verify the model can learn.

    This is the spec-mandated tiny-overfit: the FM loss should drop sharply,
    confirming the path/target/model wiring is correct before any expensive run.
    A fixed RNG generator drives the Gaussian source + flow-time draws so the run
    is reproducible.

    ``fix_source``: if True, the source ``Z0`` and flow-times ``s`` are drawn
    ONCE and reused every step. This removes the irreducible FM loss variance
    (which comes from re-sampling Z0 each step) and lets the loss approach ~0 —
    the cleanest signal that the model can memorize a fixed (Zs, s) -> target
    map. If False (default), Z0/s are resampled each step (true FM objective),
    and the loss plateaus at the irreducible variance, not 0.
    """
    device = torch.device(device)
    model = model.to(device)
    x_target = x_target.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    z1 = x_target
    fixed_z0 = fixed_s = None
    if fix_source:
        fixed_z0 = torch.randn(z1.shape, generator=gen).to(device)
        fixed_s = torch.rand(z1.shape[0], generator=gen).to(device)

    losses: list[float] = []
    for step in range(steps):
        if fix_source:
            z0, s = fixed_z0, fixed_s
        else:
            # draw on CPU generator then move, so MPS/CUDA stay reproducible
            z0 = torch.randn(z1.shape, generator=gen).to(device)
            s = torch.rand(z1.shape[0], generator=gen).to(device)
        zs = interpolate_path(z0, z1, s)
        loss = flow_matching_loss(model(zs, s), z0, z1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}  loss {losses[-1]:.6f}")
    return TrainResult(losses=losses, final_loss=losses[-1])
