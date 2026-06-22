"""M4 gate: guidance correctness.

The two most important checks here are spec correctness checks #2 and #3:

- **Guidance sign**: with H(x)=x and J(x)=0.5*(x-y)^2, a single small guidance-only
  step (base velocity disabled) must DECREASE J. If the sign were wrong, J would
  rise. This is tested numerically, never assumed.
- **No-guidance equivalence**: lambda=0 reproduces the unguided solver output
  exactly for the same source and solver.

Plus: energy/residual correctness, gradient flows through the network, and finite
gradients.
"""

from __future__ import annotations

import torch

from s3fm.flow.paths import endpoint_estimate
from s3fm.flow.solvers import euler_solve
from s3fm.guidance.cov_g import cov_g_guidance, make_guided_velocity
from s3fm.guidance.energies import normalized_residual, observation_energy
from s3fm.guidance.schedules import constant
from s3fm.measurements.base import IdentityOperator, MaskOperator


# --------------------------------------------------------------- energies

def test_observation_energy_zero_when_matched():
    x = torch.randn(2, 4, 1, 8)
    H = IdentityOperator()
    assert observation_energy(x, H(x), H).item() < 1e-12


def test_observation_energy_positive_otherwise():
    x = torch.randn(2, 4, 1, 8)
    y = torch.zeros_like(x)
    H = IdentityOperator()
    assert observation_energy(x, y, H).item() > 0


def test_normalized_residual_zero_when_matched():
    x = torch.randn(2, 4, 1, 8)
    H = MaskOperator.random(4, 1, 8, observed_fraction=0.5, seed=0)
    y = H(x)
    assert normalized_residual(x, y, H) < 1e-6


# ----------------------------------------------------- guidance sign test

def test_guidance_sign_lowers_energy_identity():
    """Correctness check #3: a small guidance-only step reduces J.

    1D-style toy: H(x)=x, J(x)=0.5||x-y||^2. We disable the base velocity by
    using a zero-velocity model, so the update is pure guidance. After one small
    Euler-like step Zs <- Zs + ds*g, the energy at the new endpoint must drop.
    """
    torch.manual_seed(0)

    class ZeroVel(torch.nn.Module):
        def forward(self, z, s):
            return torch.zeros_like(z)

    model = ZeroVel()
    H = IdentityOperator()
    y = torch.randn(3, 4, 1, 8)
    z = torch.randn(3, 4, 1, 8)
    s = torch.full((3,), 0.5)

    def energy_fn(x1):
        return observation_energy(x1, y, H)

    # energy before
    with torch.no_grad():
        v0 = model(z, s)
        x1_before = endpoint_estimate(z, v0, s)
        J_before = energy_fn(x1_before).item()

    out = cov_g_guidance(model, z, s, energy_fn, lambda_s=1.0)
    ds = 0.01
    z_new = z + ds * out["guided_velocity"]  # guided = base(0) + guidance

    with torch.no_grad():
        v1 = model(z_new, s)
        x1_after = endpoint_estimate(z_new, v1, s)
        J_after = energy_fn(x1_after).item()

    assert J_after < J_before, f"guidance increased energy: {J_before} -> {J_after}"


def test_guidance_gradient_flows_through_network():
    """The guidance must depend on the network params (g_cov-G, not g_cov-A).

    With a real (nonzero) model, the guidance vector should be nonzero and the
    energy gradient must have flowed through v_theta (we just check it ran and is
    finite and nonzero here; the through-network property is structural)."""
    from s3fm.models.video_unet_velocity import VideoUNetVelocity1D

    torch.manual_seed(0)
    model = VideoUNetVelocity1D(in_channels=1, base_channels=16, depth=2)
    H = MaskOperator.random(4, 1, 16, observed_fraction=0.3, seed=0)
    y = H(torch.randn(2, 4, 1, 16))
    z = torch.randn(2, 4, 1, 16)
    s = torch.full((2,), 0.4)
    out = cov_g_guidance(model, z, s, lambda x1: observation_energy(x1, y, H), lambda_s=0.5)
    assert torch.isfinite(out["guidance"]).all()
    assert torch.norm(out["guidance"]) > 0


def test_no_guidance_equivalence():
    """lambda=0 reproduces unguided sampling exactly (same source, same solver)."""
    from s3fm.models.video_unet_velocity import VideoUNetVelocity1D

    torch.manual_seed(0)
    model = VideoUNetVelocity1D(in_channels=1, base_channels=16, depth=2).eval()
    H = MaskOperator.random(4, 1, 16, observed_fraction=0.3, seed=1)
    y = H(torch.randn(2, 4, 1, 16))
    z0 = torch.randn(2, 4, 1, 16)

    # unguided
    with torch.no_grad():
        unguided = euler_solve(lambda z, s: model(z, s), z0, steps=10).z1

    # guided with lambda=0
    vfield = make_guided_velocity(model, lambda x1: observation_energy(x1, y, H), constant(0.0))
    guided0 = euler_solve(vfield, z0, steps=10).z1

    assert torch.allclose(unguided, guided0, atol=1e-5)
