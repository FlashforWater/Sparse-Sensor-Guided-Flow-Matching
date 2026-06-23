from __future__ import annotations

import torch

from s3fm.diffusion.guidance import pf_ode_guidance
from s3fm.diffusion.training import diffusion_epsilon_loss
from s3fm.diffusion.vp import (
    VPSchedule,
    predict_x0_from_eps,
    q_sample,
    reverse_pf_velocity,
)
from s3fm.guidance.energies import observation_energy
from s3fm.measurements.base import IdentityOperator
from s3fm.models.video_unet_velocity import VideoUNetVelocity1D


def test_predict_x0_from_true_eps_recovers_clean():
    schedule = VPSchedule()
    x0 = torch.randn(3, 4, 1, 8)
    eps = torch.randn_like(x0)
    t = torch.full((x0.shape[0],), 0.7)
    x_t = q_sample(x0, t, eps, schedule)
    x0_hat = predict_x0_from_eps(x_t, t, eps, schedule)
    assert torch.allclose(x0_hat, x0, atol=1e-5)


def test_reverse_pf_velocity_shape_and_finite():
    schedule = VPSchedule()
    model = VideoUNetVelocity1D(in_channels=1, base_channels=8, depth=1, t_emb_dim=16)
    z = torch.randn(2, 3, 1, 8)
    s = torch.full((2,), 0.2)
    v = reverse_pf_velocity(model, z, s, schedule)
    assert v.shape == z.shape
    assert torch.isfinite(v).all()


def test_diffusion_loss_backpropagates():
    schedule = VPSchedule()
    model = VideoUNetVelocity1D(in_channels=1, base_channels=8, depth=1, t_emb_dim=16)
    x0 = torch.randn(2, 3, 1, 8)
    loss = diffusion_epsilon_loss(model, x0, schedule, generator=torch.Generator().manual_seed(0))
    loss.backward()
    total = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert total > 0


def test_pf_guidance_sign_lowers_identity_energy():
    class ZeroEps(torch.nn.Module):
        def forward(self, z, t):
            return torch.zeros_like(z)

    schedule = VPSchedule()
    model = ZeroEps()
    H = IdentityOperator()
    y = torch.randn(2, 3, 1, 8)
    z = torch.randn(2, 3, 1, 8)
    s = torch.full((2,), 0.5)

    def energy_fn(x0_hat):
        return observation_energy(x0_hat, y, H)

    with torch.no_grad():
        t = schedule.noise_time_from_generation_time(s)
        before = energy_fn(predict_x0_from_eps(z, t, torch.zeros_like(z), schedule)).item()

    out = pf_ode_guidance(model, z, s, schedule, energy_fn, lambda_s=1.0)
    z_new = z + 0.001 * out["guidance"]

    with torch.no_grad():
        t = schedule.noise_time_from_generation_time(s)
        after = energy_fn(predict_x0_from_eps(z_new, t, torch.zeros_like(z), schedule)).item()

    assert after < before
