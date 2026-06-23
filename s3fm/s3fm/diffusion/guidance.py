"""Measurement guidance for VP probability-flow ODE baselines."""

from __future__ import annotations

from typing import Callable

import torch

from .vp import VPSchedule, predict_x0_from_eps, reverse_pf_velocity_from_eps

EnergyFn = Callable[[torch.Tensor], torch.Tensor]


def pf_ode_guidance(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    generation_time: torch.Tensor,
    schedule: VPSchedule,
    energy_fn: EnergyFn,
    lambda_s: float,
) -> dict:
    """Compute base PF-ODE velocity plus endpoint-energy guidance.

    The endpoint estimator is the Tweedie-style denoised estimate
    ``x0_hat(x_t,t)``. The gradient flows through both ``x_t`` and the epsilon
    predictor, matching the g_cov-G spirit used for flow matching.
    """
    x_t = x_t.detach().requires_grad_(True)
    t = schedule.noise_time_from_generation_time(generation_time)
    eps_pred = model(x_t, t)
    x0_hat = predict_x0_from_eps(x_t, t, eps_pred, schedule)
    energy = energy_fn(x0_hat)
    grad = torch.autograd.grad(energy, x_t, create_graph=False)[0]
    base_velocity = reverse_pf_velocity_from_eps(x_t, generation_time, eps_pred, schedule)
    guidance = -lambda_s * grad
    guided = (base_velocity + guidance).detach()
    return {
        "base_velocity": base_velocity.detach(),
        "guidance": guidance.detach(),
        "guided_velocity": guided,
        "energy": float(energy.detach()),
    }


def make_guided_pf_velocity(
    model: torch.nn.Module,
    schedule: VPSchedule,
    energy_fn: EnergyFn,
    guidance_schedule: Callable[[float], float],
    diagnostics: list | None = None,
):
    """Build an ODE velocity closure compatible with ``flow.solvers.solve``."""

    def vfield(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_scalar = float(s.reshape(-1)[0].detach().cpu())
        lam = guidance_schedule(s_scalar)
        out = pf_ode_guidance(model, z, s, schedule, energy_fn, lam)
        if diagnostics is not None:
            base_n = float(torch.norm(out["base_velocity"]))
            guid_n = float(torch.norm(out["guidance"]))
            diagnostics.append({
                "generation_time": s_scalar,
                "lambda": lam,
                "energy": out["energy"],
                "base_velocity_norm": base_n,
                "guidance_norm": guid_n,
                "guidance_to_velocity_ratio": guid_n / (base_n + 1e-8),
            })
        return out["guided_velocity"]

    return vfield
