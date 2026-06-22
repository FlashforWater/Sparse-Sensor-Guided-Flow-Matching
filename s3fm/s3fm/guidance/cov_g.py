"""Covariance-gradient guidance g_cov-G (the primary S3FM guidance method).

For a current flow state Zs at flow time s, with a frozen velocity model v_theta,
the guided velocity is:

    v_theta(Zs, s)  +  g_cov_G,
    g_cov_G = -lambda_s * grad_{Zs} J( X1_hat(Zs, s) )

where X1_hat = Zs + (1-s) v_theta(Zs, s) is the clean endpoint estimate, and the
gradient flows through BOTH the endpoint estimator AND v_theta (one VJP through
the network). The minus sign is the explicit descent direction; its correctness
is verified numerically by the guidance-sign test (never trusted from a paper's
equation, because flow-time conventions differ).

IMPORTANT (spec): in the main method, do NOT detach `velocity` or `x1_hat`. The
fast approximation g_cov-A (which detaches v_theta) lives in cov_a.py.

This module returns the guided velocity as a closure compatible with the ODE
solvers (v(z, s) -> tensor), so guided sampling reuses the same solver code.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..flow.paths import endpoint_estimate

EnergyFn = Callable[[torch.Tensor], torch.Tensor]


def cov_g_guidance(
    model: torch.nn.Module,
    x_s: torch.Tensor,
    flow_time: torch.Tensor,
    energy_fn: EnergyFn,
    lambda_s: float,
):
    """Compute base velocity, guidance, and guided velocity at one (x_s, s).

    ``energy_fn`` maps the endpoint estimate X1_hat -> scalar energy J. Returns a
    dict with the base velocity, the guidance vector, the guided velocity, and
    the scalar energy (detached, for logging). All tensors are detached on return
    so the caller's ODE step does not accumulate graph across steps.
    """
    x_s = x_s.detach().requires_grad_(True)
    velocity = model(x_s, flow_time)                 # through the network
    x1_hat = endpoint_estimate(x_s, velocity, flow_time)
    energy = energy_fn(x1_hat)                        # scalar J
    grad = torch.autograd.grad(energy, x_s, create_graph=False)[0]
    guidance = -lambda_s * grad                      # explicit descent
    guided = (velocity + guidance).detach()
    return {
        "base_velocity": velocity.detach(),
        "guidance": guidance.detach(),
        "guided_velocity": guided,
        "energy": float(energy.detach()),
    }


def make_guided_velocity(
    model: torch.nn.Module,
    energy_fn: EnergyFn,
    schedule: Callable[[float], float],
    diagnostics: list | None = None,
):
    """Build a v(z, s) closure that returns the g_cov-G guided velocity.

    ``schedule(s_scalar) -> lambda_s``. If ``diagnostics`` is a list, one dict of
    per-call norms/energy is appended each evaluation (for logging trajectories).
    """
    def vfield(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_scalar = float(s.reshape(-1)[0].detach().cpu())
        lam = schedule(s_scalar)
        out = cov_g_guidance(model, z, s, energy_fn, lam)
        if diagnostics is not None:
            base_n = float(torch.norm(out["base_velocity"]))
            guid_n = float(torch.norm(out["guidance"]))
            diagnostics.append({
                "flow_time": s_scalar,
                "lambda": lam,
                "energy": out["energy"],
                "base_velocity_norm": base_n,
                "guidance_norm": guid_n,
                "guidance_to_velocity_ratio": guid_n / (base_n + 1e-8),
            })
        return out["guided_velocity"]

    return vfield
