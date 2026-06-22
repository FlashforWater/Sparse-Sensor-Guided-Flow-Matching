"""Single-window guided reconstruction (M4) entry point.

Given a trained prior, a held-out clean window X_true, a sparse measurement
operator H, and observations y = H(X_true), reconstruct X by integrating the
g_cov-G guided flow from a Gaussian source. Sweeps guidance strength lambda and
compares against the unguided baseline. Reports observation residual and full-
field nRMSE, plus guidance/velocity-norm diagnostics.

This is the M4 gate object and the basic system that M4b will swap the source on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..flow.paths import endpoint_estimate
from ..flow.solvers import solve
from ..guidance.cov_g import make_guided_velocity
from ..guidance.energies import normalized_residual, observation_energy
from ..guidance.schedules import constant
from ..measurements.base import MeasurementOperator


@dataclass
class ReconResult:
    x_recon: torch.Tensor       # [B,T,C,Nx] reconstruction (normalized space)
    nrmse: float                # full-field nRMSE vs X_true
    obs_residual: float         # normalized observation residual
    nfe: int
    lambda0: float
    diagnostics: list           # per-NFE guidance/velocity norms + energy


def reconstruct_window(
    model: torch.nn.Module,
    x_true: torch.Tensor,
    observation: torch.Tensor,
    operator: MeasurementOperator,
    lambda0: float,
    steps: int,
    solver: str = "euler",
    sigma_y: float = 1.0,
    reduction: str = "mean",
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> ReconResult:
    """Reconstruct one (batch of) window(s) with g_cov-G guidance.

    All tensors are in normalized space. ``x_true`` is used only for nRMSE
    reporting, never inside the guidance (that would be cheating). ``reduction``
    selects the J_obs reduction ("mean" or "sum"); sweep lambda accordingly.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    x_true = x_true.to(device)
    observation = observation.to(device)

    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(x_true.shape, generator=g).to(device)

    def energy_fn(x1_hat):
        return observation_energy(x1_hat, observation, operator, sigma_y=sigma_y, reduction=reduction)

    diagnostics: list = []
    vfield = make_guided_velocity(model, energy_fn, constant(lambda0), diagnostics)
    result = solve(vfield, z0, steps=steps, solver=solver)
    x_recon = result.z1

    nrmse = float(torch.norm(x_recon - x_true) / torch.norm(x_true))
    obs_res = normalized_residual(x_recon, observation, operator)
    return ReconResult(
        x_recon=x_recon.detach(), nrmse=nrmse, obs_residual=obs_res,
        nfe=result.nfe, lambda0=lambda0, diagnostics=diagnostics,
    )
