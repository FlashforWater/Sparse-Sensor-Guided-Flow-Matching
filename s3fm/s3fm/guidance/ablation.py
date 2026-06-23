"""Three-way ablation comparison (the M4b gate / headline result).

At matched NFE, compare reconstruction from sparse observations under:

  (A) S3FM-Gauss + g_cov-G guidance      Gaussian source, guided
  (B) S3FM-Info  + g_cov-G guidance      informative source, guided  <- contribution
  (C) S3FM-Info  WITHOUT guidance        informative source, no obs guidance

(A) vs (B) isolates the value of the informative source (same guidance).
(B) vs (C) isolates the value of guidance (same source).
Sweeping NFE shows whether the informative-source advantage GROWS as NFE shrinks
(the few-step claim).

For S3FM-Info, the source at inference must match the training-source
distribution. Two faithful options (spec §6.0):
  - marginal: sample Z0 from the empirical S(X_ref)+eta pool (information-free).
  - warm: build a generic reconstruction from y, then apply the same S(.)+eta.
  - oracle-source (diagnostic only): Z0 = S(X_true)+eta. This uses the true
    field and is therefore an UPPER BOUND / sanity check, NOT a valid method;
    we label it clearly and never report it as the real result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..flow.paths import endpoint_estimate
from ..flow.solvers import solve
from ..flow.sources import SourceConfig, make_source, MarginalSourceSampler
from ..guidance.cov_g import make_guided_velocity
from ..guidance.energies import normalized_residual, observation_energy
from ..guidance.schedules import constant
from ..guidance.source_inference import learned_observation_informed_source
from ..guidance.source_constructor import observation_informed_source
from ..measurements.base import MeasurementOperator


@dataclass
class AblationRow:
    mode: str
    nfe: int
    nrmse: float
    obs_residual: float


def _recon(model, z0, x_true, observation, operator, lambda0, steps, solver, reduction, device):
    z0 = z0.to(device)
    x_true = x_true.to(device)
    observation = observation.to(device)
    if lambda0 == 0.0:
        with torch.no_grad():
            xr = solve(lambda z, s: model(z, s), z0, steps=steps, solver=solver).z1
    else:
        def energy_fn(x1):
            return observation_energy(x1, observation, operator, reduction=reduction)
        vfield = make_guided_velocity(model, energy_fn, constant(lambda0))
        xr = solve(vfield, z0, steps=steps, solver=solver).z1
    nrmse = float(torch.norm(xr - x_true) / torch.norm(x_true))
    res = normalized_residual(xr, observation, operator)
    return nrmse, res


def three_way_ablation(
    gauss_model,
    info_model,
    x_true: torch.Tensor,
    observation: torch.Tensor,
    operator: MeasurementOperator,
    source_cfg: SourceConfig,
    reference_fields: torch.Tensor,
    lambda0: float,
    nfe_list: list[int],
    solver: str = "euler",
    reduction: str = "sum",
    seed: int = 0,
    device: torch.device | str = "cpu",
    info_source: str = "marginal",
    source_inference_model: torch.nn.Module | None = None,
) -> list[AblationRow]:
    """Run the three modes across an NFE sweep. Returns a flat list of rows."""
    device = torch.device(device)
    gauss_model = gauss_model.to(device).eval()
    info_model = info_model.to(device).eval()

    g = torch.Generator().manual_seed(seed)
    z0_gauss = torch.randn(x_true.shape, generator=g)

    # informative inference source (distribution-matched, information-free)
    if info_source == "marginal":
        sampler = MarginalSourceSampler(reference_fields, source_cfg, seed=seed)
        z0_info = sampler.sample(x_true.shape[0], seed=seed)
    elif info_source == "warm":
        z0_info = observation_informed_source(observation, operator, source_cfg, seed=seed)
    elif info_source == "learned":
        if source_inference_model is None:
            raise ValueError("info_source='learned' requires source_inference_model")
        source_inference_model = source_inference_model.to(device).eval()
        z0_info = learned_observation_informed_source(source_inference_model, observation.to(device), operator, seed=seed)
    elif info_source == "oracle":
        gi = torch.Generator().manual_seed(seed)
        z0_info = make_source(x_true, source_cfg, generator=gi)  # diagnostic upper bound
    else:
        raise ValueError(f"unknown info_source {info_source!r}")

    rows: list[AblationRow] = []
    for steps in nfe_list:
        na, ra = _recon(gauss_model, z0_gauss, x_true, observation, operator, lambda0, steps, solver, reduction, device)
        rows.append(AblationRow("A_gauss_guided", steps, na, ra))
        nb, rb = _recon(info_model, z0_info, x_true, observation, operator, lambda0, steps, solver, reduction, device)
        rows.append(AblationRow("B_info_guided", steps, nb, rb))
        nc, rc = _recon(info_model, z0_info, x_true, observation, operator, 0.0, steps, solver, reduction, device)
        rows.append(AblationRow("C_info_unguided", steps, nc, rc))
    return rows
