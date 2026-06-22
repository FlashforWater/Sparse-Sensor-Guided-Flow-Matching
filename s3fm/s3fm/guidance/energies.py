"""Observation energy J_obs and normalized residual.

The observation energy measures how well the estimated clean endpoint matches the
sparse measurements:

    J_obs(X1_hat) = (1 / (2 sigma_y^2)) * mean_square( y - H(X1_hat) )

We use the MEAN squared residual (not the raw sum) so the guidance scale does not
silently change with the number of observed points (spec §5). All energies must
be scalar, differentiable, normalized, and logged separately.

The normalized observation residual (a diagnostic, not the energy) is:

    r_obs = || y - H(X1_hat) || / (|| y || + eps)

For the MaskOperator, H(X) zeroes unobserved entries and y is likewise zero
there, so the masked MSE only penalizes observed locations.
"""

from __future__ import annotations

import torch

from ..measurements.base import MeasurementOperator


def observation_energy(
    x1_hat: torch.Tensor,
    observation: torch.Tensor,
    operator: MeasurementOperator,
    sigma_y: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Scalar J_obs = reduce(y - H(X1_hat))^2 / (2 sigma_y^2).

    ``reduction`` is "mean" (default) or "sum". This choice matters for the
    GUIDANCE SCALE, not just bookkeeping:

    - "mean" keeps J_obs (and hence the needed lambda) independent of the number
      of observed points / resolution — the spec's preferred normalization. But
      mean divides the gradient by N = T*C*Nx, so the lambda that produces a
      given guidance-to-velocity ratio is ~N times larger than with "sum".
    - "sum" gives an un-diluted gradient (a small lambda already guides
      strongly), but the effective strength then scales with how many entries
      are observed, which is why the spec dislikes it as the default.

    Either is valid as long as lambda is swept accordingly; the guidance-sign and
    descent behaviour are identical. We default to "mean" and sweep lambda.
    """
    pred = operator.forward(x1_hat)
    resid = observation - pred
    sq = resid ** 2
    reduced = sq.mean() if reduction == "mean" else sq.sum()
    return reduced / (2.0 * sigma_y ** 2)


@torch.no_grad()
def normalized_residual(
    x1_hat: torch.Tensor,
    observation: torch.Tensor,
    operator: MeasurementOperator,
    eps: float = 1e-8,
) -> float:
    """Diagnostic r_obs = ||y - H(X1_hat)|| / (||y|| + eps)."""
    pred = operator.forward(x1_hat)
    num = torch.norm(observation - pred)
    den = torch.norm(observation) + eps
    return float(num / den)
