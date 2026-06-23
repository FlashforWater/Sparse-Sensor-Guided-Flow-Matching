"""Variance-preserving diffusion schedule and PF-ODE utilities.

This module implements the minimal score-model machinery needed for the
S3GM-PF-ODE attribution baseline. The score network is trained as an epsilon
predictor under a VP forward process:

    x_t = alpha(t) * x_0 + sigma(t) * eps,  eps ~ N(0, I)

Sampling uses the probability-flow ODE of the same VP process, integrated from
near pure noise to near data with the repository's fixed-NFE ODE solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _broadcast_time(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return t.reshape((t.shape[0],) + (1,) * (x.ndim - 1))


@dataclass(frozen=True)
class VPSchedule:
    """Linear-beta VP schedule.

    ``t`` is diffusion noise time: 0 is data, 1 is near Gaussian noise. The ODE
    solver uses generation time ``s`` in [0, 1], with ``t = t_max - s*(t_max-t_min)``.
    """

    beta_min: float = 0.1
    beta_max: float = 20.0
    t_min: float = 1e-3
    t_max: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict | None) -> "VPSchedule":
        cfg = cfg or {}
        return cls(
            beta_min=float(cfg.get("beta_min", 0.1)),
            beta_max=float(cfg.get("beta_max", 20.0)),
            t_min=float(cfg.get("t_min", 1e-3)),
            t_max=float(cfg.get("t_max", 1.0)),
        )

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.int_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - alpha.square(), min=1e-12))

    def sample_time(self, batch: int, device: torch.device, generator=None) -> torch.Tensor:
        u = torch.rand(batch, generator=generator).to(device)
        return self.t_min + (self.t_max - self.t_min) * u

    def noise_time_from_generation_time(self, s: torch.Tensor) -> torch.Tensor:
        return self.t_max - (self.t_max - self.t_min) * s


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    """Sample VP forward noising state ``x_t`` from clean ``x0`` and noise ``eps``."""
    a = _broadcast_time(x0, schedule.alpha(t))
    sig = _broadcast_time(x0, schedule.sigma(t))
    return a * x0 + sig * eps


def predict_x0_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    """Denoised endpoint estimate from an epsilon prediction."""
    a = _broadcast_time(x_t, schedule.alpha(t))
    sig = _broadcast_time(x_t, schedule.sigma(t))
    return (x_t - sig * eps_pred) / torch.clamp(a, min=1e-6)


def score_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    """Convert epsilon prediction to VP score estimate ``nabla log p_t(x_t)``."""
    sig = _broadcast_time(x_t, schedule.sigma(t))
    return -eps_pred / torch.clamp(sig, min=1e-6)


def reverse_pf_velocity_from_eps(
    x_t: torch.Tensor,
    s: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    """Velocity ``dx/ds`` for reverse probability-flow ODE.

    The forward VP PF-ODE drift in diffusion time t is

        f_pf = -0.5 beta(t) x - 0.5 beta(t) score_t(x).

    Generation integrates from high noise to low noise using ``s`` increasing,
    so ``dx/ds = -(t_max-t_min) * f_pf``.
    """
    t = schedule.noise_time_from_generation_time(s)
    beta = _broadcast_time(x_t, schedule.beta(t))
    score = score_from_eps(x_t, t, eps_pred, schedule)
    span = schedule.t_max - schedule.t_min
    return 0.5 * span * beta * (x_t + score)


def reverse_pf_velocity(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    s: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    """Evaluate reverse PF-ODE velocity using an epsilon-prediction model."""
    t = schedule.noise_time_from_generation_time(s)
    eps_pred = model(x_t, t)
    return reverse_pf_velocity_from_eps(x_t, s, eps_pred, schedule)
