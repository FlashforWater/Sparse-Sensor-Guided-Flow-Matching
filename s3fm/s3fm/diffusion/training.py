"""Training losses for VP epsilon-prediction score priors."""

from __future__ import annotations

import torch

from .vp import VPSchedule, predict_x0_from_eps, q_sample


def diffusion_epsilon_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    schedule: VPSchedule,
    generator=None,
) -> torch.Tensor:
    """One VP epsilon-prediction loss evaluation."""
    eps = torch.randn(x0.shape, generator=generator).to(x0.device)
    t = schedule.sample_time(x0.shape[0], x0.device, generator=generator)
    x_t = q_sample(x0, t, eps, schedule)
    eps_pred = model(x_t, t)
    return torch.mean((eps_pred - eps).square())


@torch.no_grad()
def denoise_recovery_nrmse(
    model: torch.nn.Module,
    x0: torch.Tensor,
    schedule: VPSchedule,
    device: torch.device,
    t_val: float = 0.9,
    seed: int = 123,
) -> float:
    """Validation nRMSE of ``x0_hat`` from a fixed noising time."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x0 = x0.to(device)
    eps = torch.randn(x0.shape, generator=g).to(device)
    t = torch.full((x0.shape[0],), t_val, device=device)
    x_t = q_sample(x0, t, eps, schedule)
    x0_hat = predict_x0_from_eps(x_t, t, model(x_t, t), schedule)
    return float(torch.norm(x0_hat - x0) / torch.norm(x0))


@torch.no_grad()
def val_epsilon_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    schedule: VPSchedule,
    device: torch.device,
    seed: int = 7,
) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x0 = x0.to(device)
    return float(diffusion_epsilon_loss(model, x0, schedule, generator=g))
