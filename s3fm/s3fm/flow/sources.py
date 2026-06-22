"""Informative source operators S(X) for S3FM-Info (the headline contribution).

The source is a cheap, structured, information-bearing approximation of the
target, produced by a fixed information-destroying operator S:

    Z0 = S(X) + eta          (dependent coupling: each source pairs with ITS target)

vs the S3FM-Gauss baseline Z0 ~ N(0, I) (independent, information-free).

CRITICAL DISCIPLINE (spec §6.0):
  - S must NOT depend on the inference measurement operator H. It is a fixed,
    generic corruption family. Otherwise the "operator-agnostic prior" story
    collapses into a disguised conditional regressor (y -> X).
  - The SAME S (and eta scale) used in pretraining must be used at inference.
    But at inference we have no X, so the inference source must be drawn from a
    distribution matching the training source. We provide the "marginal-source"
    sampler for that (information-free, safest): sample Z0 from the empirical
    distribution of S(X)+eta over a reference set.

Source constructions (KSE), increasing fidelity:
  1. coarse-blur:        spatial downsample-by-factor then upsample back
  2. spectral-truncation: keep the lowest-k Fourier modes, zero the rest

Why this shortens transport (the mechanism we MEASURE, not assume): the
per-sample displacement Z1 - Z0 = X - S(X) - eta is only the "missing detail"
(the high-frequency content S removed), not the whole field. Smaller, straighter
transport -> fewer ODE steps for a given error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SourceConfig:
    kind: str = "spectral"      # "spectral" | "coarse_blur"
    cutoff_k: int = 8           # spectral: keep lowest cutoff_k modes
    factor: int = 4             # coarse_blur: spatial downsample factor
    eta_std: float = 0.05       # residual noise scale (keeps source non-degenerate)


def spectral_truncation(x: torch.Tensor, cutoff_k: int) -> torch.Tensor:
    """Keep the lowest ``cutoff_k`` spatial Fourier modes; zero the rest.

    ``x`` has shape ``[..., Nx]``. Operates along the last axis with rFFT. The
    result is the smooth low-frequency skeleton of the field.
    """
    Nx = x.shape[-1]
    xf = torch.fft.rfft(x, dim=-1)
    mask = torch.zeros(xf.shape[-1], device=x.device, dtype=torch.bool)
    keep = min(cutoff_k, xf.shape[-1])
    mask[:keep] = True
    xf = xf * mask
    return torch.fft.irfft(xf, n=Nx, dim=-1)


def coarse_blur(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample by ``factor`` (average pooling) then linearly upsample back.

    This is what naive interpolation of a coarse measurement would give. ``x`` is
    ``[..., Nx]``; Nx must be divisible by ``factor``.
    """
    Nx = x.shape[-1]
    if Nx % factor != 0:
        raise ValueError(f"Nx={Nx} not divisible by factor={factor}")
    lead = x.shape[:-1]
    xr = x.reshape(-1, 1, Nx)
    # circular-aware average pool: pool then interpolate (linear) back
    pooled = torch.nn.functional.avg_pool1d(xr, kernel_size=factor, stride=factor)
    up = torch.nn.functional.interpolate(pooled, size=Nx, mode="linear", align_corners=False)
    return up.reshape(*lead, Nx)


def apply_source(x: torch.Tensor, cfg: SourceConfig) -> torch.Tensor:
    """Deterministic S(X) (no eta), differentiable, no dependence on H."""
    if cfg.kind == "spectral":
        return spectral_truncation(x, cfg.cutoff_k)
    if cfg.kind == "coarse_blur":
        return coarse_blur(x, cfg.factor)
    raise ValueError(f"unknown source kind {cfg.kind!r}")


def make_source(x: torch.Tensor, cfg: SourceConfig, generator=None) -> torch.Tensor:
    """Z0 = S(X) + eta. ``eta`` is fresh Gaussian noise at scale ``eta_std``.

    Used at TRAINING time (we have X). The same cfg must be used to build the
    inference-source distribution.
    """
    s = apply_source(x, cfg)
    if cfg.eta_std > 0:
        eta = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * cfg.eta_std
        s = s + eta
    return s


class MarginalSourceSampler:
    """Inference-time source sampler matching the training-source distribution.

    Builds an empirical pool of S(X_ref)+eta from a reference set (e.g. training
    windows) and draws Z0 by sampling from that pool. This is information-free
    w.r.t. the current measurement y (the safest inference source, §6.0): it does
    not inject observation information, so it cannot cheat — it only matches the
    distribution the velocity field was trained on.
    """

    def __init__(self, reference_fields: torch.Tensor, cfg: SourceConfig, seed: int = 0):
        # reference_fields: [N, T, C, Nx] clean (normalized) windows
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.pool = make_source(reference_fields, cfg, generator=g)  # [N,...]

    def sample(self, batch: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, self.pool.shape[0], (batch,), generator=g)
        return self.pool[idx].clone()

    def summary_stats(self) -> dict:
        x = self.pool.reshape(-1).cpu().numpy()
        return {"mean": float(np.mean(x)), "std": float(np.std(x))}
