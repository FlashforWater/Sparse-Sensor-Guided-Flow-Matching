"""Amortized observation-to-source inference q_phi(Z0 | y, H).

This is the learned counterpart to the fixed ``W_B(y,H)`` source constructor.
The model is deliberately restricted to the source space: it predicts a
low-bandwidth approximation to ``S(X)``, not the full target field ``X``.

Training target:

    target = S(X)
    loss   = || q_phi(y,H) - S(X) ||^2

At inference we add the same residual noise scale used by the S3FM-Info prior:

    Z0 = q_phi(y,H) + eta

The architectural projection is part of the safety boundary: for spectral
sources the returned mean is always Fourier-truncated, so q_phi cannot represent
high-frequency detail even if the network emits it internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..flow.sources import SourceConfig, apply_source
from ..measurements.base import MeasurementOperator


@dataclass(frozen=True)
class SourceInferenceConfig:
    """Small source-inference network config."""

    base_channels: int = 32
    depth: int = 3


@dataclass
class LoadedSourceInference:
    model: "MaskedSourceInferenceNet"
    source_cfg: SourceConfig
    config: dict
    device: torch.device


class _ResBlock1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, padding_mode="circular")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class _TemporalMix(nn.Module):
    def __init__(self, ch: int, kernel_t: int = 3):
        super().__init__()
        pad = kernel_t // 2
        self.conv = nn.Conv2d(ch, ch, kernel_size=(kernel_t, 1), padding=(pad, 0), groups=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class MaskedSourceInferenceNet(nn.Module):
    """Predict low-bandwidth source mean from sparse observations and mask.

    Input convention follows the rest of the project: ``observation`` is
    ``[B,T,C,Nx]`` and ``mask`` is ``[T,C,Nx]`` or ``[B,T,C,Nx]``. The mask is
    concatenated as an input channel so the same model can handle different
    sensor layouts.
    """

    def __init__(
        self,
        source_cfg: SourceConfig,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.source_cfg = source_cfg
        self.in_channels = in_channels
        self.input_channels = 2 * in_channels
        self.in_proj = nn.Conv1d(self.input_channels, base_channels, kernel_size=1)
        self.spatial_blocks = nn.ModuleList([_ResBlock1d(base_channels) for _ in range(depth)])
        self.temporal_blocks = nn.ModuleList([_TemporalMix(base_channels) for _ in range(depth)])
        self.out_norm = nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels)
        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=1)
        self.act = nn.SiLU()

    def _expand_mask(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C, Nx = observation.shape
        if mask.ndim == 3:
            mask = mask.to(observation.device, observation.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        elif mask.ndim == 4:
            mask = mask.to(observation.device, observation.dtype)
        else:
            raise ValueError(f"mask must be [T,C,Nx] or [B,T,C,Nx], got {tuple(mask.shape)}")
        if mask.shape != observation.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != observation shape {tuple(observation.shape)}")
        return mask

    def forward(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4:
            raise ValueError(f"observation must be [B,T,C,Nx], got {tuple(observation.shape)}")
        B, T, C, Nx = observation.shape
        if C != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got {C}")
        mask_b = self._expand_mask(observation, mask)
        x = torch.cat([observation, mask_b], dim=2)  # [B,T,2C,Nx]

        h = x.reshape(B * T, 2 * C, Nx)
        h = self.in_proj(h)
        for sblock, tblock in zip(self.spatial_blocks, self.temporal_blocks):
            h = sblock(h)
            ch = h.shape[1]
            h = h.reshape(B, T, ch, Nx).permute(0, 2, 1, 3)
            h = tblock(h)
            h = h.permute(0, 2, 1, 3).reshape(B * T, ch, Nx)
        h = self.out_proj(self.act(self.out_norm(h))).reshape(B, T, C, Nx)

        # Enforce the source-space bottleneck in the output itself.
        return apply_source(h, self.source_cfg)


def source_inference_target(x_true: torch.Tensor, source_cfg: SourceConfig) -> torch.Tensor:
    """The supervised target for q_phi: deterministic S(X), not X."""
    return apply_source(x_true, source_cfg)


def source_inference_loss(
    model: MaskedSourceInferenceNet,
    x_true: torch.Tensor,
    observation: torch.Tensor,
    operator: MeasurementOperator,
) -> torch.Tensor:
    """MSE loss for q_phi(y,H) against S(X)."""
    if not hasattr(operator, "mask"):
        raise ValueError("source inference currently requires MaskOperator-like `mask`")
    target = source_inference_target(x_true, model.source_cfg)
    pred = model(observation, operator.mask)
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def learned_observation_informed_source(
    model: MaskedSourceInferenceNet,
    observation: torch.Tensor,
    operator: MeasurementOperator,
    seed: int = 0,
) -> torch.Tensor:
    """Sample source ``Z0 = q_phi_mean(y,H) + eta``.

    The mean is low-bandwidth by architecture. The noise scale is inherited from
    ``model.source_cfg.eta_std`` to match the S3FM-Info prior's source.
    """
    if not hasattr(operator, "mask"):
        raise ValueError("learned source inference currently requires MaskOperator-like `mask`")
    mean = model(observation, operator.mask)
    if model.source_cfg.eta_std <= 0:
        return mean
    g = torch.Generator(device=mean.device).manual_seed(seed)
    eta = torch.randn(mean.shape, generator=g, device=mean.device, dtype=mean.dtype) * model.source_cfg.eta_std
    return mean + eta


def source_cfg_from_dict(data: dict) -> SourceConfig:
    return SourceConfig(
        kind=data.get("kind", "spectral"),
        cutoff_k=data.get("cutoff_k", 8),
        factor=data.get("factor", 4),
        eta_std=data.get("eta_std", 0.05),
    )


def source_cfg_to_dict(cfg: SourceConfig) -> dict:
    return {
        "kind": cfg.kind,
        "cutoff_k": cfg.cutoff_k,
        "factor": cfg.factor,
        "eta_std": cfg.eta_std,
    }


def load_source_inference(ckpt_path: str, device: str = "auto") -> LoadedSourceInference:
    """Load a trained q_phi checkpoint."""
    from ..reproducibility import select_device

    dev = select_device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    source_cfg = source_cfg_from_dict(ckpt["source_cfg"])
    mcfg = ckpt["model_config"]
    model = MaskedSourceInferenceNet(
        source_cfg=source_cfg,
        in_channels=mcfg.get("in_channels", 1),
        base_channels=mcfg.get("base_channels", 32),
        depth=mcfg.get("depth", 3),
    )
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return LoadedSourceInference(model=model, source_cfg=source_cfg, config=ckpt.get("config", {}), device=dev)
