"""Spatiotemporal velocity model for 1D KSE fields.

Input/output convention (spec): ``[batch, T, C, Nx]`` for both ``x_s`` and the
predicted ``velocity`` (same shape). ``flow_time`` is ``[batch]``.

This is a lightweight "video U-Net" analogue for 1D fields: spatial 1D
convolutions over Nx, lightweight temporal mixing across the T frames, and a
FiLM injection of the flow-time embedding. It is deliberately small so it trains
on a laptop MPS/CPU; capacity can be scaled via ``base_channels`` and ``depth``
when moving to a cluster.

A wrapper isolates the permutation between the spec convention [B,T,C,Nx] and the
internal conv layout, so no dimension juggling leaks into training/sampling code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .time_embedding import FlowTimeEmbedding


class _ResBlock1d(nn.Module):
    """Spatial 1D conv res-block with FiLM flow-time conditioning.

    Operates on tensors shaped ``[B*T, ch, Nx]`` (frames folded into batch for
    the spatial convs). The flow-time embedding is shared across frames.
    """

    def __init__(self, ch: int, t_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, padding_mode="circular")
        self.norm2 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, padding_mode="circular")
        self.film = nn.Linear(t_emb_dim, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, T: int) -> torch.Tensor:
        # x: [B*T, ch, Nx]; t_emb: [B, t_emb_dim]
        h = self.conv1(self.act(self.norm1(x)))
        shift = self.film(t_emb)  # [B, ch]
        shift = shift.repeat_interleave(T, dim=0).unsqueeze(-1)  # [B*T, ch, 1]
        h = h + shift
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class _TemporalMix(nn.Module):
    """Mix information across the T frames with a depthwise temporal conv.

    Operates on ``[B, ch, T, Nx]``: a depthwise conv over the T axis lets each
    spatial location share information across nearby physical frames. Circular
    padding is NOT used in time (trajectories are not periodic in time).
    """

    def __init__(self, ch: int, kernel_t: int = 3):
        super().__init__()
        pad = kernel_t // 2
        self.conv = nn.Conv2d(ch, ch, kernel_size=(kernel_t, 1), padding=(pad, 0), groups=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class VideoUNetVelocity1D(nn.Module):
    """Velocity model v_theta(x_s, flow_time) for 1D spatiotemporal KSE windows."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        depth: int = 3,
        t_emb_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.t_emb = FlowTimeEmbedding(t_emb_dim)
        self.in_proj = nn.Conv1d(in_channels, base_channels, kernel_size=1)
        self.spatial_blocks = nn.ModuleList(
            [_ResBlock1d(base_channels, t_emb_dim) for _ in range(depth)]
        )
        self.temporal_blocks = nn.ModuleList(
            [_TemporalMix(base_channels) for _ in range(depth)]
        )
        self.out_norm = nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels)
        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x_s: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        if x_s.ndim != 4:
            raise ValueError(f"x_s must be [B,T,C,Nx], got {tuple(x_s.shape)}")
        B, T, C, Nx = x_s.shape
        if C != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got {C}")
        if flow_time.ndim != 1 or flow_time.shape[0] != B:
            raise ValueError(f"flow_time must be [B={B}], got {tuple(flow_time.shape)}")

        t_emb = self.t_emb(flow_time)  # [B, t_emb_dim]

        # Fold frames into batch for spatial convs: [B*T, C, Nx]
        h = x_s.reshape(B * T, C, Nx)
        h = self.in_proj(h)  # [B*T, ch, Nx]

        for sblock, tblock in zip(self.spatial_blocks, self.temporal_blocks):
            h = sblock(h, t_emb, T)                       # spatial + FiLM
            ch = h.shape[1]
            h = h.reshape(B, T, ch, Nx).permute(0, 2, 1, 3)  # [B, ch, T, Nx]
            h = tblock(h)                                  # temporal mix
            h = h.permute(0, 2, 1, 3).reshape(B * T, ch, Nx)

        h = self.out_proj(self.act(self.out_norm(h)))      # [B*T, C, Nx]
        return h.reshape(B, T, C, Nx)


class OracleVelocity:
    """Test double: returns the analytic velocity Z1 - Z0.

    Not an nn.Module; used to validate the sampler/guidance against ground truth
    without training. Requires knowing Z0 and Z1.
    """

    def __init__(self, x_source: torch.Tensor, x_target: torch.Tensor):
        self.u = x_target - x_source

    def __call__(self, x_s: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        return self.u
