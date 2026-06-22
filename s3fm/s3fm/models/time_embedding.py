"""Continuous flow-time embedding (sinusoidal + MLP).

The velocity model conditions on flow_time s in [0, 1]. We embed s with a
sinusoidal feature map followed by a small MLP, then inject it into the network
as a per-channel bias (FiLM-style shift). s is the FLOW time, never the physical
frame index.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(s: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    """Map ``s`` of shape ``[B]`` to ``[B, dim]`` sinusoidal features.

    ``s`` lives in [0, 1]; we scale it into the sinusoid argument. ``dim`` must
    be even.
    """
    if dim % 2 != 0:
        raise ValueError("embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=s.device, dtype=torch.float32) / half
    )
    args = s.float().unsqueeze(1) * freqs.unsqueeze(0) * (2.0 * math.pi)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class FlowTimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 4
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(s, self.dim))
