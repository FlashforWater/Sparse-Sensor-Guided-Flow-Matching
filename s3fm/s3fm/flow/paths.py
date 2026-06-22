"""Linear conditional flow-matching path, velocity target, endpoint, and loss.

Notation (non-negotiable, from the spec):
  Z0 = x_source     source sample  (Gaussian for S3FM-Gauss)
  Z1 = x_target     clean target
  Zs = x_s          intermediate flow state at flow_time s in [0, 1]
  v_theta           learned velocity
  X1_hat = x1_hat   clean endpoint estimate

Linear (affine) conditional path:
    Zs = (1 - s) * Z0 + s * Z1
Conditional velocity (the FM target):
    u = Z1 - Z0                       (constant along the path)
Endpoint estimate (Tweedie analogue for the linear path):
    X1_hat = Zs + (1 - s) * v_theta(Zs, s)

`s` (flow_time) is broadcast against the leading batch dim only; it never shares
a variable with the physical frame index (n / frame_idx).
"""

from __future__ import annotations

import torch


def _broadcast_s(s: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-batch flow time ``s`` of shape ``[B]`` to broadcast over
    ``like`` of shape ``[B, ...]``."""
    if s.ndim == 0:
        return s
    if s.ndim != 1 or s.shape[0] != like.shape[0]:
        raise ValueError(f"flow_time must be scalar or [B={like.shape[0]}], got {tuple(s.shape)}")
    return s.view(s.shape[0], *([1] * (like.ndim - 1)))


def interpolate_path(x_source: torch.Tensor, x_target: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
    """Zs = (1 - s) Z0 + s Z1."""
    s = _broadcast_s(flow_time, x_source)
    return (1.0 - s) * x_source + s * x_target


def velocity_target(x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """The conditional FM target u = Z1 - Z0 (independent of s for the linear path)."""
    return x_target - x_source


def endpoint_estimate(x_s: torch.Tensor, velocity: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
    """X1_hat = Zs + (1 - s) v.

    Source-agnostic: identical formula for S3FM-Gauss and S3FM-Info, because it
    follows only from the linear path, not from the source distribution.
    """
    s = _broadcast_s(flow_time, x_s)
    return x_s + (1.0 - s) * velocity


def flow_matching_loss(
    predicted_velocity: torch.Tensor,
    x_source: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between predicted velocity and the target Z1 - Z0."""
    target = velocity_target(x_source, x_target)
    return torch.mean((predicted_velocity - target) ** 2)


def sample_flow_time(batch_size: int, device=None, generator=None) -> torch.Tensor:
    """Sample s ~ Uniform(0, 1), shape [batch_size]."""
    return torch.rand(batch_size, device=device, generator=generator)
