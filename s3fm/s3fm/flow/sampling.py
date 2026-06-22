"""Unguided sampling from a trained flow prior + checkpoint loading.

Sampling = integrate dZ/ds = v_theta(Z, s) from a Gaussian Z0 at s=0 to s=1,
with no measurement guidance. The result lives in normalized space; callers
de-normalize with the saved ChannelStandardizer to get physical KSE fields.

This is the M3 path: it exercises the solver + trained prior with exact NFE
accounting, before any guidance is added in M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.splits import ChannelStandardizer
from ..models.video_unet_velocity import VideoUNetVelocity1D
from .solvers import solve


@dataclass
class LoadedPrior:
    model: torch.nn.Module
    standardizer: ChannelStandardizer
    config: dict
    device: torch.device


def load_prior(ckpt_path: str, device: str = "auto", use_ema: bool = True) -> LoadedPrior:
    """Load a trained prior checkpoint (EMA weights by default)."""
    from ..reproducibility import select_device

    dev = select_device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = VideoUNetVelocity1D(
        in_channels=1, base_channels=mcfg["base_channels"],
        depth=mcfg["depth"], t_emb_dim=mcfg["t_emb_dim"],
    )
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.to(dev).eval()

    std = ChannelStandardizer(mean=ckpt["norm_mean"], std=ckpt["norm_std"])
    return LoadedPrior(model=model, standardizer=std, config=ckpt["config"], device=dev)


@torch.no_grad()
def sample_unguided(
    prior: LoadedPrior,
    shape: tuple[int, int, int, int],
    steps: int,
    solver: str = "euler",
    seed: int = 0,
    denormalize: bool = True,
):
    """Draw samples by integrating the unguided flow.

    ``shape`` is ``[batch, T, C, Nx]``. Returns ``(samples, solve_result)`` where
    samples are in physical space if ``denormalize`` else normalized space.
    """
    dev = prior.device
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(shape, generator=g).to(dev)

    def vfield(z, s):
        return prior.model(z, s)

    result = solve(vfield, z0, steps=steps, solver=solver)
    samples = result.z1
    if denormalize:
        arr = samples.cpu().numpy()
        arr = prior.standardizer.inverse(arr, channel_axis=2)
        samples = torch.tensor(arr)
    return samples, result
