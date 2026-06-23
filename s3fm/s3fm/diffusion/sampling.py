"""Checkpoint loading and PF-ODE sampling for score priors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.splits import ChannelStandardizer
from ..flow.solvers import solve
from ..models.video_unet_velocity import VideoUNetVelocity1D
from .vp import VPSchedule, reverse_pf_velocity


@dataclass
class LoadedScorePrior:
    model: torch.nn.Module
    standardizer: ChannelStandardizer
    config: dict
    schedule: VPSchedule
    device: torch.device


def load_score_prior(ckpt_path: str, device: str = "auto", use_ema: bool = True) -> LoadedScorePrior:
    """Load a trained VP epsilon-prediction prior checkpoint."""
    from ..reproducibility import select_device

    dev = select_device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    mcfg = ckpt["config"]["model"]
    model = VideoUNetVelocity1D(
        in_channels=1,
        base_channels=mcfg["base_channels"],
        depth=mcfg["depth"],
        t_emb_dim=mcfg["t_emb_dim"],
    )
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.to(dev).eval()
    std = ChannelStandardizer(mean=ckpt["norm_mean"], std=ckpt["norm_std"])
    schedule = VPSchedule.from_config(ckpt["config"].get("diffusion"))
    return LoadedScorePrior(model=model, standardizer=std, config=ckpt["config"], schedule=schedule, device=dev)


@torch.no_grad()
def sample_pf_ode(
    prior: LoadedScorePrior,
    shape: tuple[int, int, int, int],
    steps: int,
    solver: str = "euler",
    seed: int = 0,
    denormalize: bool = True,
):
    """Draw samples with the reverse probability-flow ODE."""
    dev = prior.device
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(shape, generator=g).to(dev)
    result = solve(lambda z, s: reverse_pf_velocity(prior.model, z, s, prior.schedule), z0, steps=steps, solver=solver)
    samples = result.z1
    if denormalize:
        arr = samples.cpu().numpy()
        arr = prior.standardizer.inverse(arr, channel_axis=2)
        samples = torch.tensor(arr)
    return samples, result
