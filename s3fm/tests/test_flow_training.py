"""M2 training tests: tiny-overfit memorization + endpoint recovery.

These run small/fast (CPU, tiny model) so they belong in the test suite. They
verify the spec's M2 gate: the model can learn (loss drops under fixed source),
and the learned velocity yields endpoint estimates that improve toward s=1.
"""

from __future__ import annotations

import torch

from s3fm.flow.paths import endpoint_estimate, interpolate_path
from s3fm.flow.training import overfit, train_step, gaussian_source_like
from s3fm.models.video_unet_velocity import VideoUNetVelocity1D, OracleVelocity
from s3fm.reproducibility import seed_everything


def _tiny_batch(b=4, T=8, Nx=32):
    seed_everything(0)
    return torch.randn(b, T, 1, Nx)


def test_train_step_returns_scalar_loss():
    seed_everything(0)
    model = VideoUNetVelocity1D(in_channels=1, base_channels=16, depth=2)
    x = _tiny_batch()
    loss = train_step(model, x)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_gaussian_source_shape():
    x = _tiny_batch()
    z0 = gaussian_source_like(x)
    assert z0.shape == x.shape


def test_overfit_fixed_source_memorizes():
    """GATE: with fixed source, FM loss drops sharply (model can learn)."""
    seed_everything(0)
    model = VideoUNetVelocity1D(in_channels=1, base_channels=32, depth=2)
    x = _tiny_batch(b=4, T=8, Nx=32)
    res = overfit(model, x, steps=300, lr=1e-3, device="cpu", seed=0, fix_source=True)
    assert res.final_loss < 0.1 * res.losses[0]
    assert res.final_loss < 0.05


def test_endpoint_recovery_improves_toward_s1():
    """The learned velocity's endpoint estimate must get better as s -> 1."""
    seed_everything(0)
    model = VideoUNetVelocity1D(in_channels=1, base_channels=32, depth=2)
    x = _tiny_batch(b=4, T=8, Nx=32)
    overfit(model, x, steps=600, lr=1e-3, device="cpu", seed=0, fix_source=False)
    model.eval()

    g = torch.Generator().manual_seed(3)
    z0 = torch.randn(x.shape, generator=g)
    nrmse = {}
    with torch.no_grad():
        for s_val in [0.1, 0.9]:
            s = torch.full((x.shape[0],), s_val)
            zs = interpolate_path(z0, x, s)
            x1 = endpoint_estimate(zs, model(zs, s), s)
            nrmse[s_val] = (torch.norm(x1 - x) / torch.norm(x)).item()
    # endpoint estimate near the end of the flow must beat the estimate near start
    assert nrmse[0.9] < nrmse[0.1]


def test_oracle_velocity_gives_perfect_endpoint():
    """Sanity: the OracleVelocity test double recovers Z1 exactly at any s."""
    z0 = torch.randn(3, 6, 1, 16)
    z1 = torch.randn(3, 6, 1, 16)
    oracle = OracleVelocity(z0, z1)
    s = torch.rand(3)
    zs = interpolate_path(z0, z1, s)
    x1 = endpoint_estimate(zs, oracle(zs, s), s)
    assert torch.allclose(x1, z1, atol=1e-5)
