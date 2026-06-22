"""M4b tests: informative source S(X), consistency, endpoint, transport shortening."""

from __future__ import annotations

import numpy as np
import torch

from s3fm.flow.paths import endpoint_estimate, interpolate_path
from s3fm.flow.sources import (
    MarginalSourceSampler,
    SourceConfig,
    apply_source,
    coarse_blur,
    make_source,
    spectral_truncation,
)
from s3fm.models.video_unet_velocity import OracleVelocity


def _kse_like(b=8, T=10, Nx=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, 1, Nx, generator=g)


# --------------------------------------------------------- source operators

def test_spectral_truncation_removes_high_freq():
    x = _kse_like()
    s = spectral_truncation(x, cutoff_k=8)
    # truncated field has strictly less energy than the original
    assert s.pow(2).sum() < x.pow(2).sum()
    # and the high modes are gone
    sf = torch.fft.rfft(s, dim=-1)
    assert torch.allclose(sf[..., 8:], torch.zeros_like(sf[..., 8:]), atol=1e-5)


def test_coarse_blur_shape_preserved():
    x = _kse_like(Nx=64)
    s = coarse_blur(x, factor=8)
    assert s.shape == x.shape
    assert s.pow(2).sum() < x.pow(2).sum()


def test_apply_source_no_eta_deterministic():
    x = _kse_like()
    cfg = SourceConfig(kind="spectral", cutoff_k=8, eta_std=0.0)
    a = apply_source(x, cfg)
    b = apply_source(x, cfg)
    assert torch.equal(a, b)


def test_make_source_eta_deterministic_with_seed():
    """Source determinism: same eta seed -> same Z0."""
    x = _kse_like()
    cfg = SourceConfig(kind="spectral", cutoff_k=8, eta_std=0.05)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    assert torch.equal(make_source(x, cfg, generator=g1), make_source(x, cfg, generator=g2))


def test_source_does_not_depend_on_H():
    """S(X) is a fixed corruption; it takes only X and cfg, never any operator."""
    import inspect

    sig = inspect.signature(apply_source)
    assert "operator" not in sig.parameters and "H" not in sig.parameters


# --------------------------------------------- train/inference consistency

def test_marginal_source_matches_training_distribution():
    """Inference Z0 (marginal sampler) matches training S(X)+eta stats in tol."""
    ref = _kse_like(b=200, seed=1)
    cfg = SourceConfig(kind="spectral", cutoff_k=8, eta_std=0.05)
    # training-source stats
    g = torch.Generator().manual_seed(0)
    train_src = make_source(ref, cfg, generator=g)
    sampler = MarginalSourceSampler(ref, cfg, seed=0)
    drawn = sampler.sample(200, seed=5)
    assert abs(train_src.mean().item() - drawn.mean().item()) < 0.05
    assert abs(train_src.std().item() - drawn.std().item()) < 0.1


# --------------------------------------------- endpoint with informative src

def test_endpoint_oracle_with_informative_source():
    """Endpoint formula is source-agnostic: oracle velocity recovers Z1 even when
    Z0 = S(X)+eta (informative, dependent coupling)."""
    z1 = _kse_like(seed=2)
    cfg = SourceConfig(kind="spectral", cutoff_k=8, eta_std=0.05)
    g = torch.Generator().manual_seed(0)
    z0 = make_source(z1, cfg, generator=g)
    oracle = OracleVelocity(z0, z1)
    s = torch.rand(z1.shape[0])
    zs = interpolate_path(z0, z1, s)
    x1 = endpoint_estimate(zs, oracle(zs, s), s)
    assert torch.allclose(x1, z1, atol=1e-5)


# ------------------------------------------------ transport shortening

def test_informative_source_shortens_transport():
    """Mechanism: mean ||Z1-Z0|| with informative source < with Gaussian."""
    z1 = _kse_like(b=64, seed=4)
    g = torch.Generator().manual_seed(0)
    z0_gauss = torch.randn(z1.shape, generator=g)
    disp_gauss = (z1 - z0_gauss).flatten(1).norm(dim=1).mean()

    cfg = SourceConfig(kind="spectral", cutoff_k=8, eta_std=0.05)
    gi = torch.Generator().manual_seed(0)
    z0_info = make_source(z1, cfg, generator=gi)
    disp_info = (z1 - z0_info).flatten(1).norm(dim=1).mean()

    assert disp_info < disp_gauss


def test_transport_shortens_with_higher_cutoff():
    """Higher cutoff_k (richer source) -> shorter transport (monotone trend)."""
    z1 = _kse_like(b=64, seed=6)
    disps = []
    for k in [2, 4, 8]:
        cfg = SourceConfig(kind="spectral", cutoff_k=k, eta_std=0.0)
        d = (z1 - apply_source(z1, cfg)).flatten(1).norm(dim=1).mean().item()
        disps.append(d)
    # displacement decreases as we keep more modes
    assert disps[0] > disps[1] > disps[2]
