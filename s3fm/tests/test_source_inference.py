"""Tests for learned observation-to-source inference q_phi."""

from __future__ import annotations

import torch

from s3fm.flow.sources import SourceConfig, apply_source
from s3fm.guidance.source_inference import (
    MaskedSourceInferenceNet,
    learned_observation_informed_source,
    source_inference_loss,
    source_inference_target,
)
from s3fm.measurements.base import MaskOperator


def _batch(b=3, T=5, Nx=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, 1, Nx, generator=g)


def test_source_inference_target_is_source_not_full_x():
    x = _batch()
    cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)
    target = source_inference_target(x, cfg)

    assert torch.equal(target, apply_source(x, cfg))
    assert not torch.allclose(target, x)


def test_masked_source_inference_output_is_low_frequency():
    cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)
    model = MaskedSourceInferenceNet(cfg, base_channels=8, depth=1)
    x = _batch()
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=0.4, seed=1)
    y = H(x)

    out = model(y, H.mask)
    fft = torch.fft.rfft(out, dim=-1)
    assert torch.allclose(fft[..., 4:], torch.zeros_like(fft[..., 4:]), atol=1e-5)


def test_source_inference_loss_runs_and_backpropagates():
    cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)
    model = MaskedSourceInferenceNet(cfg, base_channels=8, depth=1)
    x = _batch(seed=2)
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=0.5, seed=3)
    y = H(x)

    loss = source_inference_loss(model, x, y, H)
    loss.backward()
    grad_norm = sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(loss)
    assert grad_norm > 0


def test_learned_observation_informed_source_seed_controls_eta():
    cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)
    model = MaskedSourceInferenceNet(cfg, base_channels=8, depth=1).eval()
    x = _batch(seed=4)
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=0.5, seed=5)
    y = H(x)

    a = learned_observation_informed_source(model, y, H, seed=6)
    b = learned_observation_informed_source(model, y, H, seed=6)
    c = learned_observation_informed_source(model, y, H, seed=7)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
