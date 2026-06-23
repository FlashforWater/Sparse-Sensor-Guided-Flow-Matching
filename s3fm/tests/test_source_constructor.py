"""Tests for the fixed observation-informed source constructor W_B(y,H)."""

from __future__ import annotations

import torch

from s3fm.flow.sources import SourceConfig, spectral_truncation
from s3fm.guidance.source_constructor import (
    LowBandwidthSourceConfig,
    low_bandwidth_mean_from_observation,
    observation_informed_source,
)
from s3fm.measurements.base import MaskOperator


def _low_freq_field(b=2, T=5, Nx=32, cutoff_k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, T, 1, Nx, generator=g)
    return spectral_truncation(x, cutoff_k=cutoff_k)


def test_low_bandwidth_mean_recovers_full_observed_low_freq_field():
    """With full observations, W_B should recover a low-frequency field."""
    x = _low_freq_field(cutoff_k=4)
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=1.0, seed=0)
    y = H(x)
    source_cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.0)
    ctor_cfg = LowBandwidthSourceConfig(ridge=1e-8, temporal_ridge=0.0)

    warm = low_bandwidth_mean_from_observation(y, H, source_cfg, ctor_cfg)
    assert torch.allclose(warm, x, atol=1e-4)


def test_observation_informed_source_deterministic_with_seed():
    """Same y/H/source config/seed -> same source sample."""
    x = _low_freq_field(cutoff_k=4, seed=1)
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=0.4, seed=2)
    y = H(x)
    source_cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)

    a = observation_informed_source(y, H, source_cfg, seed=3)
    b = observation_informed_source(y, H, source_cfg, seed=3)
    assert torch.equal(a, b)


def test_source_noise_changes_but_low_bandwidth_mean_does_not():
    """The fixed W_B(y,H) mean is deterministic; eta supplies source randomness."""
    x = _low_freq_field(cutoff_k=4, seed=4)
    H = MaskOperator.random(T=5, C=1, Nx=32, observed_fraction=0.5, seed=5)
    y = H(x)
    source_cfg = SourceConfig(kind="spectral", cutoff_k=4, eta_std=0.05)

    mean_a = low_bandwidth_mean_from_observation(y, H, source_cfg)
    mean_b = low_bandwidth_mean_from_observation(y, H, source_cfg)
    assert torch.equal(mean_a, mean_b)

    src_a = observation_informed_source(y, H, source_cfg, seed=6)
    src_b = observation_informed_source(y, H, source_cfg, seed=7)
    assert not torch.equal(src_a, src_b)
