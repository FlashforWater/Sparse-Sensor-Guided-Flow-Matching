"""M2 analytical tests: linear path, velocity target, endpoint oracle identity.

These are the spec's foundational correctness checks. They use *oracle* velocity
(the analytic Z1 - Z0), no neural network, so any failure is a math/code bug, not
a training issue. The endpoint oracle test is correctness check #1 from the spec.
"""

from __future__ import annotations

import torch

from s3fm.flow.paths import (
    endpoint_estimate,
    flow_matching_loss,
    interpolate_path,
    velocity_target,
)


def test_path_endpoints():
    """Zs = Z0 at s=0 and Zs = Z1 at s=1."""
    z0 = torch.randn(4, 5, 1, 16)
    z1 = torch.randn(4, 5, 1, 16)
    s0 = torch.zeros(4)
    s1 = torch.ones(4)
    assert torch.allclose(interpolate_path(z0, z1, s0), z0)
    assert torch.allclose(interpolate_path(z0, z1, s1), z1)


def test_path_midpoint():
    z0 = torch.randn(3, 4, 1, 8)
    z1 = torch.randn(3, 4, 1, 8)
    s = torch.full((3,), 0.5)
    expected = 0.5 * z0 + 0.5 * z1
    assert torch.allclose(interpolate_path(z0, z1, s), expected)


def test_path_matches_analytic_for_random_s():
    z0 = torch.randn(6, 3, 1, 8)
    z1 = torch.randn(6, 3, 1, 8)
    s = torch.rand(6)
    sb = s.view(6, 1, 1, 1)
    assert torch.allclose(interpolate_path(z0, z1, s), (1 - sb) * z0 + sb * z1)


def test_velocity_target_is_difference():
    z0 = torch.randn(2, 4, 1, 8)
    z1 = torch.randn(2, 4, 1, 8)
    assert torch.equal(velocity_target(z0, z1), z1 - z0)


def test_endpoint_oracle_identity():
    """Correctness check #1: with oracle velocity v = Z1 - Z0,
    X1_hat = Zs + (1-s) v must equal Z1 for ANY s."""
    z0 = torch.randn(8, 5, 1, 16)
    z1 = torch.randn(8, 5, 1, 16)
    for s_val in [0.0, 0.1, 0.37, 0.5, 0.9, 1.0]:
        s = torch.full((8,), s_val)
        zs = interpolate_path(z0, z1, s)
        v = velocity_target(z0, z1)  # oracle
        x1_hat = endpoint_estimate(zs, v, s)
        assert torch.allclose(x1_hat, z1, atol=1e-5), f"failed at s={s_val}"


def test_endpoint_oracle_per_sample_s():
    z0 = torch.randn(8, 5, 1, 16)
    z1 = torch.randn(8, 5, 1, 16)
    s = torch.rand(8)
    zs = interpolate_path(z0, z1, s)
    v = velocity_target(z0, z1)
    x1_hat = endpoint_estimate(zs, v, s)
    assert torch.allclose(x1_hat, z1, atol=1e-5)


def test_fm_loss_zero_at_oracle():
    z0 = torch.randn(4, 5, 1, 16)
    z1 = torch.randn(4, 5, 1, 16)
    oracle = velocity_target(z0, z1)
    assert flow_matching_loss(oracle, z0, z1).item() < 1e-12


def test_fm_loss_positive_otherwise():
    z0 = torch.randn(4, 5, 1, 16)
    z1 = torch.randn(4, 5, 1, 16)
    wrong = torch.zeros_like(z0)
    assert flow_matching_loss(wrong, z0, z1).item() > 0
