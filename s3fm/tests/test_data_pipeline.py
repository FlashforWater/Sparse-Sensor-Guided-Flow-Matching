"""M1 gate: KSE data pipeline correctness.

Covers: deterministic generation, bounded/chaotic dynamics, no split leakage,
normalization round-trip < 1e-6, window dimensions and frame indexing, and
measurement-operator alignment (y = H(X_true)).
"""

from __future__ import annotations

import numpy as np
import torch

from s3fm.data.kse import KSEConfig, generate_trajectory, generate_dataset
from s3fm.data.splits import ChannelStandardizer, SplitIndices, make_splits
from s3fm.data.windows import add_channel_axis, extract_windows
from s3fm.measurements.base import IdentityOperator, MaskOperator


# ---------------------------------------------------------------- generation

def _small_cfg(**kw):
    base = dict(L=22.0, Nx=64, dt=0.25, n_steps=120, warmup_steps=80, n_trajectories=6)
    base.update(kw)
    return KSEConfig(**base)


def test_kse_deterministic_and_bounded():
    cfg = _small_cfg()
    u1 = generate_trajectory(cfg, seed=0)
    u2 = generate_trajectory(cfg, seed=0)
    assert np.array_equal(u1, u2)
    assert np.isfinite(u1).all()
    # KSE stays bounded (not blowing up); textbook range is a few units.
    assert np.abs(u1).max() < 10.0
    # chaotic, not constant in time
    assert u1[:, cfg.Nx // 2].std() > 0.1


def test_kse_different_seed_differs():
    cfg = _small_cfg()
    assert not np.array_equal(generate_trajectory(cfg, seed=0), generate_trajectory(cfg, seed=1))


def test_dataset_shape_and_reproducible():
    cfg = _small_cfg()
    ds = generate_dataset(cfg, base_seed=0)
    assert ds.shape == (cfg.n_trajectories, cfg.n_steps, cfg.Nx)
    assert np.array_equal(ds[0], generate_trajectory(cfg, seed=0))


# --------------------------------------------------------------------- splits

def test_splits_no_leakage():
    splits = make_splits(n_trajectories=10, seed=0)
    train, val, test = set(splits.train), set(splits.val), set(splits.test)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(10))


def test_splits_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        SplitIndices(train=(0, 1), val=(1,), test=(2,))


def test_splits_deterministic():
    assert make_splits(10, seed=0) == make_splits(10, seed=0)


# ------------------------------------------------------------- normalization

def test_normalization_roundtrip_below_1e6():
    cfg = _small_cfg()
    ds = add_channel_axis(generate_dataset(cfg, base_seed=0))  # (n,steps,1,Nx)
    splits = make_splits(cfg.n_trajectories, seed=0)
    train = ds[list(splits.train)]
    std = ChannelStandardizer.fit(train, channel_axis=2)
    # round-trip on the *test* split (held out from fit)
    test = ds[list(splits.test)]
    z = std.transform(test, channel_axis=2)
    back = std.inverse(z, channel_axis=2)
    assert np.max(np.abs(back - test)) < 1e-6


def test_normalization_standardizes_train():
    cfg = _small_cfg()
    ds = add_channel_axis(generate_dataset(cfg, base_seed=0))
    splits = make_splits(cfg.n_trajectories, seed=0)
    train = ds[list(splits.train)]
    std = ChannelStandardizer.fit(train, channel_axis=2)
    z = std.transform(train, channel_axis=2)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-3


# ------------------------------------------------------------------- windows

def test_window_dims_and_frame_indexing():
    cfg = _small_cfg()
    ds = add_channel_axis(generate_dataset(cfg, base_seed=0))
    splits = make_splits(cfg.n_trajectories, seed=0)
    train_rows = list(splits.train)
    wb = extract_windows(
        ds[train_rows], traj_ids=tuple(train_rows),
        window_length=20, stride=10, physical_dt=cfg.dt,
    )
    b, T, C, Nx = wb.shape
    assert (T, C, Nx) == (20, 1, cfg.Nx)
    assert wb.frame_idx.shape == (b, T)
    # frames within a window are consecutive global indices
    assert np.array_equal(wb.frame_idx[0], np.arange(wb.frame_idx[0, 0], wb.frame_idx[0, 0] + T))
    # every window's traj_idx is one of the train trajectories (no leakage into windows)
    assert set(wb.traj_idx.tolist()).issubset(set(train_rows))


def test_window_no_frame_skipped_with_unit_stride():
    cfg = _small_cfg(n_steps=50)
    ds = add_channel_axis(generate_dataset(cfg, base_seed=0))
    wb = extract_windows(ds[:1], traj_ids=(0,), window_length=10, stride=1, physical_dt=cfg.dt)
    # with stride 1 the first frames of consecutive windows increase by exactly 1
    firsts = wb.frame_idx[:, 0]
    assert np.array_equal(np.diff(firsts), np.ones(len(firsts) - 1))


# -------------------------------------------------------------- measurements

def test_identity_operator():
    x = torch.randn(2, 5, 1, 16)
    H = IdentityOperator()
    assert torch.equal(H(x), x)
    H.validate(x, H(x))


def test_mask_operator_alignment():
    """y = H(X_true): observed entries equal the field, unobserved are zero."""
    x = torch.randn(3, 5, 1, 16)
    H = MaskOperator.random(T=5, C=1, Nx=16, observed_fraction=0.4, seed=0)
    y = H(x)
    mask = H.mask.unsqueeze(0).expand(3, -1, -1, -1)
    # observed entries match exactly
    assert torch.equal(y[mask], x[mask])
    # unobserved entries are exactly zero
    assert torch.all(y[~mask] == 0)


def test_mask_operator_metadata_and_count():
    H = MaskOperator.random(T=5, C=1, Nx=16, observed_fraction=0.25, seed=1)
    md = H.metadata()
    assert md["type"] == "mask"
    assert md["num_observed"] == int(H.mask.sum())
    assert 0 < md["observed_fraction"] <= 1.0


def test_mask_operator_differentiable():
    x = torch.randn(2, 4, 1, 8, requires_grad=True)
    H = MaskOperator.random(T=4, C=1, Nx=8, observed_fraction=0.5, seed=0)
    loss = (H(x) ** 2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mask_adjoint_is_self():
    x = torch.randn(1, 4, 1, 8)
    H = MaskOperator.random(T=4, C=1, Nx=8, observed_fraction=0.5, seed=0)
    assert torch.equal(H.adjoint(x), H.forward(x))
