"""Trajectory-level splits and reversible normalization.

Two M1 guarantees:

1. **No leakage.** Splits are made at the *trajectory* level, never at the
   window/frame level. A trajectory is wholly in train, val, or test. Window
   extraction (windows.py) only ever runs *within* a split.

2. **Reversible normalization.** We use channel standardization: subtract a
   per-channel mean and divide by a per-channel std, both computed on the
   training split only (test/val must not see their own statistics, to avoid
   leakage). The transform is exactly invertible up to float round-off
   (< 1e-6 round-trip, enforced by tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Immutable trajectory-index assignment for the three splits."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self):
        all_idx = list(self.train) + list(self.val) + list(self.test)
        if len(all_idx) != len(set(all_idx)):
            raise ValueError("split indices overlap — that is data leakage")

    @property
    def all(self) -> tuple[int, ...]:
        return tuple(sorted(self.train + self.val + self.test))


def make_splits(
    n_trajectories: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> SplitIndices:
    """Assign whole trajectories to train/val/test deterministically.

    The remaining fraction after train+val goes to test. Shuffling is seeded so
    the split is fixed and reproducible. Each trajectory index appears in exactly
    one split (enforced by :class:`SplitIndices`).
    """
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("invalid split fractions")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trajectories)
    n_train = int(round(train_frac * n_trajectories))
    n_val = int(round(val_frac * n_trajectories))
    # guarantee at least one trajectory per split when possible
    n_train = max(1, n_train)
    n_val = max(1, n_val) if n_trajectories >= 3 else 0
    n_train = min(n_train, n_trajectories - n_val - 1) if n_trajectories >= 3 else n_train
    train = tuple(int(i) for i in perm[:n_train])
    val = tuple(int(i) for i in perm[n_train : n_train + n_val])
    test = tuple(int(i) for i in perm[n_train + n_val :])
    return SplitIndices(train=train, val=val, test=test)


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel standardization: (x - mean) / std, exactly invertible.

    ``mean`` and ``std`` have shape ``(C,)``. For KSE there is a single channel
    (C=1). Fit on the training split only.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, data: np.ndarray, channel_axis: int) -> "ChannelStandardizer":
        """Fit statistics over every axis except ``channel_axis``.

        ``data`` is the training split, with an explicit channel axis. Returns a
        standardizer whose ``mean``/``std`` are 1-D arrays of length C.
        """
        axes = tuple(a for a in range(data.ndim) if a != channel_axis)
        mean = data.mean(axis=axes)
        std = data.std(axis=axes)
        return cls(mean=mean, std=std)

    def _broadcast(self, x: np.ndarray, channel_axis: int):
        shape = [1] * x.ndim
        shape[channel_axis] = self.mean.shape[0]
        return self.mean.reshape(shape), (self.std.reshape(shape) + self.eps)

    def transform(self, x: np.ndarray, channel_axis: int) -> np.ndarray:
        mean, std = self._broadcast(x, channel_axis)
        return (x - mean) / std

    def inverse(self, x: np.ndarray, channel_axis: int) -> np.ndarray:
        mean, std = self._broadcast(x, channel_axis)
        return x * std + mean
