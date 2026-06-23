"""Observation-informed low-bandwidth source construction.

This module is the code counterpart of the paper-level ``W_B(y, H)`` object:
a fixed, low-capacity source constructor that uses measurements only through the
operator residual and returns a source matching the S3FM-Info training source.

The important boundary is intentional: this is not a learned inverse solver and
does not access ``x_true``. It solves a small low-bandwidth problem so the flow
prior still has to supply missing fine-scale dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..flow.sources import SourceConfig, make_source
from ..measurements.base import MeasurementOperator


@dataclass(frozen=True)
class LowBandwidthSourceConfig:
    """Fixed hyperparameters for ``W_B(y, H)``.

    ``basis`` is domain-level, not measurement-operator-level. For KSE we use a
    periodic Fourier basis. ``ridge`` and ``temporal_ridge`` are fixed weak
    regularizers used for every mask/sensor layout.
    """

    basis: str = "fourier"
    ridge: float = 1e-3
    temporal_ridge: float = 0.5


def _periodic_interp_1d(mask_1d: np.ndarray, values_1d: np.ndarray) -> np.ndarray:
    """Fill one periodic spatial frame from sparse samples using linear interp."""
    Nx = mask_1d.shape[0]
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return np.zeros(Nx, dtype=np.float32)
    vals = values_1d[idx].astype(np.float32)
    if idx.size == 1:
        return np.full(Nx, vals[0], dtype=np.float32)
    xq = np.arange(Nx)
    xp = np.concatenate(([idx[-1] - Nx], idx, [idx[0] + Nx]))
    fp = np.concatenate(([vals[-1]], vals, [vals[0]]))
    return np.interp(xq, xp, fp).astype(np.float32)


def _fourier_design(x: np.ndarray, Nx: int, keep: int) -> np.ndarray:
    cols = [np.ones_like(x, dtype=np.float64)]
    for k in range(1, keep):
        phase = 2.0 * np.pi * k * x / Nx
        cols.append(np.cos(phase))
        cols.append(np.sin(phase))
    return np.stack(cols, axis=1)


def _spectral_fit_sequence(
    mask: np.ndarray,
    values: np.ndarray,
    cutoff_k: int,
    ridge: float,
    temporal_ridge: float,
) -> np.ndarray:
    """Fit low spatial Fourier modes jointly across time with smooth coeffs."""
    T, C, Nx = values.shape
    keep = min(cutoff_k, Nx // 2 + 1)
    P = 1 + 2 * max(0, keep - 1)
    full_design = _fourier_design(np.arange(Nx, dtype=np.float64), Nx, keep)
    out = np.zeros((T, C, Nx), dtype=np.float32)

    for c in range(C):
        dim = T * P
        lhs = ridge * np.eye(dim, dtype=np.float64)
        rhs = np.zeros(dim, dtype=np.float64)

        for t in range(T):
            idx = np.flatnonzero(mask[t, c])
            if idx.size == 0:
                continue
            A = _fourier_design(idx.astype(np.float64), Nx, keep)
            y = values[t, c, idx].astype(np.float64)
            sl = slice(t * P, (t + 1) * P)
            lhs[sl, sl] += A.T @ A
            rhs[sl] += A.T @ y

        if temporal_ridge > 0:
            I = np.eye(P, dtype=np.float64)
            for t in range(T - 1):
                a = slice(t * P, (t + 1) * P)
                b = slice((t + 1) * P, (t + 2) * P)
                lhs[a, a] += temporal_ridge * I
                lhs[b, b] += temporal_ridge * I
                lhs[a, b] -= temporal_ridge * I
                lhs[b, a] -= temporal_ridge * I

        coef = np.linalg.solve(lhs, rhs).reshape(T, P)
        for t in range(T):
            out[t, c] = (full_design @ coef[t]).astype(np.float32)
    return out


def _fill_missing_time_frames(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill frames with no observations by linear interpolation in physical time."""
    T, C, Nx = field.shape
    frames = np.arange(T)
    for c in range(C):
        valid_c = valid[:, c]
        if valid_c.all():
            continue
        if not valid_c.any():
            field[:, c, :] = 0.0
            continue
        known = frames[valid_c]
        for x in range(Nx):
            field[:, c, x] = np.interp(frames, known, field[valid_c, c, x])
    return field


def low_bandwidth_mean_from_observation(
    observation: torch.Tensor,
    operator: MeasurementOperator,
    source_cfg: SourceConfig,
    constructor_cfg: LowBandwidthSourceConfig = LowBandwidthSourceConfig(),
) -> torch.Tensor:
    """Return the deterministic low-bandwidth mean ``W_B(y,H)``.

    Current implementation covers the KSE/M4 sparse-mask case. The extension
    path for other ``H`` is to keep this objective fixed and replace only the
    measurement operator inside the residual.
    """
    if not hasattr(operator, "mask"):
        raise ValueError("low-bandwidth source currently requires MaskOperator-like `mask`")
    if constructor_cfg.basis != "fourier":
        raise ValueError(f"unknown low-bandwidth basis {constructor_cfg.basis!r}")

    y = observation.detach().cpu()
    mask = operator.mask.detach().cpu().numpy().astype(bool)
    B, T, C, Nx = y.shape
    if mask.shape != (T, C, Nx):
        raise ValueError(f"mask shape {mask.shape} does not match observation {(T, C, Nx)}")

    warm = np.zeros((B, T, C, Nx), dtype=np.float32)
    valid = mask.any(axis=-1)
    yn = y.numpy()
    for b in range(B):
        if source_cfg.kind == "spectral":
            warm[b] = _spectral_fit_sequence(
                mask,
                yn[b],
                source_cfg.cutoff_k,
                ridge=constructor_cfg.ridge,
                temporal_ridge=constructor_cfg.temporal_ridge,
            )
        else:
            for t in range(T):
                for c in range(C):
                    warm[b, t, c] = _periodic_interp_1d(mask[t, c], yn[b, t, c])
            warm[b] = _fill_missing_time_frames(warm[b], valid)
    return torch.tensor(warm, dtype=observation.dtype)


def observation_informed_source(
    observation: torch.Tensor,
    operator: MeasurementOperator,
    source_cfg: SourceConfig,
    seed: int = 0,
    constructor_cfg: LowBandwidthSourceConfig = LowBandwidthSourceConfig(),
) -> torch.Tensor:
    """Sample ``Z0`` from the fixed low-bandwidth conditional source.

    This returns ``S(W_B(y,H)) + eta`` with the same ``S`` and ``eta`` scale used
    by the S3FM-Info training source.
    """
    warm_mean = low_bandwidth_mean_from_observation(observation, operator, source_cfg, constructor_cfg)
    generator = torch.Generator().manual_seed(seed)
    return make_source(warm_mean, source_cfg, generator=generator)
