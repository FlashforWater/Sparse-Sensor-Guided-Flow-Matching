"""Distribution-level statistics for evaluating unconditional sample fidelity.

For a chaotic system like KSE, pointwise match is meaningless; what matters is
whether generated samples reproduce the *statistics* of the dynamics. We use:

- **Spatial two-point correlation** C(r) = < u(x) u(x+r) >, averaged over space,
  frames and samples. For a periodic domain this is computed via the FFT power
  spectrum (Wiener-Khinchin), normalized to C(0)=1.
- **Marginal value distribution**: histogram of all field values.

These are compared between generated samples and held-out test data. The
two-point correlation error is the primary M3 fidelity metric.
"""

from __future__ import annotations

import numpy as np


def two_point_correlation(fields: np.ndarray) -> np.ndarray:
    """Normalized spatial two-point correlation C(r), averaged over all frames.

    ``fields`` has shape ``[..., Nx]`` (any leading dims are flattened and
    averaged over). Returns ``C`` of length ``Nx`` with ``C[0] = 1`` (lag 0).

    Uses the Wiener-Khinchin theorem: the autocorrelation is the inverse FFT of
    the power spectrum. Each spatial line is de-meaned first so C measures
    fluctuation structure, not the (near-zero) mean.
    """
    x = fields.reshape(-1, fields.shape[-1]).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    fk = np.fft.rfft(x, axis=1)
    power = (fk * np.conj(fk)).real
    corr = np.fft.irfft(power, n=x.shape[1], axis=1)
    corr = corr.mean(axis=0)
    if corr[0] <= 0:
        return corr
    return corr / corr[0]


def two_point_correlation_error(gen: np.ndarray, ref: np.ndarray) -> float:
    """L2 distance between generated and reference two-point correlations.

    Compared over the first half of lags (the rest is the periodic mirror image).
    """
    cg = two_point_correlation(gen)
    cr = two_point_correlation(ref)
    half = len(cg) // 2
    return float(np.sqrt(np.mean((cg[:half] - cr[:half]) ** 2)))


def marginal_stats(fields: np.ndarray) -> dict:
    """Summary of the marginal value distribution."""
    x = fields.reshape(-1).astype(np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "skew": float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)),
        "kurtosis": float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12)),
    }


def marginal_distance(gen: np.ndarray, ref: np.ndarray, bins: int = 60) -> float:
    """Histogram L1 distance (total variation-like) of marginal value distros."""
    lo = min(gen.min(), ref.min())
    hi = max(gen.max(), ref.max())
    hg, _ = np.histogram(gen.reshape(-1), bins=bins, range=(lo, hi), density=True)
    hr, edges = np.histogram(ref.reshape(-1), bins=bins, range=(lo, hi), density=True)
    width = edges[1] - edges[0]
    return float(0.5 * np.sum(np.abs(hg - hr)) * width)
