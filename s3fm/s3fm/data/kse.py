"""Kuramoto-Sivashinsky (KSE) trajectory generation via ETDRK4.

The KSE in 1D on a periodic domain of length L:

    u_t = -u u_x - u_xx - u_xxxx

We integrate in Fourier space with the Exponential Time Differencing 4th-order
Runge-Kutta scheme (ETDRK4) of Kassam & Trefethen (2005), the standard accurate
solver for this stiff PDE. The linear operator is

    L_hat(k) = k^2 - k^4        (note: -u_xx -> +k^2, -u_xxxx -> -k^4)

and the nonlinear term -u u_x is computed pseudo-spectrally as -0.5 (u^2)_x.

Determinism: trajectories depend only on (L, Nx, dt, n_steps, seed for the
initial condition). The same arguments reproduce the same trajectory.

Parameters are configurable so we can later match an external reference (e.g.
the S3GM Zenodo KSE or D-Flow SGLD's L=20pi, 256 grid) without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class KSEConfig:
    """KSE generation parameters. All physical/numerical knobs live here."""

    L: float = 22.0          # domain length (22 is the classic chaotic KSE box)
    Nx: int = 64             # number of spatial grid points
    dt: float = 0.25         # integration time step
    n_steps: int = 1500      # number of recorded frames per trajectory
    warmup_steps: int = 200  # steps discarded so the trajectory is on the attractor
    n_trajectories: int = 8  # how many independent trajectories to generate
    ic_scale: float = 0.6    # amplitude of the random initial condition

    def as_dict(self) -> dict:
        return asdict(self)


def _etdrk4_coeffs(L_hat: np.ndarray, dt: float):
    """Precompute ETDRK4 coefficients via contour integration (Kassam-Trefethen).

    The scalar quantities E, E2, Q, f1, f2, f3 are evaluated using a complex
    contour-mean to avoid cancellation error when L_hat*dt is near zero.
    """
    E = np.exp(dt * L_hat)
    E2 = np.exp(dt * L_hat / 2.0)

    M = 32  # number of points on the unit circle for the contour mean
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # roots of unity
    LR = dt * L_hat[:, None] + r[None, :]  # (Nx_half, M)

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
    return E, E2, Q, f1, f2, f3


def generate_trajectory(cfg: KSEConfig, seed: int) -> np.ndarray:
    """Integrate one KSE trajectory.

    Returns an array of shape ``(n_steps, Nx)`` in physical space (real-valued).
    ``seed`` controls only the random initial condition; the dynamics are
    deterministic thereafter.
    """
    Nx, L, dt = cfg.Nx, cfg.L, cfg.dt
    x = L * np.arange(Nx) / Nx

    rng = np.random.default_rng(seed)
    # Smooth random initial condition (low-amplitude), then let it develop.
    u = cfg.ic_scale * (rng.standard_normal(Nx))
    # smooth it a touch so high-k noise doesn't blow up the very first steps
    u = u - u.mean()

    # Wavenumbers for a length-L periodic domain.
    k = (2.0 * np.pi / L) * np.fft.fftfreq(Nx, d=1.0 / Nx)
    L_hat = k**2 - k**4  # linear operator in Fourier space
    g = -0.5j * k        # multiplier for the nonlinear term -0.5 (u^2)_x

    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(L_hat, dt)

    v = np.fft.fft(u)

    def nonlinear(v_hat):
        return g * np.fft.fft(np.real(np.fft.ifft(v_hat)) ** 2)

    frames = np.empty((cfg.n_steps, Nx), dtype=np.float64)
    total = cfg.warmup_steps + cfg.n_steps
    rec = 0
    for step in range(total):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if step >= cfg.warmup_steps:
            frames[rec] = np.real(np.fft.ifft(v))
            rec += 1

    return frames


def generate_dataset(cfg: KSEConfig, base_seed: int = 0) -> np.ndarray:
    """Generate ``cfg.n_trajectories`` trajectories.

    Returns shape ``(n_trajectories, n_steps, Nx)``. Trajectory ``i`` uses
    ``seed = base_seed + i`` for its initial condition, so the whole dataset is
    reproducible from ``(cfg, base_seed)``.
    """
    trajs = np.stack(
        [generate_trajectory(cfg, seed=base_seed + i) for i in range(cfg.n_trajectories)],
        axis=0,
    )
    return trajs
