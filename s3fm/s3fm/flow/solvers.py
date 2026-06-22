"""Fixed-step ODE solvers for the flow, with exact NFE accounting.

The guided/unguided flow integrates dZ/ds = v(Z, s) from s=0 to s=1. We use
fixed-step solvers (Euler first, then RK4) because they make the NFE count, the
gradients, and the sign conventions transparent — the spec forbids starting with
an adaptive black-box solver that hides NFE.

NFE accounting (spec, non-negotiable): NFE counts EVERY call to the velocity
field. Therefore for N steps:
    Euler    -> N   NFE
    Midpoint -> 2N  NFE
    RK4      -> 4N  NFE
The solver returns both the trajectory endpoint and the realized NFE so callers
never confuse "solver steps" with NFE.

The velocity field is any callable ``v(z, s) -> tensor`` where ``s`` is a
per-batch flow time of shape ``[B]``. This is exactly the model interface and
also the guided-velocity closure used in M4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

VelocityField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SolveResult:
    z1: torch.Tensor      # final state at s=1
    nfe: int              # number of velocity evaluations actually performed
    steps: int            # number of solver steps
    solver: str


def _full_s(z: torch.Tensor, s_val: float) -> torch.Tensor:
    return torch.full((z.shape[0],), s_val, device=z.device, dtype=z.dtype)


def euler_solve(v: VelocityField, z0: torch.Tensor, steps: int) -> SolveResult:
    """Forward Euler on s in [0, 1]. NFE = steps."""
    z = z0
    ds = 1.0 / steps
    nfe = 0
    for i in range(steps):
        s = _full_s(z, i * ds)
        k = v(z, s); nfe += 1
        z = z + ds * k
    return SolveResult(z1=z, nfe=nfe, steps=steps, solver="euler")


def midpoint_solve(v: VelocityField, z0: torch.Tensor, steps: int) -> SolveResult:
    """Fixed-step midpoint (RK2). NFE = 2 * steps."""
    z = z0
    ds = 1.0 / steps
    nfe = 0
    for i in range(steps):
        s0 = _full_s(z, i * ds)
        k1 = v(z, s0); nfe += 1
        s_mid = _full_s(z, (i + 0.5) * ds)
        k2 = v(z + 0.5 * ds * k1, s_mid); nfe += 1
        z = z + ds * k2
    return SolveResult(z1=z, nfe=nfe, steps=steps, solver="midpoint")


def rk4_solve(v: VelocityField, z0: torch.Tensor, steps: int) -> SolveResult:
    """Classic RK4. NFE = 4 * steps."""
    z = z0
    ds = 1.0 / steps
    nfe = 0
    for i in range(steps):
        s0 = _full_s(z, i * ds)
        s_mid = _full_s(z, (i + 0.5) * ds)
        s_end = _full_s(z, (i + 1.0) * ds)
        k1 = v(z, s0); nfe += 1
        k2 = v(z + 0.5 * ds * k1, s_mid); nfe += 1
        k3 = v(z + 0.5 * ds * k2, s_mid); nfe += 1
        k4 = v(z + ds * k3, s_end); nfe += 1
        z = z + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return SolveResult(z1=z, nfe=nfe, steps=steps, solver="rk4")


SOLVERS = {"euler": euler_solve, "midpoint": midpoint_solve, "rk4": rk4_solve}


def solve(v: VelocityField, z0: torch.Tensor, steps: int, solver: str = "euler") -> SolveResult:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {list(SOLVERS)}")
    return SOLVERS[solver](v, z0, steps)
