"""Guidance-strength schedules lambda_s.

Start with a constant positive lambda for debugging, then compare schedules.
lambda_s >= 0 always; the descent direction is handled by the explicit minus sign
in the guidance term, never by the schedule.
"""

from __future__ import annotations

from typing import Callable

Schedule = Callable[[float], float]


def constant(lambda0: float) -> Schedule:
    """lambda_s = lambda0 for all s."""
    def f(s: float) -> float:
        return lambda0
    return f


def linear_decay(lambda0: float, final_frac: float = 0.0) -> Schedule:
    """Decay linearly from lambda0 at s=0 to lambda0*final_frac at s=1."""
    def f(s: float) -> float:
        return lambda0 * ((1.0 - s) + final_frac * s)
    return f


def constant_then_decay(lambda0: float, hold_until: float = 0.5) -> Schedule:
    """Hold lambda0 until s=hold_until, then decay linearly to 0 at s=1."""
    def f(s: float) -> float:
        if s <= hold_until:
            return lambda0
        return lambda0 * (1.0 - s) / (1.0 - hold_until)
    return f
