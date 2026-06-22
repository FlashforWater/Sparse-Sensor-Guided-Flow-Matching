"""Deterministic seed handling and device selection.

The single most important M0 guarantee: two runs with the same seed produce
the same numbers within tolerance. Everything that touches randomness must go
through :func:`seed_everything`, and every experiment must record the resolved
device via :func:`select_device`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The integer seed applied to every RNG.
    deterministic:
        When ``True``, request deterministic cuDNN/algorithm behaviour. This can
        be slower but is required for the M0 reproducibility gate. We do not call
        ``torch.use_deterministic_algorithms(True)`` here because some ops lack
        deterministic kernels on MPS/CPU; we instead fix the seeds and disable
        cuDNN autotuning, which is sufficient for the smoke test tolerance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device.

    ``prefer`` may be ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``. ``"auto"``
    picks CUDA, then MPS, then CPU. The chosen device is returned so callers can
    log it into the resolved config.
    """
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if prefer != "auto":
        raise ValueError(f"unknown device preference: {prefer!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class RunContext:
    """Resolved runtime facts recorded into every experiment's config."""

    seed: int
    device: str
    torch_version: str
    numpy_version: str

    @classmethod
    def resolve(cls, seed: int, prefer_device: str = "auto") -> "RunContext":
        seed_everything(seed)
        device = select_device(prefer_device)
        return cls(
            seed=seed,
            device=str(device),
            # torch.__version__ is a TorchVersion object, not a plain str; cast
            # so it is YAML-serialisable in the resolved config.
            torch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
        )
