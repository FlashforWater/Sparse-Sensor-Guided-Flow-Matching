"""Measurement-operator interface.

A measurement operator maps a full clean spatiotemporal field to a measurement,
``y = H(X)``. Operators must be:

- **differentiable** in PyTorch (guidance backprops ``J_obs`` through ``H``);
- **metadata-carrying** (which frames/coords/channels, units, noise scale);
- **self-validating** (``validate`` checks shapes/alignment).

Tensor convention everywhere: ``[batch, T, C, Nx]`` (KSE). Operators act on a
torch.Tensor and return a torch.Tensor; ``metadata`` returns a plain dict.

A correctness rule from the spec: for every operator there must be a synthetic
test where ``y = H(X_true)`` and the measured entries align exactly with the
intended locations. That test lives in tests/test_measurements.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class MeasurementOperator(ABC):
    """Common interface for all measurement operators."""

    is_linear: bool = False

    @abstractmethod
    def forward(self, full_field: torch.Tensor) -> torch.Tensor:
        """Map a full field ``[batch, T, C, Nx]`` to measurement space."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Describe the measurement: frames, coords, channels, units, noise."""

    def validate(self, full_field: torch.Tensor, observation: torch.Tensor) -> None:
        """Default validation: forward(full_field) must match observation shape.

        Subclasses may extend with coordinate/time/channel checks.
        """
        produced = self.forward(full_field)
        if produced.shape != observation.shape:
            raise ValueError(
                f"{type(self).__name__}: forward shape {tuple(produced.shape)} "
                f"!= observation shape {tuple(observation.shape)}"
            )

    def __call__(self, full_field: torch.Tensor) -> torch.Tensor:
        return self.forward(full_field)


class IdentityOperator(MeasurementOperator):
    """``H(X) = X``. Used for debugging and the guidance-sign sanity test."""

    is_linear = True

    def forward(self, full_field: torch.Tensor) -> torch.Tensor:
        return full_field

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        return residual

    def metadata(self) -> dict[str, Any]:
        return {"type": "identity", "is_linear": True}


class MaskOperator(MeasurementOperator):
    """Sparse space-time-channel point sampling.

    A boolean mask of shape ``[T, C, Nx]`` selects which entries of each window
    are observed. ``forward`` returns the *masked field* (same shape as input,
    unobserved entries zeroed) rather than a ragged vector, which keeps the
    operator shape-stable and differentiable and lets ``J_obs`` be a plain masked
    MSE. The mask is registered as the operator's metadata.

    The same mask is applied to every item in the batch (a fixed sensor layout).
    """

    is_linear = True

    def __init__(self, mask: torch.Tensor, noise_std: float = 0.0):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.ndim != 3:
            raise ValueError(f"mask must be [T, C, Nx], got {tuple(mask.shape)}")
        self.mask = mask
        self.noise_std = float(noise_std)

    @classmethod
    def random(
        cls,
        T: int,
        C: int,
        Nx: int,
        observed_fraction: float,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> "MaskOperator":
        """Build a random sparse mask observing ``observed_fraction`` of entries."""
        if not (0.0 < observed_fraction <= 1.0):
            raise ValueError("observed_fraction must be in (0, 1]")
        g = torch.Generator().manual_seed(seed)
        probs = torch.rand(T, C, Nx, generator=g)
        mask = probs < observed_fraction
        # guarantee at least one observed point
        if not mask.any():
            mask.view(-1)[0] = True
        return cls(mask=mask, noise_std=noise_std)

    def _broadcast_mask(self, full_field: torch.Tensor) -> torch.Tensor:
        b = full_field.shape[0]
        return self.mask.to(full_field.device).unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, full_field: torch.Tensor) -> torch.Tensor:
        if full_field.ndim != 4:
            raise ValueError(f"expected [batch,T,C,Nx], got {tuple(full_field.shape)}")
        mask = self._broadcast_mask(full_field)
        return full_field * mask

    def measure(self, full_field: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Produce a (possibly noisy) observation ``y = H(X) + eps``.

        Noise is added only on observed entries (the same masking is applied to
        the noise), so unobserved positions stay exactly zero.
        """
        y = self.forward(full_field)
        if self.noise_std > 0.0:
            g = None
            if seed is not None:
                g = torch.Generator(device=full_field.device).manual_seed(seed)
            noise = torch.randn(y.shape, generator=g, device=full_field.device) * self.noise_std
            mask = self._broadcast_mask(full_field)
            y = y + noise * mask
        return y

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        # H is a diagonal 0/1 projection, so H^T = H.
        return self.forward(residual)

    def num_observed(self) -> int:
        return int(self.mask.sum().item())

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mask",
            "is_linear": True,
            "mask_shape": tuple(self.mask.shape),
            "num_observed": self.num_observed(),
            "observed_fraction": self.num_observed() / self.mask.numel(),
            "noise_std": self.noise_std,
        }
