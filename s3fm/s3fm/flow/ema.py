"""Exponential moving average of model parameters.

EMA weights typically give better unconditional samples than the raw training
weights in flow/diffusion models, so we track them and save both. Minimal
implementation: a shadow copy updated after each optimizer step.
"""

from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(m_p.detach(), alpha=1.0 - self.decay)
        # keep buffers (e.g. GroupNorm has none, but be safe) in sync
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    def to(self, device):
        self.shadow.to(device)
        return self
