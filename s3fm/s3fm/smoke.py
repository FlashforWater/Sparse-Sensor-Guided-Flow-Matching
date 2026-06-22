"""Deterministic CPU smoke test entry point.

Runs a tiny, dependency-light computation that exercises the M0 plumbing:
seed handling, device selection, config resolution and JSONL logging. It does
NOT train a model or touch KSE/flow/guidance — those arrive in later milestones.

The computation is a small fixed-size random matmul + a one-step SGD update on a
toy quadratic, which is enough to surface any nondeterminism in the RNG path.

Usage:
    python -m s3fm.smoke --seed 0 --out experiments/smoke_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import save_resolved_config
from .logging_utils import JsonlLogger
from .reproducibility import RunContext, seed_everything


def run_smoke(seed: int, out_dir: str | Path, steps: int = 20, device: str = "cpu") -> dict:
    """Run the toy computation and return summary scalars.

    Forces CPU by default for the reproducibility gate (CPU has the most stable
    cross-run numerics). Returns a dict of final scalars; also logs per-step.
    """
    ctx = RunContext.resolve(seed=seed, prefer_device=device)
    # Re-seed after device resolution so the toy draws are seed-determined.
    seed_everything(seed)

    dev = torch.device(ctx.device)
    logger = JsonlLogger(out_dir)

    # Toy problem: minimise 0.5 * ||A x - b||^2 with fixed random A, b.
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(8, 8, generator=g)
    b = torch.randn(8, generator=g)
    x = torch.zeros(8, requires_grad=True)
    lr = 0.05

    final_loss = float("nan")
    for step in range(steps):
        residual = A @ x - b
        loss = 0.5 * (residual @ residual)
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
        final_loss = float(loss.detach())
        logger.log({"loss": final_loss, "x_norm": float(x.detach().norm())}, step=step)

    summary = {
        "seed": seed,
        "device": ctx.device,
        "final_loss": final_loss,
        "x_final_norm": float(x.detach().norm()),
        "torch_version": ctx.torch_version,
    }

    config = {
        "experiment": {"name": "m0_smoke", "seed": seed},
        "runtime": {
            "device": ctx.device,
            "torch_version": ctx.torch_version,
            "numpy_version": ctx.numpy_version,
        },
        "smoke": {"steps": steps, "lr": lr},
        "result": summary,
    }
    save_resolved_config(config, out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="S3FM M0 deterministic smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="experiments/smoke_run")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    summary = run_smoke(seed=args.seed, out_dir=args.out, steps=args.steps, device=args.device)
    print("Smoke summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
