"""Train amortized source inference q_phi(U | y,H).

This trains only the observation-to-source model, not the flow prior. The target
is the deterministic source ``U = S(X)``, so the network learns to infer the
low-bandwidth source variable from sparse observations without learning a direct
``y -> X`` reconstruction.

Usage:
    python -m s3fm.train_source_inference \
      --config configs/kse_source_inference.yaml \
      --out experiments/kse_source_inference
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import load_yaml, save_resolved_config
from .flow.sources import apply_source
from .guidance.source_inference import (
    MaskedSourceInferenceNet,
    source_cfg_from_dict,
    source_cfg_to_dict,
    source_inference_loss,
)
from .logging_utils import JsonlLogger
from .measurements.base import MaskOperator
from .reproducibility import seed_everything, select_device
from .train_flow import build_windows


@torch.no_grad()
def val_source_nrmse(model, val_x, observed_fraction: float, mask_seed: int, device) -> float:
    model.eval()
    T, C, Nx = val_x.shape[1:]
    H = MaskOperator.random(T, C, Nx, observed_fraction=observed_fraction, seed=mask_seed)
    x = val_x.to(device)
    y = H(x)
    target = apply_source(x, model.source_cfg)
    pred = model(y, H.mask)
    return float(torch.norm(pred - target) / (torch.norm(target) + 1e-8))


def _observed_fractions(config: dict) -> list[float]:
    scfg = config["source_inference"]
    if "observed_fractions" in scfg:
        values = scfg["observed_fractions"]
    else:
        values = scfg["observed_fraction"]
    if isinstance(values, str):
        values = [float(x.strip()) for x in values.split(",") if x.strip()]
    elif isinstance(values, (int, float)):
        values = [float(values)]
    else:
        values = [float(x) for x in values]
    if not values:
        raise ValueError("source_inference observed_fractions cannot be empty")
    for value in values:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"observed fraction must be in (0,1], got {value}")
    return values


def train(config: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    seed_everything(config["experiment"]["seed"])
    device = select_device(config["train"]["device"])
    logger = JsonlLogger(out_dir)

    train_x, val_x, std, splits, kse = build_windows(config["data"])
    source_cfg = source_cfg_from_dict(config["source"])
    mcfg = config["source_inference"]["model"]
    model = MaskedSourceInferenceNet(
        source_cfg=source_cfg,
        in_channels=1,
        base_channels=mcfg["base_channels"],
        depth=mcfg["depth"],
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"device={device}  train windows={tuple(train_x.shape)}  val windows={tuple(val_x.shape)}")
    print(f"source inference params: {nparam/1e6:.3f}M")

    tcfg = config["train"]
    observed_fractions = _observed_fractions(config)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    train_x = train_x.to(device)
    n_train = train_x.shape[0]
    gen = torch.Generator().manual_seed(config["experiment"]["seed"])

    best_val = float("inf")
    t0 = time.time()
    for step in range(tcfg["steps"]):
        model.train()
        idx = torch.randint(0, n_train, (tcfg["batch_size"],), generator=gen)
        x = train_x[idx]
        mask_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
        frac_idx = int(torch.randint(0, len(observed_fractions), (1,), generator=gen).item())
        observed_fraction = observed_fractions[frac_idx]
        H = MaskOperator.random(x.shape[1], x.shape[2], x.shape[3], observed_fraction=observed_fraction, seed=mask_seed)
        y = H(x)
        loss = source_inference_loss(model, x, y, H)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % tcfg["log_every"] == 0 or step == tcfg["steps"] - 1:
            logger.log({"train_loss": float(loss.detach().cpu())}, step=step)

        if step % tcfg["val_every"] == 0 or step == tcfg["steps"] - 1:
            val_subset = val_x[: min(config["source_inference"]["val_windows"], val_x.shape[0])]
            val_by_fraction = {
                f"{frac:g}": val_source_nrmse(model, val_subset, frac, mask_seed=1234, device=device)
                for frac in observed_fractions
            }
            vn = sum(val_by_fraction.values()) / len(val_by_fraction)
            logger.log({"val_source_nrmse": vn, "val_source_nrmse_by_fraction": val_by_fraction}, step=step)
            print(
                f"step {step:5d}  train {float(loss.detach().cpu()):.5f}  "
                f"val_source_nrmse {vn:.5f}  by_fraction {val_by_fraction}"
            )
            if vn < best_val:
                best_val = vn
                _save_ckpt(out_dir / "best.pt", model, config, step, vn)

    _save_ckpt(out_dir / "final.pt", model, config, tcfg["steps"], best_val)
    wall = time.time() - t0
    summary = {
        "params_M": nparam / 1e6,
        "wall_s": wall,
        "best_val_source_nrmse": best_val,
        "device": str(device),
        "n_train_windows": n_train,
    }
    resolved = dict(config)
    resolved["result"] = summary
    resolved["runtime"] = {"device": str(device), "torch": str(torch.__version__)}
    save_resolved_config(resolved, out_dir)
    print(f"done in {wall:.1f}s  best val_source_nrmse={best_val:.5f}")
    return summary


def _save_ckpt(path, model, config, step, metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "source_cfg": source_cfg_to_dict(model.source_cfg),
        "model_config": {
            "in_channels": model.in_channels,
            "base_channels": model.in_proj.out_channels,
            "depth": len(model.spatial_blocks),
        },
        "config": config,
        "step": step,
        "metric": metric,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train amortized source inference q_phi")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    config = load_yaml(args.config)
    if args.device:
        config["train"]["device"] = args.device
    if args.steps:
        config["train"]["steps"] = args.steps
    train(config, args.out)


if __name__ == "__main__":
    main()
