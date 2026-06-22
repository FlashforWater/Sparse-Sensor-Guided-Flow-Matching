"""Full training entry point for the unconditional KSE flow-matching prior.

Builds the KSE dataset, fits normalization on train, trains the velocity model
with the flow-matching loss (resampled Gaussian source each step — the true FM
objective), tracks an EMA copy, monitors validation loss AND endpoint-recovery
nRMSE (since low FM loss alone is insufficient, per the spec), and saves
checkpoints + normalization stats + the resolved config.

Device-agnostic: runs on CPU / MPS / CUDA. The same script works on a laptop
and on a cluster; only `train.device` and sizes change.

Usage:
    python -m s3fm.train_flow --config configs/kse_flow_prior.yaml --out experiments/kse_prior
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_yaml, save_resolved_config
from .data.kse import KSEConfig, generate_dataset
from .data.splits import ChannelStandardizer, make_splits
from .data.windows import add_channel_axis, extract_windows
from .flow.ema import EMA
from .flow.paths import endpoint_estimate, flow_matching_loss, interpolate_path
from .flow.training import gaussian_source_like
from .logging_utils import JsonlLogger
from .models.video_unet_velocity import VideoUNetVelocity1D
from .reproducibility import seed_everything, select_device


def build_windows(dcfg: dict):
    """Generate KSE, split, normalize, and extract train/val window tensors."""
    kse = KSEConfig(
        L=dcfg["L"], Nx=dcfg["Nx"], dt=dcfg["dt"], n_steps=dcfg["n_steps"],
        warmup_steps=dcfg["warmup_steps"], n_trajectories=dcfg["n_trajectories"],
        ic_scale=dcfg["ic_scale"],
    )
    ds = add_channel_axis(generate_dataset(kse, base_seed=0))  # (n, steps, 1, Nx)
    splits = make_splits(kse.n_trajectories, dcfg["train_frac"], dcfg["val_frac"], dcfg["split_seed"])
    std = ChannelStandardizer.fit(ds[list(splits.train)], channel_axis=2)

    def windows_for(ids):
        wb = extract_windows(
            ds[list(ids)], traj_ids=tuple(ids),
            window_length=dcfg["window_length"], stride=dcfg["stride"], physical_dt=kse.dt,
        )
        return torch.tensor(std.transform(wb.data, 2), dtype=torch.float32)

    return windows_for(splits.train), windows_for(splits.val), std, splits, kse


@torch.no_grad()
def endpoint_recovery_nrmse(model, val_x, device, s_val=0.9, seed=123):
    """Mean nRMSE of X1_hat vs true Z1 at flow_time s_val (held-out windows)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(val_x.shape, generator=g).to(device)
    z1 = val_x.to(device)
    s = torch.full((val_x.shape[0],), s_val, device=device)
    zs = interpolate_path(z0, z1, s)
    x1 = endpoint_estimate(zs, model(zs, s), s)
    return float(torch.norm(x1 - z1) / torch.norm(z1))


@torch.no_grad()
def val_fm_loss(model, val_x, device, seed=7):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(val_x.shape, generator=g).to(device)
    z1 = val_x.to(device)
    s = torch.rand(val_x.shape[0], generator=g).to(device)
    zs = interpolate_path(z0, z1, s)
    return float(flow_matching_loss(model(zs, s), z0, z1))


def train(config: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    seed_everything(config["experiment"]["seed"])
    device = select_device(config["train"]["device"])
    logger = JsonlLogger(out_dir)

    train_x, val_x, std, splits, kse = build_windows(config["data"])
    print(f"device={device}  train windows={tuple(train_x.shape)}  val windows={tuple(val_x.shape)}")

    mcfg = config["model"]
    model = VideoUNetVelocity1D(
        in_channels=1, base_channels=mcfg["base_channels"],
        depth=mcfg["depth"], t_emb_dim=mcfg["t_emb_dim"],
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M")

    tcfg = config["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    ema = EMA(model, decay=tcfg["ema_decay"]).to(device)

    train_x = train_x.to(device)
    n_train = train_x.shape[0]
    gen = torch.Generator().manual_seed(config["experiment"]["seed"])

    best_val = float("inf")
    t0 = time.time()
    for step in range(tcfg["steps"]):
        model.train()
        idx = torch.randint(0, n_train, (tcfg["batch_size"],), generator=gen)
        z1 = train_x[idx]
        z0 = torch.randn(z1.shape, generator=gen).to(device)
        s = torch.rand(z1.shape[0], generator=gen).to(device)
        zs = interpolate_path(z0, z1, s)
        loss = flow_matching_loss(model(zs, s), z0, z1)
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(model)
        loss_val = float(loss.detach().cpu())

        if step % tcfg["log_every"] == 0 or step == tcfg["steps"] - 1:
            logger.log({"train_loss": loss_val}, step=step)

        if step % tcfg["val_every"] == 0 or step == tcfg["steps"] - 1:
            vl = val_fm_loss(ema.shadow, val_x, device)
            er = endpoint_recovery_nrmse(ema.shadow, val_x, device)
            logger.log({"val_fm_loss": vl, "val_endpoint_nrmse": er}, step=step)
            print(f"step {step:5d}  train {loss_val:.4f}  val_fm {vl:.4f}  endpoint_nrmse {er:.4f}")
            if er < best_val:
                best_val = er
                _save_ckpt(out_dir / "best.pt", model, ema, std, config, step, er)

        if tcfg["ckpt_every"] and step > 0 and step % tcfg["ckpt_every"] == 0:
            _save_ckpt(out_dir / f"step_{step}.pt", model, ema, std, config, step, None)

    _save_ckpt(out_dir / "final.pt", model, ema, std, config, tcfg["steps"], best_val)
    wall = time.time() - t0
    summary = {
        "params_M": nparam / 1e6, "wall_s": wall, "best_endpoint_nrmse": best_val,
        "device": str(device), "n_train_windows": n_train,
    }
    config_to_save = dict(config)
    config_to_save["result"] = summary
    config_to_save["runtime"] = {"device": str(device), "torch": str(torch.__version__)}
    save_resolved_config(config_to_save, out_dir)
    print(f"done in {wall:.1f}s  best endpoint nRMSE={best_val:.4f}")
    return summary


def _save_ckpt(path, model, ema, std, config, step, metric):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "norm_mean": std.mean, "norm_std": std.std,
        "config": config, "step": step, "metric": metric,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train unconditional KSE flow prior")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None, help="override train.device")
    p.add_argument("--steps", type=int, default=None, help="override train.steps")
    args = p.parse_args()
    config = load_yaml(args.config)
    if args.device:
        config["train"]["device"] = args.device
    if args.steps:
        config["train"]["steps"] = args.steps
    train(config, args.out)


if __name__ == "__main__":
    main()
