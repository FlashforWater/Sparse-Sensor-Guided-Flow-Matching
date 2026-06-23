"""Train a VP epsilon-prediction score prior for S3GM-PF-ODE baselines.

This is intentionally parallel to ``train_flow.py`` but trains a diffusion-style
score prior instead of a flow velocity prior. The checkpoint can be sampled with
the deterministic probability-flow ODE to isolate the "ODE instead of SDE"
attribution baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import load_yaml, save_resolved_config
from .data.kse import KSEConfig, generate_dataset
from .data.splits import ChannelStandardizer, make_splits
from .data.windows import add_channel_axis, extract_windows
from .diffusion.training import denoise_recovery_nrmse, diffusion_epsilon_loss, val_epsilon_loss
from .diffusion.vp import VPSchedule
from .flow.ema import EMA
from .logging_utils import JsonlLogger
from .models.video_unet_velocity import VideoUNetVelocity1D
from .reproducibility import seed_everything, select_device


def build_windows(dcfg: dict):
    kse = KSEConfig(
        L=dcfg["L"],
        Nx=dcfg["Nx"],
        dt=dcfg["dt"],
        n_steps=dcfg["n_steps"],
        warmup_steps=dcfg["warmup_steps"],
        n_trajectories=dcfg["n_trajectories"],
        ic_scale=dcfg["ic_scale"],
    )
    ds = add_channel_axis(generate_dataset(kse, base_seed=0))
    splits = make_splits(kse.n_trajectories, dcfg["train_frac"], dcfg["val_frac"], dcfg["split_seed"])
    std = ChannelStandardizer.fit(ds[list(splits.train)], channel_axis=2)

    def windows_for(ids):
        wb = extract_windows(
            ds[list(ids)],
            traj_ids=tuple(ids),
            window_length=dcfg["window_length"],
            stride=dcfg["stride"],
            physical_dt=kse.dt,
        )
        return torch.tensor(std.transform(wb.data, 2), dtype=torch.float32)

    return windows_for(splits.train), windows_for(splits.val), std, splits, kse


def train(config: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    seed_everything(config["experiment"]["seed"])
    device = select_device(config["train"]["device"])
    logger = JsonlLogger(out_dir)
    schedule = VPSchedule.from_config(config.get("diffusion"))

    train_x, val_x, std, splits, kse = build_windows(config["data"])
    print(f"device={device}  train windows={tuple(train_x.shape)}  val windows={tuple(val_x.shape)}")
    print(f"VP schedule: {schedule}")

    mcfg = config["model"]
    model = VideoUNetVelocity1D(
        in_channels=1,
        base_channels=mcfg["base_channels"],
        depth=mcfg["depth"],
        t_emb_dim=mcfg["t_emb_dim"],
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"score params: {nparam/1e6:.2f}M")

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
        x0 = train_x[idx]
        loss = diffusion_epsilon_loss(model, x0, schedule, generator=gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        loss_val = float(loss.detach().cpu())

        if step % tcfg["log_every"] == 0 or step == tcfg["steps"] - 1:
            logger.log({"train_eps_loss": loss_val}, step=step)

        if step % tcfg["val_every"] == 0 or step == tcfg["steps"] - 1:
            vl = val_epsilon_loss(ema.shadow, val_x, schedule, device)
            er = denoise_recovery_nrmse(
                ema.shadow,
                val_x,
                schedule,
                device,
                t_val=float(config.get("validation", {}).get("t_val", 0.5)),
            )
            logger.log({"val_eps_loss": vl, "val_denoise_nrmse": er}, step=step)
            print(f"step {step:5d}  train {loss_val:.4f}  val_eps {vl:.4f}  denoise_nrmse {er:.4f}")
            if er < best_val:
                best_val = er
                _save_ckpt(out_dir / "best.pt", model, ema, std, config, step, er)

        if tcfg["ckpt_every"] and step > 0 and step % tcfg["ckpt_every"] == 0:
            _save_ckpt(out_dir / f"step_{step}.pt", model, ema, std, config, step, None)

    _save_ckpt(out_dir / "final.pt", model, ema, std, config, tcfg["steps"], best_val)
    wall = time.time() - t0
    summary = {
        "params_M": nparam / 1e6,
        "wall_s": wall,
        "best_denoise_nrmse": best_val,
        "device": str(device),
        "n_train_windows": n_train,
    }
    config_to_save = dict(config)
    config_to_save["result"] = summary
    config_to_save["runtime"] = {"device": str(device), "torch": str(torch.__version__)}
    save_resolved_config(config_to_save, out_dir)
    print(f"done in {wall:.1f}s  best denoise nRMSE={best_val:.4f}")
    return summary


def _save_ckpt(path, model, ema, std, config, step, metric):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "norm_mean": std.mean,
        "norm_std": std.std,
        "config": config,
        "step": step,
        "metric": metric,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train VP epsilon-prediction score prior")
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
