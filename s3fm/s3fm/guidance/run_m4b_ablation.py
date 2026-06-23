"""Run the M4b three-way informative-source ablation.

This is the executable gate around :func:`three_way_ablation`. It compares:

  A. S3FM-Gauss + g_cov-G guidance
  B. S3FM-Info  + g_cov-G guidance
  C. S3FM-Info  without guidance

at matched NFE on the same held-out KSE windows, sparse mask, and seeds. The
default informative inference source is the distribution-matched marginal source
from the spec; the oracle source is exposed only as an explicitly labelled
diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "s3fm-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.kse import KSEConfig, generate_dataset
from ..data.splits import make_splits
from ..data.windows import add_channel_axis, extract_windows
from ..flow.sampling import load_prior
from ..flow.sources import SourceConfig, MarginalSourceSampler, make_source
from ..guidance.ablation import AblationRow, three_way_ablation
from ..guidance.source_inference import learned_observation_informed_source, load_source_inference
from ..guidance.source_constructor import observation_informed_source
from ..measurements.base import MaskOperator
from ..reproducibility import seed_everything


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _source_config(config: dict) -> SourceConfig:
    scfg = config.get("source")
    if scfg is None:
        raise ValueError("info checkpoint config has no source block")
    return SourceConfig(
        kind=scfg.get("kind", "spectral"),
        cutoff_k=scfg.get("cutoff_k", 8),
        factor=scfg.get("factor", 4),
        eta_std=scfg.get("eta_std", 0.05),
    )


def _build_windows(config: dict, standardizer, split_name: str):
    dcfg = config["data"]
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
    splits = make_splits(
        kse.n_trajectories,
        dcfg["train_frac"],
        dcfg["val_frac"],
        dcfg["split_seed"],
    )
    ids = getattr(splits, split_name)
    wb = extract_windows(
        ds[list(ids)],
        traj_ids=tuple(ids),
        window_length=dcfg["window_length"],
        stride=dcfg["stride"],
        physical_dt=kse.dt,
    )
    arr = standardizer.transform(wb.data, channel_axis=2)
    return torch.tensor(arr, dtype=torch.float32), wb


def _select_eval_windows(windows: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if count > windows.shape[0]:
        raise ValueError(f"requested {count} windows, but split has only {windows.shape[0]}")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(windows.shape[0], generator=g)[:count]
    return windows[idx]


def _flat_norm_per_item(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1)


@torch.no_grad()
def _path_curvature_proxy(model, z0: torch.Tensor, steps: int, device: torch.device) -> float:
    """Discrete Euler path bending proxy: sum ||d_i-d_{i-1}|| / sum ||d_i||."""
    model = model.to(device).eval()
    z = z0.to(device)
    ds = 1.0 / steps
    increments = []
    for i in range(steps):
        s = torch.full((z.shape[0],), i * ds, device=device, dtype=z.dtype)
        dz = ds * model(z, s)
        increments.append(dz.detach())
        z = z + dz
    if len(increments) < 2:
        return 0.0
    total_bend = sum(_flat_norm_per_item(increments[i] - increments[i - 1]).mean() for i in range(1, len(increments)))
    total_len = sum(_flat_norm_per_item(d).mean() for d in increments)
    return float((total_bend / (total_len + 1e-8)).detach().cpu())


def _transport_metrics(
    gauss_model,
    info_model,
    x_eval: torch.Tensor,
    observation: torch.Tensor,
    operator: MaskOperator,
    reference_fields: torch.Tensor,
    source_cfg: SourceConfig,
    seeds: list[int],
    curvature_steps: int,
    device: torch.device,
    source_inference_model=None,
) -> dict:
    out: dict[str, dict[str, float]] = {}
    gauss_disp, info_disp, marginal_disp, warm_disp = [], [], [], []
    learned_disp, learned_nrmse = [], []
    gauss_curv, info_curv = [], []

    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        z0_gauss = torch.randn(x_eval.shape, generator=g)
        z0_info = make_source(x_eval, source_cfg, generator=torch.Generator().manual_seed(seed))
        sampler = MarginalSourceSampler(reference_fields, source_cfg, seed=seed)
        z0_marginal = sampler.sample(x_eval.shape[0], seed=seed)
        z0_warm = observation_informed_source(observation, operator, source_cfg, seed=seed)

        gauss_disp.append(float(_flat_norm_per_item(x_eval - z0_gauss).mean()))
        info_disp.append(float(_flat_norm_per_item(x_eval - z0_info).mean()))
        marginal_disp.append(float(_flat_norm_per_item(x_eval - z0_marginal).mean()))
        warm_disp.append(float(_flat_norm_per_item(x_eval - z0_warm).mean()))
        if source_inference_model is not None:
            z0_learned = learned_observation_informed_source(
                source_inference_model.to(device).eval(),
                observation.to(device),
                operator,
                seed=seed,
            ).cpu()
            learned_disp.append(float(_flat_norm_per_item(x_eval - z0_learned).mean()))
            learned_nrmse.append(float(torch.norm(x_eval - z0_learned) / torch.norm(x_eval)))
        gauss_curv.append(_path_curvature_proxy(gauss_model, z0_gauss, curvature_steps, device))
        info_curv.append(_path_curvature_proxy(info_model, z0_info, curvature_steps, device))

    def summarize(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    out["gauss_train_coupling"] = {
        "displacement": summarize(gauss_disp),
        "curvature_proxy": summarize(gauss_curv),
    }
    out["info_dependent_train_coupling"] = {
        "displacement": summarize(info_disp),
        "curvature_proxy": summarize(info_curv),
    }
    out["info_marginal_inference_source"] = {
        "displacement_to_eval_target": summarize(marginal_disp),
        "note": "This is the real marginal inference source, not target-coupled S(X_true).",
    }
    out["info_warm_inference_source"] = {
        "displacement_to_eval_target": summarize(warm_disp),
        "note": "This uses only sparse observations y and a fixed mask-aware low-frequency reconstruction source.",
    }
    if learned_disp:
        out["info_learned_inference_source"] = {
            "displacement_to_eval_target": summarize(learned_disp),
            "nrmse_to_eval_target": summarize(learned_nrmse),
            "note": "This uses q_phi(y,H) projected to the low-bandwidth source space, not x_true.",
        }
    return out


def _write_rows(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["nfe"])].append(row)

    summary = []
    for (mode, nfe), vals in sorted(groups.items(), key=lambda kv: (int(kv[0][1]), kv[0][0])):
        nrmse = np.asarray([float(v["nrmse"]) for v in vals])
        obs = np.asarray([float(v["obs_residual"]) for v in vals])
        summary.append({
            "mode": mode,
            "nfe": int(nfe),
            "n_seeds": len(vals),
            "nrmse_mean": float(nrmse.mean()),
            "nrmse_std": float(nrmse.std(ddof=0)),
            "obs_residual_mean": float(obs.mean()),
            "obs_residual_std": float(obs.std(ddof=0)),
        })
    return summary


def _plot_metric(summary: list[dict], metric: str, out_path: Path) -> None:
    modes = sorted({row["mode"] for row in summary})
    plt.figure(figsize=(6.5, 4.0))
    for mode in modes:
        rows = [r for r in summary if r["mode"] == mode]
        xs = [r["nfe"] for r in rows]
        ys = [r[f"{metric}_mean"] for r in rows]
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xlabel("NFE")
    plt.ylabel(metric.replace("_", " "))
    plt.xscale("log")
    plt.xticks(sorted({r["nfe"] for r in summary}), sorted({r["nfe"] for r in summary}))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _gate_flags(summary: list[dict]) -> dict:
    by_key = {(r["mode"], r["nfe"]): r for r in summary}
    nfes = sorted({r["nfe"] for r in summary})
    beats_gauss = {}
    guidance_helps = {}
    for nfe in nfes:
        gauss = by_key[("A_gauss_guided", nfe)]["nrmse_mean"]
        info = by_key[("B_info_guided", nfe)]["nrmse_mean"]
        unguided = by_key[("C_info_unguided", nfe)]["nrmse_mean"]
        beats_gauss[str(nfe)] = info <= gauss
        guidance_helps[str(nfe)] = info <= unguided
    return {
        "info_guided_beats_gauss_guided_by_nfe": beats_gauss,
        "info_guidance_beats_info_unguided_by_nfe": guidance_helps,
        "pass_all_matched_nfe": all(beats_gauss.values()) and all(guidance_helps.values()),
    }


def _write_readme(path: Path, args, summary: list[dict], metrics: dict, flags: dict) -> None:
    lines = [
        "# M4b three-way ablation",
        "",
        f"Info inference source: `{args.info_source}`.",
        f"Eval split: `{args.eval_split}`, windows: `{args.num_windows}`, seeds: `{args.seeds}`.",
        f"Guidance lambda: `{args.lambda0}`, reduction: `{args.reduction}`, observed fraction: `{args.observed_fraction}`.",
        "",
        "## Reconstruction summary",
        "",
        "| NFE | A Gauss guided nRMSE | B Info guided nRMSE | C Info unguided nRMSE | B<=A | B<=C |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    by_key = {(r["mode"], r["nfe"]): r for r in summary}
    for nfe in sorted({r["nfe"] for r in summary}):
        a = by_key[("A_gauss_guided", nfe)]["nrmse_mean"]
        b = by_key[("B_info_guided", nfe)]["nrmse_mean"]
        c = by_key[("C_info_unguided", nfe)]["nrmse_mean"]
        lines.append(f"| {nfe} | {a:.4f} | {b:.4f} | {c:.4f} | {b <= a} | {b <= c} |")

    gd = metrics["gauss_train_coupling"]["displacement"]["mean"]
    idp = metrics["info_dependent_train_coupling"]["displacement"]["mean"]
    gc = metrics["gauss_train_coupling"]["curvature_proxy"]["mean"]
    ic = metrics["info_dependent_train_coupling"]["curvature_proxy"]["mean"]
    md = metrics["info_marginal_inference_source"]["displacement_to_eval_target"]["mean"]
    wd = metrics["info_warm_inference_source"]["displacement_to_eval_target"]["mean"]
    lines += [
        "",
        "## Mechanism diagnostics",
        "",
        f"- Train-coupled displacement, Gauss: `{gd:.4f}`",
        f"- Train-coupled displacement, Info: `{idp:.4f}`",
        f"- Train-coupled curvature proxy, Gauss: `{gc:.4f}`",
        f"- Train-coupled curvature proxy, Info: `{ic:.4f}`",
        f"- Marginal inference source displacement to current eval target: `{md:.4f}`",
        f"- Warm inference source displacement to current eval target: `{wd:.4f}`",
    ]
    if "info_learned_inference_source" in metrics:
        ld = metrics["info_learned_inference_source"]["displacement_to_eval_target"]["mean"]
        ln = metrics["info_learned_inference_source"]["nrmse_to_eval_target"]["mean"]
        lines += [
            f"- Learned inference source displacement to current eval target: `{ld:.4f}`",
            f"- Learned inference source nRMSE to current eval target: `{ln:.4f}`",
        ]
    lines += [
        "",
        "The marginal source is distribution-matched but not target-short. The warm source is a fixed measurement-informed low-frequency reconstruction from `y`, not an oracle `S(X_true)`.",
        "",
        "## Gate",
        "",
        f"Pass all matched NFE: `{flags['pass_all_matched_nfe']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M4b three-way source/guidance ablation")
    parser.add_argument("--gauss-ckpt", default="experiments/kse_prior/final.pt")
    parser.add_argument("--info-ckpt", default="experiments/kse_prior_info/final.pt")
    parser.add_argument("--out", default="results/m4b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--ref-windows", type=int, default=1024)
    parser.add_argument("--nfe", default="10,20,50,100")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--lambda0", type=float, default=5.0)
    parser.add_argument("--observed-fraction", type=float, default=0.15)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--solver", default="euler", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean"])
    parser.add_argument("--info-source", default="marginal", choices=["marginal", "warm", "learned", "oracle"])
    parser.add_argument("--source-inference-ckpt", default=None)
    parser.add_argument("--curvature-steps", type=int, default=50)
    args = parser.parse_args()

    seed_everything(args.eval_seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gauss_prior = load_prior(args.gauss_ckpt, device=args.device)
    info_prior = load_prior(args.info_ckpt, device=args.device)
    device = info_prior.device
    source_cfg = _source_config(info_prior.config)
    source_inference_model = None
    if args.info_source == "learned":
        if args.source_inference_ckpt is None:
            raise ValueError("--info-source learned requires --source-inference-ckpt")
        source_inference_model = load_source_inference(args.source_inference_ckpt, device=args.device).model

    train_x, _ = _build_windows(info_prior.config, info_prior.standardizer, "train")
    eval_x_all, _ = _build_windows(info_prior.config, info_prior.standardizer, args.eval_split)
    x_eval = _select_eval_windows(eval_x_all, args.num_windows, args.eval_seed)
    reference_fields = _select_eval_windows(train_x, min(args.ref_windows, train_x.shape[0]), args.eval_seed + 17)

    T, C, Nx = x_eval.shape[1:]
    operator = MaskOperator.random(T, C, Nx, observed_fraction=args.observed_fraction, seed=args.mask_seed)
    observation = operator.measure(x_eval, seed=args.mask_seed + 1)

    nfe_list = _parse_ints(args.nfe)
    seeds = _parse_ints(args.seeds)
    raw_rows: list[dict] = []
    for seed in seeds:
        rows: list[AblationRow] = three_way_ablation(
            gauss_prior.model,
            info_prior.model,
            x_eval,
            observation,
            operator,
            source_cfg,
            reference_fields,
            lambda0=args.lambda0,
            nfe_list=nfe_list,
            solver=args.solver,
            reduction=args.reduction,
            seed=seed,
            device=device,
            info_source=args.info_source,
            source_inference_model=source_inference_model,
        )
        for row in rows:
            raw_rows.append({
                "seed": seed,
                "mode": row.mode,
                "nfe": row.nfe,
                "nrmse": row.nrmse,
                "obs_residual": row.obs_residual,
            })

    summary = _summarize_rows(raw_rows)
    metrics = _transport_metrics(
        gauss_prior.model,
        info_prior.model,
        x_eval,
        observation,
        operator,
        reference_fields,
        source_cfg,
        seeds,
        args.curvature_steps,
        device,
        source_inference_model=source_inference_model,
    )
    flags = _gate_flags(summary)

    _write_rows(out_dir / "rows.csv", raw_rows)
    _write_rows(out_dir / "summary.csv", summary)
    (out_dir / "transport_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (out_dir / "gate.json").write_text(json.dumps(flags, indent=2, sort_keys=True) + "\n")
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    _plot_metric(summary, "nrmse", out_dir / "nrmse_vs_nfe.png")
    _plot_metric(summary, "obs_residual", out_dir / "obs_residual_vs_nfe.png")
    _write_readme(out_dir / "README.md", args, summary, metrics, flags)

    print(f"wrote {out_dir}")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
