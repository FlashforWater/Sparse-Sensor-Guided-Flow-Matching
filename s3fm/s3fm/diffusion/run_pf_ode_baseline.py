"""Run S3GM-PF-ODE attribution baseline against S3FM learned-source flow.

This baseline asks whether S3FM's advantage is merely "using an ODE instead of
an SDE". It compares a trained VP score prior sampled with its probability-flow
ODE against the learned-source S3FM prior under the same sparse observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..diffusion.guidance import make_guided_pf_velocity
from ..diffusion.sampling import load_score_prior
from ..diffusion.vp import reverse_pf_velocity
from ..flow.sampling import load_prior
from ..flow.solvers import solve
from ..guidance.cov_g import make_guided_velocity
from ..guidance.energies import normalized_residual, observation_energy
from ..guidance.run_m4b_ablation import _build_windows, _parse_ints, _select_eval_windows
from ..guidance.schedules import constant
from ..guidance.source_inference import learned_observation_informed_source, load_source_inference
from ..measurements.base import MaskOperator
from ..reproducibility import seed_everything


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


def _recon_score_pf(score_prior, z0, x_true, observation, operator, lambda0, steps, solver, reduction):
    device = score_prior.device
    z0 = z0.to(device)
    x_true = x_true.to(device)
    observation = observation.to(device)
    if lambda0 == 0.0:
        with torch.no_grad():
            xr = solve(
                lambda z, s: reverse_pf_velocity(score_prior.model, z, s, score_prior.schedule),
                z0,
                steps=steps,
                solver=solver,
            ).z1
    else:
        def energy_fn(x0_hat):
            return observation_energy(x0_hat, observation, operator, reduction=reduction)

        vfield = make_guided_pf_velocity(score_prior.model, score_prior.schedule, energy_fn, constant(lambda0))
        xr = solve(vfield, z0, steps=steps, solver=solver).z1
    nrmse = float(torch.norm(xr - x_true) / torch.norm(x_true))
    obs = normalized_residual(xr, observation, operator)
    return nrmse, obs


def _recon_flow(model, z0, x_true, observation, operator, lambda0, steps, solver, reduction, device):
    z0 = z0.to(device)
    x_true = x_true.to(device)
    observation = observation.to(device)
    if lambda0 == 0.0:
        with torch.no_grad():
            xr = solve(lambda z, s: model(z, s), z0, steps=steps, solver=solver).z1
    else:
        def energy_fn(x1_hat):
            return observation_energy(x1_hat, observation, operator, reduction=reduction)

        vfield = make_guided_velocity(model, energy_fn, constant(lambda0))
        xr = solve(vfield, z0, steps=steps, solver=solver).z1
    nrmse = float(torch.norm(xr - x_true) / torch.norm(x_true))
    obs = normalized_residual(xr, observation, operator)
    return nrmse, obs


def _gate(summary: list[dict]) -> dict:
    by = {(row["mode"], row["nfe"]): row for row in summary}
    nfes = sorted({row["nfe"] for row in summary})
    s3fm_beats_pf = {}
    s3fm_guidance_helps = {}
    pf_guidance_helps = {}
    for nfe in nfes:
        pf_g = by[("A_s3gm_pf_guided", nfe)]["nrmse_mean"]
        pf_u = by[("A0_s3gm_pf_unguided", nfe)]["nrmse_mean"]
        s3_g = by[("B_s3fm_learned_guided", nfe)]["nrmse_mean"]
        s3_u = by[("C_s3fm_learned_unguided", nfe)]["nrmse_mean"]
        s3fm_beats_pf[str(nfe)] = s3_g <= pf_g
        s3fm_guidance_helps[str(nfe)] = s3_g <= s3_u
        pf_guidance_helps[str(nfe)] = pf_g <= pf_u
    return {
        "s3fm_learned_guided_beats_s3gm_pf_guided_by_nfe": s3fm_beats_pf,
        "s3fm_guidance_helps_by_nfe": s3fm_guidance_helps,
        "s3gm_pf_guidance_helps_by_nfe": pf_guidance_helps,
        "pass_s3fm_vs_pf_all_nfe": all(s3fm_beats_pf.values()) and all(s3fm_guidance_helps.values()),
    }


def _write_readme(path: Path, args, summary: list[dict], flags: dict) -> None:
    by = {(row["mode"], row["nfe"]): row for row in summary}
    lines = [
        "# S3GM-PF-ODE attribution baseline",
        "",
        f"Eval split: `{args.eval_split}`, windows: `{args.num_windows}`, seeds: `{args.seeds}`.",
        f"Mask seed: `{args.mask_seed}`, observed fraction: `{args.observed_fraction}`.",
        f"Guidance lambda: `{args.lambda0}`, reduction: `{args.reduction}`.",
        "",
        "| NFE | S3GM-PF guided | S3GM-PF unguided | S3FM learned guided | S3FM learned unguided | S3FM<=PF |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for nfe in sorted({row["nfe"] for row in summary}):
        pf_g = by[("A_s3gm_pf_guided", nfe)]["nrmse_mean"]
        pf_u = by[("A0_s3gm_pf_unguided", nfe)]["nrmse_mean"]
        s3_g = by[("B_s3fm_learned_guided", nfe)]["nrmse_mean"]
        s3_u = by[("C_s3fm_learned_unguided", nfe)]["nrmse_mean"]
        lines.append(f"| {nfe} | {pf_g:.4f} | {pf_u:.4f} | {s3_g:.4f} | {s3_u:.4f} | {s3_g <= pf_g} |")
    lines += [
        "",
        "## Gate",
        "",
        f"Pass S3FM learned guided vs S3GM-PF guided at all NFE: `{flags['pass_s3fm_vs_pf_all_nfe']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3GM-PF-ODE attribution baseline")
    parser.add_argument("--score-ckpt", default="experiments/kse_score_prior/best.pt")
    parser.add_argument("--info-ckpt", default="experiments/kse_prior_info/final.pt")
    parser.add_argument("--source-inference-ckpt", default="experiments/kse_source_inference/best.pt")
    parser.add_argument("--out", default="results/s3gm_pf_ode_baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--nfe", default="10,20,50,100")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--lambda0", type=float, default=2.0)
    parser.add_argument("--observed-fraction", type=float, default=0.15)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--solver", default="euler", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean"])
    args = parser.parse_args()

    seed_everything(args.eval_seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_prior = load_score_prior(args.score_ckpt, device=args.device)
    info_prior = load_prior(args.info_ckpt, device=args.device)
    source_inference = load_source_inference(args.source_inference_ckpt, device=args.device).model
    device = info_prior.device

    eval_x_all, _ = _build_windows(info_prior.config, info_prior.standardizer, args.eval_split)
    x_eval = _select_eval_windows(eval_x_all, args.num_windows, args.eval_seed)
    T, C, Nx = x_eval.shape[1:]
    operator = MaskOperator.random(T, C, Nx, observed_fraction=args.observed_fraction, seed=args.mask_seed)
    observation = operator.measure(x_eval, seed=args.mask_seed + 1)

    raw_rows: list[dict] = []
    for seed in _parse_ints(args.seeds):
        g = torch.Generator().manual_seed(seed)
        z0_score = torch.randn(x_eval.shape, generator=g)
        z0_info = learned_observation_informed_source(
            source_inference.to(device).eval(),
            observation.to(device),
            operator,
            seed=seed,
        )
        for nfe in _parse_ints(args.nfe):
            for mode, nrmse, obs in [
                ("A_s3gm_pf_guided", *_recon_score_pf(score_prior, z0_score, x_eval, observation, operator, args.lambda0, nfe, args.solver, args.reduction)),
                ("A0_s3gm_pf_unguided", *_recon_score_pf(score_prior, z0_score, x_eval, observation, operator, 0.0, nfe, args.solver, args.reduction)),
                ("B_s3fm_learned_guided", *_recon_flow(info_prior.model, z0_info, x_eval, observation, operator, args.lambda0, nfe, args.solver, args.reduction, device)),
                ("C_s3fm_learned_unguided", *_recon_flow(info_prior.model, z0_info, x_eval, observation, operator, 0.0, nfe, args.solver, args.reduction, device)),
            ]:
                raw_rows.append({
                    "seed": seed,
                    "mode": mode,
                    "nfe": nfe,
                    "nrmse": nrmse,
                    "obs_residual": obs,
                })

    summary = _summarize_rows(raw_rows)
    flags = _gate(summary)
    _write_rows(out_dir / "rows.csv", raw_rows)
    _write_rows(out_dir / "summary.csv", summary)
    (out_dir / "gate.json").write_text(json.dumps(flags, indent=2, sort_keys=True) + "\n")
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    _write_readme(out_dir / "README.md", args, summary, flags)
    print(f"wrote {out_dir}")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
