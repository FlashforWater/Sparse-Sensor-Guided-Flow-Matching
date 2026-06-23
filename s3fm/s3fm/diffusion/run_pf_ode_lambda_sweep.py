"""Sweep PF-ODE guidance strength for a fair S3GM-PF-ODE baseline.

The flow and PF-ODE endpoint estimators have different gradient scales, so a
single shared guidance lambda is not a fair comparison. This runner gives the
PF-ODE baseline its own lambda sweep and compares S3FM learned-source guidance
against the best PF-ODE result at each matched NFE.

Selecting the best PF lambda on the same evaluation set is optimistic for PF.
If S3FM still wins this sweep, the attribution evidence is stronger. For final
paper numbers, choose PF lambdas on a validation split and report on test.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..diffusion.run_pf_ode_baseline import _recon_flow, _recon_score_pf
from ..diffusion.sampling import load_score_prior
from ..flow.sampling import load_prior
from ..guidance.run_m4b_ablation import _build_windows, _parse_ints, _select_eval_windows
from ..guidance.source_inference import learned_observation_informed_source, load_source_inference
from ..measurements.base import MaskOperator
from ..reproducibility import seed_everything


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_lambda_by_nfe(text: str | None) -> dict[int, float] | None:
    if text is None or not text.strip():
        return None
    out: dict[int, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("lambda mapping must look like '10:0.1,20:0.25'")
        nfe, lam = item.split(":", 1)
        out[int(nfe.strip())] = float(lam.strip())
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
        groups[(row["mode"], row["nfe"], row["lambda0"])].append(row)

    summary = []
    for (mode, nfe, lambda0), vals in sorted(groups.items(), key=lambda kv: (int(kv[0][1]), kv[0][0], float(kv[0][2]))):
        nrmse = np.asarray([float(v["nrmse"]) for v in vals], dtype=np.float64)
        obs = np.asarray([float(v["obs_residual"]) for v in vals], dtype=np.float64)
        summary.append({
            "mode": mode,
            "nfe": int(nfe),
            "lambda0": float(lambda0),
            "n_seeds": len(vals),
            "nrmse_mean": float(nrmse.mean()),
            "nrmse_std": float(nrmse.std(ddof=0)),
            "obs_residual_mean": float(obs.mean()),
            "obs_residual_std": float(obs.std(ddof=0)),
        })
    return summary


def _best_pf_by_nfe(summary: list[dict]) -> list[dict]:
    out = []
    nfes = sorted({row["nfe"] for row in summary})
    for nfe in nfes:
        candidates = [
            row for row in summary
            if row["mode"] == "A_s3gm_pf_guided" and row["nfe"] == nfe and np.isfinite(row["nrmse_mean"])
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["nrmse_mean"])
        out.append(dict(best))
    return out


def _by_mode_nfe(summary: list[dict], mode: str) -> dict[int, dict]:
    matches = [row for row in summary if row["mode"] == mode]
    return {int(row["nfe"]): row for row in matches}


def _gate(summary: list[dict], best_pf: list[dict]) -> dict:
    best_by_nfe = {int(row["nfe"]): row for row in best_pf}
    s3_guided = _by_mode_nfe(summary, "B_s3fm_learned_guided")
    s3_unguided = _by_mode_nfe(summary, "C_s3fm_learned_unguided")
    pf_unguided = _by_mode_nfe(summary, "A0_s3gm_pf_unguided")

    s3_beats_best_pf = {}
    s3_guidance_helps = {}
    best_pf_guidance_helps = {}
    best_pf_lambda_by_nfe = {}
    for nfe in sorted(best_by_nfe):
        best = best_by_nfe[nfe]
        s3_g = s3_guided[nfe]
        s3_u = s3_unguided[nfe]
        pf_u = pf_unguided.get(nfe)
        s3_beats_best_pf[str(nfe)] = s3_g["nrmse_mean"] <= best["nrmse_mean"]
        s3_guidance_helps[str(nfe)] = s3_g["nrmse_mean"] <= s3_u["nrmse_mean"]
        best_pf_guidance_helps[str(nfe)] = (
            True if pf_u is None else best["nrmse_mean"] <= pf_u["nrmse_mean"]
        )
        best_pf_lambda_by_nfe[str(nfe)] = best["lambda0"]

    return {
        "best_pf_lambda_by_nfe": best_pf_lambda_by_nfe,
        "s3fm_learned_guided_beats_best_s3gm_pf_by_nfe": s3_beats_best_pf,
        "s3fm_guidance_helps_by_nfe": s3_guidance_helps,
        "best_s3gm_pf_guidance_helps_by_nfe": best_pf_guidance_helps,
        "pass_s3fm_vs_best_pf_all_nfe": all(s3_beats_best_pf.values()) and all(s3_guidance_helps.values()),
    }


def _write_readme(path: Path, args, summary: list[dict], best_pf: list[dict], flags: dict) -> None:
    s3_guided = _by_mode_nfe(summary, "B_s3fm_learned_guided")
    s3_unguided = _by_mode_nfe(summary, "C_s3fm_learned_unguided")
    pf_unguided = _by_mode_nfe(summary, "A0_s3gm_pf_unguided")

    lines = [
        "# S3GM-PF-ODE lambda sweep baseline",
        "",
        f"Eval split: `{args.eval_split}`, windows: `{args.num_windows}`, seeds: `{args.seeds}`.",
        f"Mask seed: `{args.mask_seed}`, observed fraction: `{args.observed_fraction}`.",
        f"PF lambdas: `{args.pf_lambdas}`.",
        f"PF lambda by NFE: `{args.pf_lambda_by_nfe}`.",
        f"S3FM lambda: `{args.s3fm_lambda}`, reduction: `{args.reduction}`.",
        "",
        (
            "The PF lambda is fixed per NFE from `--pf-lambda-by-nfe`."
            if args.pf_lambda_by_nfe
            else "The best PF lambda is selected per NFE on this run. This is optimistic for the PF baseline."
        ),
        "",
        "| NFE | best PF lambda | best PF guided | PF unguided | S3FM learned guided | S3FM unguided | S3FM<=best PF |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for best in best_pf:
        nfe = int(best["nfe"])
        pf_u = pf_unguided[nfe]["nrmse_mean"]
        s3_g = s3_guided[nfe]["nrmse_mean"]
        s3_u = s3_unguided[nfe]["nrmse_mean"]
        lines.append(
            f"| {nfe} | {best['lambda0']:.6g} | {best['nrmse_mean']:.4f} | {pf_u:.4f} | "
            f"{s3_g:.4f} | {s3_u:.4f} | {s3_g <= best['nrmse_mean']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"Pass S3FM learned guided vs best PF at all NFE: `{flags['pass_s3fm_vs_best_pf_all_nfe']}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PF-ODE guidance lambda against S3FM learned source")
    parser.add_argument("--score-ckpt", default="experiments/kse_score_prior/best.pt")
    parser.add_argument("--info-ckpt", default="experiments/kse_prior_info/final.pt")
    parser.add_argument("--source-inference-ckpt", default="experiments/kse_source_inference/best.pt")
    parser.add_argument("--out", default="results/s3gm_pf_ode_lambda_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--num-windows", type=int, default=64)
    parser.add_argument("--nfe", default="10,20,50,100")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--pf-lambdas", default="0,0.0001,0.001,0.01,0.1,0.25,0.5,1.0")
    parser.add_argument(
        "--pf-lambda-by-nfe",
        default=None,
        help="fixed validation-selected mapping, e.g. '10:0.1,20:0.25,50:0.5,100:1.0'",
    )
    parser.add_argument("--s3fm-lambda", type=float, default=2.0)
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

    nfes = _parse_ints(args.nfe)
    seeds = _parse_ints(args.seeds)
    pf_lambdas = _parse_floats(args.pf_lambdas)
    pf_lambda_by_nfe = _parse_lambda_by_nfe(args.pf_lambda_by_nfe)
    if pf_lambda_by_nfe is not None:
        missing = sorted(set(nfes) - set(pf_lambda_by_nfe))
        if missing:
            raise ValueError(f"--pf-lambda-by-nfe missing NFE entries: {missing}")

    raw_rows: list[dict] = []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        z0_score = torch.randn(x_eval.shape, generator=g)
        z0_info = learned_observation_informed_source(
            source_inference.to(device).eval(),
            observation.to(device),
            operator,
            seed=seed,
        )
        for nfe in nfes:
            # PF sweep, or one fixed validation-selected lambda for this NFE.
            # Lambda 0 is both an unguided row and a sweep candidate when swept.
            candidate_lambdas = [pf_lambda_by_nfe[nfe]] if pf_lambda_by_nfe is not None else pf_lambdas
            for pf_lambda in candidate_lambdas:
                nrmse, obs = _recon_score_pf(
                    score_prior,
                    z0_score,
                    x_eval,
                    observation,
                    operator,
                    pf_lambda,
                    nfe,
                    args.solver,
                    args.reduction,
                )
                raw_rows.append({
                    "seed": seed,
                    "mode": "A_s3gm_pf_guided",
                    "nfe": nfe,
                    "lambda0": pf_lambda,
                    "nrmse": nrmse,
                    "obs_residual": obs,
                })

            nrmse, obs = _recon_score_pf(
                score_prior,
                z0_score,
                x_eval,
                observation,
                operator,
                0.0,
                nfe,
                args.solver,
                args.reduction,
            )
            raw_rows.append({
                "seed": seed,
                "mode": "A0_s3gm_pf_unguided",
                "nfe": nfe,
                "lambda0": 0.0,
                "nrmse": nrmse,
                "obs_residual": obs,
            })

            for mode, lambda0 in [
                ("B_s3fm_learned_guided", args.s3fm_lambda),
                ("C_s3fm_learned_unguided", 0.0),
            ]:
                nrmse, obs = _recon_flow(
                    info_prior.model,
                    z0_info,
                    x_eval,
                    observation,
                    operator,
                    lambda0,
                    nfe,
                    args.solver,
                    args.reduction,
                    device,
                )
                raw_rows.append({
                    "seed": seed,
                    "mode": mode,
                    "nfe": nfe,
                    "lambda0": lambda0,
                    "nrmse": nrmse,
                    "obs_residual": obs,
                })

    summary = _summarize_rows(raw_rows)
    best_pf = _best_pf_by_nfe(summary)
    flags = _gate(summary, best_pf)

    _write_rows(out_dir / "rows.csv", raw_rows)
    _write_rows(out_dir / "summary.csv", summary)
    _write_rows(out_dir / "best_pf_by_nfe.csv", best_pf)
    (out_dir / "gate.json").write_text(json.dumps(flags, indent=2, sort_keys=True) + "\n")
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    _write_readme(out_dir / "README.md", args, summary, best_pf, flags)

    print(f"wrote {out_dir}")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
