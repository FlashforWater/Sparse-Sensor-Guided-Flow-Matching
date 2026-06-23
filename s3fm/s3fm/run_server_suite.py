"""Server-side experiment suite for S3FM learned-source attribution tests.

This runner is intentionally a thin orchestrator around the existing executable
modules. It runs, per observed fraction and mask seed:

1. S3FM learned-source M4b gate.
2. PF-ODE lambda sweep on validation.
3. PF-ODE test with the validation-selected per-NFE lambdas fixed.
4. A compact aggregate CSV/README.

The suite is resume-friendly: completed sub-runs are skipped unless
``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _slug_float(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _format_lambda_by_nfe(best_pf_rows: list[dict]) -> str:
    pairs = sorted((int(row["nfe"]), float(row["lambda0"])) for row in best_pf_rows)
    return ",".join(f"{nfe}:{lam:g}" for nfe, lam in pairs)


def _row_by_mode_nfe(rows: list[dict], mode: str, nfe: int) -> dict | None:
    for row in rows:
        if row.get("mode") == mode and int(row.get("nfe", -1)) == nfe:
            return row
    return None


@dataclass(frozen=True)
class SubRun:
    name: str
    out_dir: Path
    cmd: list[str]
    required_files: tuple[str, ...]


def _is_complete(run: SubRun) -> bool:
    return all((run.out_dir / rel).exists() for rel in run.required_files)


def _run_subprocess(run: SubRun, suite_out: Path, *, dry_run: bool, overwrite: bool) -> None:
    suite_out.mkdir(parents=True, exist_ok=True)
    log_path = suite_out / "commands.jsonl"
    record = {
        "name": run.name,
        "out_dir": str(run.out_dir),
        "cmd": run.cmd,
    }
    if _is_complete(run) and not overwrite:
        record["status"] = "skipped_complete"
        with open(log_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[skip] {run.name}: {run.out_dir}")
        return

    run.out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        record["status"] = "dry_run"
        with open(log_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print("[dry-run] " + " ".join(run.cmd))
        return

    print(f"[run] {run.name}")
    print("      " + " ".join(run.cmd))
    with open(log_path, "a") as f:
        record["status"] = "started"
        f.write(json.dumps(record, sort_keys=True) + "\n")
    subprocess.run(run.cmd, check=True)
    with open(log_path, "a") as f:
        record["status"] = "completed"
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _m4b_run(args, obs: float, mask_seed: int, out_dir: Path) -> SubRun:
    cmd = [
        sys.executable,
        "-m",
        "s3fm.guidance.run_m4b_ablation",
        "--device",
        args.device,
        "--out",
        str(out_dir),
        "--gauss-ckpt",
        args.gauss_ckpt,
        "--info-ckpt",
        args.info_ckpt,
        "--info-source",
        "learned",
        "--source-inference-ckpt",
        args.source_inference_ckpt,
        "--eval-split",
        "test",
        "--num-windows",
        str(args.num_windows),
        "--ref-windows",
        str(args.ref_windows),
        "--nfe",
        args.nfe,
        "--seeds",
        args.seeds,
        "--lambda0",
        str(args.s3fm_lambda),
        "--observed-fraction",
        str(obs),
        "--mask-seed",
        str(mask_seed),
        "--eval-seed",
        str(args.eval_seed),
        "--solver",
        args.solver,
        "--reduction",
        args.reduction,
    ]
    return SubRun("m4b_learned", out_dir, cmd, ("summary.csv", "gate.json", "transport_metrics.json"))


def _pf_val_run(args, obs: float, mask_seed: int, out_dir: Path) -> SubRun:
    cmd = [
        sys.executable,
        "-m",
        "s3fm.diffusion.run_pf_ode_lambda_sweep",
        "--device",
        args.device,
        "--out",
        str(out_dir),
        "--score-ckpt",
        args.score_ckpt,
        "--info-ckpt",
        args.info_ckpt,
        "--source-inference-ckpt",
        args.source_inference_ckpt,
        "--eval-split",
        "val",
        "--num-windows",
        str(args.num_windows),
        "--nfe",
        args.nfe,
        "--seeds",
        args.seeds,
        "--pf-lambdas",
        args.pf_lambdas,
        "--s3fm-lambda",
        str(args.s3fm_lambda),
        "--observed-fraction",
        str(obs),
        "--mask-seed",
        str(mask_seed),
        "--eval-seed",
        str(args.eval_seed),
        "--solver",
        args.solver,
        "--reduction",
        args.reduction,
    ]
    return SubRun("pf_val_sweep", out_dir, cmd, ("best_pf_by_nfe.csv", "summary.csv", "gate.json"))


def _pf_test_run(args, obs: float, mask_seed: int, out_dir: Path, lambda_by_nfe: str) -> SubRun:
    cmd = [
        sys.executable,
        "-m",
        "s3fm.diffusion.run_pf_ode_lambda_sweep",
        "--device",
        args.device,
        "--out",
        str(out_dir),
        "--score-ckpt",
        args.score_ckpt,
        "--info-ckpt",
        args.info_ckpt,
        "--source-inference-ckpt",
        args.source_inference_ckpt,
        "--eval-split",
        "test",
        "--num-windows",
        str(args.num_windows),
        "--nfe",
        args.nfe,
        "--seeds",
        args.seeds,
        "--pf-lambda-by-nfe",
        lambda_by_nfe,
        "--s3fm-lambda",
        str(args.s3fm_lambda),
        "--observed-fraction",
        str(obs),
        "--mask-seed",
        str(mask_seed),
        "--eval-seed",
        str(args.eval_seed),
        "--solver",
        args.solver,
        "--reduction",
        args.reduction,
    ]
    return SubRun("pf_test_val_tuned", out_dir, cmd, ("best_pf_by_nfe.csv", "summary.csv", "gate.json"))


def _aggregate_case(case_dir: Path, obs: float, mask_seed: int, nfes: Iterable[int]) -> list[dict]:
    m4b_dir = case_dir / "m4b_learned"
    pf_val_dir = case_dir / "pf_val_sweep"
    pf_test_dir = case_dir / "pf_test_val_tuned"

    if not (m4b_dir / "summary.csv").exists() or not (pf_test_dir / "summary.csv").exists():
        return []

    m4b_summary = _read_csv(m4b_dir / "summary.csv")
    pf_test_summary = _read_csv(pf_test_dir / "summary.csv")
    best_pf = _read_csv(pf_test_dir / "best_pf_by_nfe.csv")
    m4b_gate = _read_json(m4b_dir / "gate.json")
    pf_gate = _read_json(pf_test_dir / "gate.json")
    source_metrics = {}
    metrics_path = m4b_dir / "transport_metrics.json"
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        learned = metrics.get("info_learned_inference_source", {})
        source_metrics = {
            "learned_source_disp": learned.get("displacement_to_eval_target", {}).get("mean", ""),
            "learned_source_nrmse": learned.get("nrmse_to_eval_target", {}).get("mean", ""),
        }

    best_pf_by_nfe = {int(row["nfe"]): row for row in best_pf}
    out = []
    for nfe in nfes:
        gauss = _row_by_mode_nfe(m4b_summary, "A_gauss_guided", nfe)
        s3_m4b = _row_by_mode_nfe(m4b_summary, "B_info_guided", nfe)
        s3_m4b_ung = _row_by_mode_nfe(m4b_summary, "C_info_unguided", nfe)
        pf = best_pf_by_nfe.get(nfe)
        pf_ung = _row_by_mode_nfe(pf_test_summary, "A0_s3gm_pf_unguided", nfe)
        s3_pf = _row_by_mode_nfe(pf_test_summary, "B_s3fm_learned_guided", nfe)
        s3_pf_ung = _row_by_mode_nfe(pf_test_summary, "C_s3fm_learned_unguided", nfe)
        if not all([gauss, s3_m4b, s3_m4b_ung, pf, pf_ung, s3_pf, s3_pf_ung]):
            continue
        row = {
            "observed_fraction": obs,
            "mask_seed": mask_seed,
            "nfe": nfe,
            "m4b_pass_all": m4b_gate.get("pass_all_matched_nfe", ""),
            "pf_val_tuned_gate": pf_gate.get("pass_s3fm_vs_best_pf_all_nfe", ""),
            "gauss_guided_nrmse": gauss["nrmse_mean"],
            "s3fm_learned_guided_nrmse": s3_m4b["nrmse_mean"],
            "s3fm_learned_unguided_nrmse": s3_m4b_ung["nrmse_mean"],
            "pf_val_selected_lambda": pf["lambda0"],
            "pf_val_tuned_guided_nrmse": pf["nrmse_mean"],
            "pf_val_tuned_guided_std": pf["nrmse_std"],
            "pf_unguided_nrmse": pf_ung["nrmse_mean"],
            "pf_runner_s3fm_guided_nrmse": s3_pf["nrmse_mean"],
            "pf_runner_s3fm_unguided_nrmse": s3_pf_ung["nrmse_mean"],
        }
        row.update(source_metrics)
        out.append(row)
    return out


def _write_readme(out_dir: Path, args, aggregate_rows: list[dict]) -> None:
    cases = sorted({(row["observed_fraction"], row["mask_seed"]) for row in aggregate_rows})
    pass_m4b = sum(
        1 for obs, mask in cases
        if all(str(row["m4b_pass_all"]) == "True" for row in aggregate_rows if row["observed_fraction"] == obs and row["mask_seed"] == mask)
    )
    pass_pf = sum(
        1 for obs, mask in cases
        if all(str(row["pf_val_tuned_gate"]) == "True" for row in aggregate_rows if row["observed_fraction"] == obs and row["mask_seed"] == mask)
    )
    lines = [
        "# S3FM Server Suite",
        "",
        "This directory was produced by `python -m s3fm.run_server_suite`.",
        "",
        "## Configuration",
        "",
        f"- observed fractions: `{args.observed_fractions}`",
        f"- mask seeds: `{args.mask_seeds}`",
        f"- source seeds: `{args.seeds}`",
        f"- num windows: `{args.num_windows}`",
        f"- NFE: `{args.nfe}`",
        f"- S3FM lambda: `{args.s3fm_lambda}`",
        f"- PF lambda sweep: `{args.pf_lambdas}`",
        "",
        "## Aggregate",
        "",
        f"- completed cases: `{len(cases)}`",
        f"- M4b learned-source pass cases: `{pass_m4b}/{len(cases)}`",
        f"- val-tuned PF attribution pass cases: `{pass_pf}/{len(cases)}`",
        "",
        "Detailed rows are in `aggregate_by_mask.csv`.",
        "",
        "## Directory Layout",
        "",
        "- `obs_<fraction>/mask_<seed>/m4b_learned/`",
        "- `obs_<fraction>/mask_<seed>/pf_val_sweep/`",
        "- `obs_<fraction>/mask_<seed>/pf_test_val_tuned/`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run server-side S3FM attribution suite")
    parser.add_argument("--out", default="results/server_suite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observed-fractions", default="0.15")
    parser.add_argument("--mask-seeds", default="0,1,2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--num-windows", type=int, default=128)
    parser.add_argument("--ref-windows", type=int, default=1024)
    parser.add_argument("--nfe", default="10,20,50,100")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--solver", default="euler", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--reduction", default="sum", choices=["sum", "mean"])
    parser.add_argument("--s3fm-lambda", type=float, default=2.0)
    parser.add_argument("--pf-lambdas", default="0,0.0001,0.001,0.01,0.03,0.1,0.25,0.5,1.0")
    parser.add_argument("--gauss-ckpt", default="experiments/kse_prior/final.pt")
    parser.add_argument("--info-ckpt", default="experiments/kse_prior_info/final.pt")
    parser.add_argument("--score-ckpt", default="experiments/kse_score_prior/best.pt")
    parser.add_argument("--source-inference-ckpt", default="experiments/kse_source_inference/best.pt")
    parser.add_argument("--skip-m4b", action="store_true")
    parser.add_argument("--skip-pf", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    observed_fractions = _parse_floats(args.observed_fractions)
    mask_seeds = _parse_ints(args.mask_seeds)
    nfes = _parse_ints(args.nfe)

    for obs in observed_fractions:
        for mask_seed in mask_seeds:
            case_dir = out_dir / f"obs_{_slug_float(obs)}" / f"mask_{mask_seed}"
            if not args.skip_m4b:
                _run_subprocess(
                    _m4b_run(args, obs, mask_seed, case_dir / "m4b_learned"),
                    out_dir,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
            if not args.skip_pf:
                pf_val = _pf_val_run(args, obs, mask_seed, case_dir / "pf_val_sweep")
                _run_subprocess(pf_val, out_dir, dry_run=args.dry_run, overwrite=args.overwrite)
                if args.dry_run:
                    continue
                best_pf_path = pf_val.out_dir / "best_pf_by_nfe.csv"
                if not best_pf_path.exists():
                    raise FileNotFoundError(f"PF validation sweep did not produce {best_pf_path}")
                lambda_by_nfe = _format_lambda_by_nfe(_read_csv(best_pf_path))
                _run_subprocess(
                    _pf_test_run(args, obs, mask_seed, case_dir / "pf_test_val_tuned", lambda_by_nfe),
                    out_dir,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )

    if args.dry_run:
        print(f"dry-run complete; commands written to {out_dir / 'commands.jsonl'}")
        return

    aggregate_rows: list[dict] = []
    for obs in observed_fractions:
        for mask_seed in mask_seeds:
            case_dir = out_dir / f"obs_{_slug_float(obs)}" / f"mask_{mask_seed}"
            aggregate_rows.extend(_aggregate_case(case_dir, obs, mask_seed, nfes))
    _write_csv(out_dir / "aggregate_by_mask.csv", aggregate_rows)
    _write_readme(out_dir, args, aggregate_rows)
    print(f"wrote aggregate: {out_dir / 'aggregate_by_mask.csv'}")
    print(f"wrote summary: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
