"""M0 gate: determinism and config/logging plumbing."""

from __future__ import annotations

import json

import numpy as np
import torch

from s3fm.config import apply_dotted_overrides, deep_merge, save_resolved_config
from s3fm.logging_utils import JsonlLogger
from s3fm.reproducibility import seed_everything, select_device
from s3fm.smoke import run_smoke


def test_seed_everything_matches_across_runs():
    seed_everything(123)
    a = torch.randn(16)
    n_a = np.random.rand(16)

    seed_everything(123)
    b = torch.randn(16)
    n_b = np.random.rand(16)

    assert torch.equal(a, b)
    assert np.allclose(n_a, n_b)


def test_different_seeds_differ():
    seed_everything(0)
    a = torch.randn(16)
    seed_everything(1)
    b = torch.randn(16)
    assert not torch.equal(a, b)


def test_select_device_cpu():
    assert select_device("cpu").type == "cpu"


def test_smoke_two_runs_identical(tmp_path):
    """The core M0 reproducibility gate: same seed -> identical results."""
    s1 = run_smoke(seed=0, out_dir=tmp_path / "run1", steps=20, device="cpu")
    s2 = run_smoke(seed=0, out_dir=tmp_path / "run2", steps=20, device="cpu")

    assert s1["final_loss"] == s2["final_loss"]
    assert s1["x_final_norm"] == s2["x_final_norm"]

    # And the logged metric streams must match line-for-line. Read the files
    # directly (re-instantiating JsonlLogger would truncate them).
    lines1 = (tmp_path / "run1" / "metrics.jsonl").read_text().splitlines()
    lines2 = (tmp_path / "run2" / "metrics.jsonl").read_text().splitlines()
    assert len(lines1) == len(lines2) == 20
    for l1, l2 in zip(lines1, lines2):
        r1, r2 = json.loads(l1), json.loads(l2)
        # wall_time_s is allowed to differ; everything else must match.
        r1.pop("wall_time_s"), r2.pop("wall_time_s")
        assert r1 == r2


def test_smoke_different_seeds_differ(tmp_path):
    s1 = run_smoke(seed=0, out_dir=tmp_path / "a", steps=20, device="cpu")
    s2 = run_smoke(seed=7, out_dir=tmp_path / "b", steps=20, device="cpu")
    assert s1["final_loss"] != s2["final_loss"]


def test_deep_merge_override_wins():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 99, "z": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}}
    # inputs unchanged
    assert base["nested"]["y"] == 2


def test_dotted_overrides():
    cfg = {"solver": {"steps": 50}}
    out = apply_dotted_overrides(cfg, {"solver.steps": 20, "guidance.lambda0": 1.0})
    assert out["solver"]["steps"] == 20
    assert out["guidance"]["lambda0"] == 1.0


def test_save_resolved_config_roundtrip(tmp_path):
    import yaml

    cfg = {"experiment": {"seed": 0}, "flow": {"path": "linear"}}
    p = save_resolved_config(cfg, tmp_path)
    assert p.exists()
    loaded = yaml.safe_load(p.read_text())
    assert loaded == cfg
