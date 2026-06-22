"""Configuration loading with resolved-config saving.

Spec requirement (M0): load a YAML config, merge with defaults and CLI-style
overrides, and save the fully *resolved* config alongside results so every run
is reproducible from its own output directory.

We keep configs as plain nested dicts rather than a heavy schema library: the
pilot configs are small, and the spec's notation contract is enforced in code,
not in the config layer.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level config in {path} must be a mapping, got {type(data)}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins).

    Returns a new dict; inputs are not mutated. Nested mappings merge key-by-key;
    any non-mapping value (including lists) is replaced wholesale.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_dotted_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ``{"a.b.c": value}`` style overrides onto a nested config.

    Useful for CLI overrides like ``--set solver.steps=20``. Creates intermediate
    dicts as needed.
    """
    result = copy.deepcopy(dict(config))
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node: dict[str, Any] = result
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = value
    return result


def save_resolved_config(config: Mapping[str, Any], out_dir: str | Path) -> Path:
    """Write the resolved config to ``<out_dir>/resolved_config.yaml``.

    Returns the path written. Creates ``out_dir`` if needed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resolved_config.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(dict(config), f, sort_keys=True, default_flow_style=False)
    return out_path
