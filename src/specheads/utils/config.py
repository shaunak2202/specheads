"""YAML run configs.

Kaggle notebooks are thin launchers, so a run is fully described by its config
file. Each result directory gets a verbatim copy of the config it ran under.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config is missing required keys or has the wrong shape."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional `extends` parent.

    `extends` keeps the sweep configs from restating the whole benchmark
    protocol, which is frozen after Phase 1 and must not drift between runs.
    Only one level is resolved: deeper chains make it too easy to lose track of
    what a run actually used.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")

    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a mapping, got {type(config).__name__}")

    parent_name = config.pop("extends", None)
    if parent_name is None:
        return config

    parent_path = (path.parent / parent_name).resolve()
    with parent_path.open() as handle:
        parent = yaml.safe_load(handle) or {}
    if "extends" in parent:
        raise ConfigError(f"nested extends not supported: {parent_path}")

    return deep_merge(parent, config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def require(config: dict[str, Any], *keys: str) -> None:
    """Assert that dotted keys are present, naming every one that is missing."""
    missing = []
    for key in keys:
        node: Any = config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(key)
                break
            node = node[part]
    if missing:
        raise ConfigError(f"config missing required key(s): {', '.join(missing)}")
