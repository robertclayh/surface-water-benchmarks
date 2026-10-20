"""YAML configs with ``_base_`` inheritance and dotted CLI overrides."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_scalar(raw)
    return cfg


_SCI = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def _parse_scalar(raw: str) -> Any:
    # PyYAML follows YAML 1.1, where "3e-4" (no dot) is a string, not a float.
    return float(raw) if _SCI.fullmatch(raw) else yaml.safe_load(raw)
