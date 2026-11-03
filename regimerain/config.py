"""Configuration loading: layered YAML files, mode resolution and a settings hash.

TRD section 5 / MODEL_SPEC section 19. Later files override earlier ones key by key.
`config_sha256` covers every model setting but not `paths`, so the same experiment run on
different machines (laptop, Colab, Kaggle) hashes identically.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
MODES = ("fast", "full")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(paths: Iterable[str | Path] | None = None, data_root: str | Path | None = None) -> dict:
    """Load and merge YAML configs (default.yaml when `paths` is empty).

    `data_root`, if given, re-roots every relative entry under `paths`.
    """
    files = [Path(p) for p in (paths or [])] or [DEFAULT_CONFIG]
    cfg: dict = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if cfg.get("mode") not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.get('mode')!r}")
    if data_root is not None:
        root = Path(data_root)
        cfg["paths"] = {k: str(v if Path(v).is_absolute() else root / v) for k, v in cfg.get("paths", {}).items()}
    return cfg


def by_mode(cfg: dict, value: Any) -> Any:
    """Resolve a `{fast: x, full: y}` setting for the active mode; pass anything else through."""
    if isinstance(value, dict) and set(value) <= set(MODES):
        return value[cfg["mode"]]
    return value


def config_sha256(cfg: dict) -> str:
    settings = {k: v for k, v in cfg.items() if k != "paths"}
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def lgb_params(cfg: dict, params: dict) -> dict:
    """Copy of LightGBM params with the run-wide seed and thread count injected (NFR-1)."""
    p = dict(params)
    seed = int(cfg["seed"])
    p.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed,
             num_threads=int(cfg["num_threads"]), deterministic=True)
    p.setdefault("verbose", -1)
    return p
