"""Leave-one-monsoon-out folds and fold-completion markers (MODEL_SPEC section 8, TRD section 9)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
DONE = "DONE"


def lomo_folds(seasons: list[int] = SEASONS, only: list[int] | None = None) -> Iterator[dict]:
    """Yield one fold per test season.

    inner_val is the season just before the test season (cyclic); `fit` = train minus inner_val.
    """
    seasons = list(seasons)
    for i, test in enumerate(seasons):
        if only is not None and test not in only:
            continue
        train = [s for s in seasons if s != test]
        inner_val = seasons[i - 1] if i > 0 else seasons[-1]
        fit = [s for s in train if s != inner_val]
        yield {"test": test, "train": train, "inner_val": inner_val, "fit": fit}


def crossfit_groups(train: list[int], n_groups: int) -> list[list[int]]:
    """Split training seasons into groups for out-of-fold predictions."""
    n_groups = max(1, min(n_groups, len(train)))
    return [train[i::n_groups] for i in range(n_groups)]


def fold_dir(cache: str | Path, test: int) -> Path:
    return Path(cache) / f"fold={test}"


def mark_done(cache: str | Path, test: int, info: dict) -> None:
    """Write the DONE marker last, after all fold artefacts exist."""
    d = fold_dir(cache, test)
    d.mkdir(parents=True, exist_ok=True)
    (d / DONE).write_text(json.dumps(info, indent=2, sort_keys=True))


def is_done(cache: str | Path, test: int) -> bool:
    return (fold_dir(cache, test) / DONE).exists()


def done_info(cache: str | Path) -> dict[int, dict]:
    out = {}
    for marker in Path(cache).glob(f"fold=*/{DONE}"):
        out[int(marker.parent.name.split("=")[1])] = json.loads(marker.read_text() or "{}")
    return dict(sorted(out.items()))


def check_poolable(cache: str | Path, seasons: list[int] = SEASONS) -> dict[int, dict]:
    """Raise unless every season has a DONE fold and all folds share one config hash and commit."""
    info = done_info(cache)
    missing = [s for s in seasons if s not in info]
    if missing:
        raise RuntimeError(f"Folds not done: {missing}")
    for key in ("config_sha256", "git_commit"):
        values = {info[s].get(key) for s in seasons}
        if len(values) != 1:
            raise RuntimeError(f"Folds disagree on {key}: {values}. Rerun the odd folds with --force.")
    return info
