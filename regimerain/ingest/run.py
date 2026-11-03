"""`regimerain ingest` orchestration (TRD 4.1). Every part is idempotent and can be run alone with --only."""
from __future__ import annotations

import json
from pathlib import Path

PARTS = ("tracks", "mjo", "imd", "hres")
MIN_CLIM_YEARS = 25


def run_ingest(cfg: dict, years: list[int], truth: str | None = None, only: list[str] | None = None,
               force: bool = False, log=print) -> dict:
    parts = [p for p in PARTS if p in (only or PARTS)]
    unknown = set(only or []) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown ingest parts {sorted(unknown)}; choose from {PARTS}")
    truth = truth or cfg["sources"]["truth"]
    raw = Path(cfg["paths"]["raw"])
    done: dict = {}

    if "tracks" in parts:
        from regimerain.ingest.tracks import IBTRACS_NI, parse_ibtracs
        out = raw / "tracks" / "ibtracs_ni.parquet"
        if force or not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            df = parse_ibtracs(IBTRACS_NI, years=(min(years) - 1, max(years)))
            df.to_parquet(out, index=False)
            log(f"tracks: {len(df)} fixes -> {out}")
        done["tracks"] = str(out)

    if "mjo" in parts:
        from regimerain.ingest.mjo import fetch_text, parse
        out = raw / "mjo" / "rmm.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        text, source = fetch_text(cfg["sources"].get("rmm_file"))       # always refresh: the index updates daily
        df = parse(text)
        df.to_parquet(out, index=False)
        (out.with_suffix(".json")).write_text(json.dumps({"source": source, "last_date": str(df.date.max().date())}))
        log(f"mjo: {len(df)} days up to {df.date.max().date()} from {source} -> {out}")
        done["mjo"] = str(out)

    if "imd" in parts:
        if truth != "imd":
            raise NotImplementedError("CHIRPS truth ingest not built yet (MODEL_SPEC 4.4)")
        from regimerain.ingest import imd
        failed = imd.download(cfg["paths"]["imd"], years, log=log)
        if failed:
            raise RuntimeError(f"IMD download failed for season years {failed}; rerun `ingest --only imd`")
        done["imd"] = cfg["paths"]["imd"]

    if "hres" in parts:
        from regimerain.ingest.hres import ingest_season, open_hres
        ds = open_hres(cfg["sources"]["hres_url"])
        leads = tuple(cfg["leads"])
        done["hres"] = {y: {k: str(v) for k, v in ingest_season(ds, y, cfg, leads, force, log).items()} for y in years}
    return done


def run_imd_climatology(cfg: dict, start: int, end: int, log=print) -> None:
    from regimerain.ingest import imd
    failed = imd.download(cfg["paths"]["imd"], range(start, end + 1), log=log)
    if failed:
        log(f"imd climatology: {len(failed)} year(s) missing {failed}; labels will use the remaining years "
            f"(allowed while at least {MIN_CLIM_YEARS} years are present)")
