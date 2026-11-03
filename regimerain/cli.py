"""Command-line interface (TRD section 9).

Usage: python -m regimerain.cli <command> [options] [--config FILE ...] [--data-root DIR]

Commands not yet implemented exit with code 2 and point to the MODEL_SPEC section to build next.
Heavy dependencies are imported inside commands so `--help` works on a bare install.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from regimerain.config import config_sha256, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

PLANNED = {
    "fit-final": "MODEL_SPEC section 17 (final fit)",
    "run --source replay": "docs/BACKEND_BUILD_PLAN.md phase B3 (replay from the backtest cache)",
    "run --source hres": "docs/BACKEND_BUILD_PLAN.md phase B4 (model set on an HRES init)",
    "run --source gfs": "docs/BACKEND_BUILD_PLAN.md phase B5 / MODEL_SPEC section 18 (live GFS run)",
    "serve": "docs/BACKEND_BUILD_PLAN.md phase B2 (FastAPI + web)",
}


class NotBuiltYet(SystemExit):
    def __init__(self, command: str):
        super().__init__(2)
        print(f"`{command}` is not implemented yet -> build {PLANNED[command]}.", file=sys.stderr)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", action="append", default=[], metavar="FILE",
                   help="YAML config; repeat to layer overrides (default: config/default.yaml)")
    p.add_argument("--data-root", default=None, help="re-root relative paths.* entries")
    p.add_argument("--seasons", default=None, help="override config seasons, e.g. 2019-2021")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="regimerain", description="SIH26080 regime-aware rainfall post-processing")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, help_):
        p = sub.add_parser(name, help=help_)
        _common(p)
        return p

    add("config", "print the merged config and its sha256")
    p = add("ingest", "download/prepare forecasts, truth, tracks, MJO")
    p.add_argument("--years", default=None)
    p.add_argument("--truth", choices=["imd", "chirps"], default=None)
    p.add_argument("--imd-climatology", default=None, help="e.g. 1981-2015: only fetch IMD years for the climatology")
    p.add_argument("--only", default=None, help="comma list of parts: tracks,mjo,imd,hres (default: all)")
    p.add_argument("--force", action="store_true", help="redo parts that already exist")
    add("static", "static layers, zones, district weights")
    add("label", "synoptic regime labels")
    p = add("features", "feature tables")
    p.add_argument("--leads", default=None, help="default: config leads")
    p.add_argument("--force", action="store_true")
    p = add("backtest", "leave-one-monsoon-out backtest")
    p.add_argument("--variants", default="raw,A,B,C")
    p.add_argument("--mode", choices=["fast", "full"], default=None)
    p.add_argument("--folds", default=None, help="comma-separated test seasons (default: all)")
    p.add_argument("--force", action="store_true", help="rerun folds that already have a DONE marker")
    add("report", "pool all DONE folds -> verification report + skill gate")
    add("fit-final", "refit all models on every season")
    p = add("run", "produce a run and export it for the web (runs/<run_id>/web/)")
    p.add_argument("--source", choices=["mock", "replay", "hres", "gfs"], default="gfs")
    p.add_argument("--init", default=None, help="init date YYYY-MM-DD (mock default 2022-07-14; gfs: today|cached)")
    p.add_argument("--seed", type=int, default=None, help="mock only")
    p.add_argument("--publish", default=None, metavar="NAME",
                   help="also copy the web export to mvp/web/public/run/NAME (static hosting, `npm run dev`)")
    p = add("publish", "copy a run's web export into the static web app")
    p.add_argument("run_id")
    p.add_argument("--as", dest="name", default=None, help="folder name under mvp/web/public/run (default: run_id)")
    p = add("contract", "web contract v2: validate an export folder or write the JSON Schema")
    p.add_argument("--check", default=None, metavar="DIR", help="folder with manifest.json, grid.json, ...")
    p.add_argument("--schema", default=None, metavar="FILE", help="write the JSON Schema here")
    p = add("selftest", "run the test suite")
    p.add_argument("--quick", action="store_true", help="only tests marked `quick`")
    add("serve", "start the API + web app")
    return ap


def parse_range(text: str) -> list[int]:
    """'2016-2018' -> [2016, 2017, 2018]; '1,3,5' -> [1, 3, 5]."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_config(cfg: dict, args) -> int:
    print(json.dumps(cfg, indent=2, default=str))
    print("config_sha256:", config_sha256(cfg))
    return 0


def cmd_selftest(cfg: dict, args) -> int:
    cmd = [sys.executable, "-m", "pytest", str(REPO_ROOT / "tests"), "-q"]
    if args.quick:
        cmd += ["-m", "quick"]
    return subprocess.call(cmd, cwd=REPO_ROOT)


def cmd_ingest(cfg: dict, args) -> int:
    from regimerain.ingest.run import run_imd_climatology, run_ingest
    if args.imd_climatology:
        a, b = parse_range(args.imd_climatology)[0], parse_range(args.imd_climatology)[-1]
        run_imd_climatology(cfg, a, b)
        return 0
    years = parse_range(args.years) if args.years else list(cfg["seasons"])
    only = [p.strip() for p in args.only.split(",")] if args.only else None
    run_ingest(cfg, years, truth=args.truth, only=only, force=args.force)
    return 0


def cmd_static(cfg: dict, args) -> int:
    from regimerain.static.basic import build_static_basic, static_path
    from regimerain.truth import load_truth, prepare_truth
    prepare_truth(cfg, cfg["seasons"])
    st = build_static_basic(load_truth(cfg, cfg["seasons"]))
    out = static_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    st.to_zarr(out, mode="w")
    print(f"static ({st.attrs['mode']}): {int(st.land.sum())} land cells -> {out}")
    return 0


def cmd_label(cfg: dict, args) -> int:
    from regimerain.label.build import build_labels
    build_labels(cfg, cfg["seasons"])
    return 0


def cmd_features(cfg: dict, args) -> int:
    from regimerain.features.build import build_features
    leads = parse_range(args.leads) if args.leads else cfg["leads"]
    build_features(cfg, cfg["seasons"], leads, force=args.force)
    return 0


def cmd_backtest(cfg: dict, args) -> int:
    from regimerain.backtest import run_backtest
    only = parse_range(args.folds) if args.folds else None
    run_backtest(cfg, [v.strip() for v in args.variants.split(",")], only=only, force=args.force)
    return 0


def cmd_report(cfg: dict, args) -> int:
    from regimerain.verify.report import build_report
    out = build_report(cfg)
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


WEB_RUNS = REPO_ROOT / "mvp" / "web" / "public" / "run"


def _publish(web_dir: Path, name: str) -> Path:
    import shutil
    dest = WEB_RUNS / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(web_dir, dest)
    return dest


def cmd_run(cfg: dict, args) -> int:
    from datetime import date
    if args.source != "mock":
        raise NotBuiltYet(f"run --source {args.source}")
    from regimerain.runs.export_web import export_run
    from regimerain.runs.mock import DEFAULT_INIT, DEFAULT_SEED, MOCK_REGIME_DAYS, mock_curves, mock_fields, mock_report
    seed = DEFAULT_SEED if args.seed is None else args.seed
    init = date.fromisoformat(args.init) if args.init else DEFAULT_INIT
    fields = mock_fields(cfg, init=init, seed=seed)
    web = export_run(fields, cfg, cfg["paths"]["runs"], report=mock_report(cfg, seed), curves=mock_curves(cfg, seed),
                     regime_days=MOCK_REGIME_DAYS)
    print(f"run {web.parent.name} (mock, SYNTHETIC) -> {web}")
    if args.publish:
        print(f"published -> {_publish(web, args.publish)}")
    return 0


def cmd_publish(cfg: dict, args) -> int:
    web = Path(cfg["paths"]["runs"]) / args.run_id / "web"
    if not (web / "manifest.json").exists():
        print(f"no web export at {web}", file=sys.stderr)
        return 1
    print(f"published -> {_publish(web, args.name or args.run_id)}")
    return 0


def cmd_contract(cfg: dict, args) -> int:
    from regimerain.runs.contract import check_folder, write_schema
    if args.schema:
        print(f"schema -> {write_schema(args.schema)}")
    if args.check:
        problems = check_folder(args.check)
        for p in problems:
            print("FAIL", p)
        print("OK: web contract v2" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.data_root)
    if getattr(args, "mode", None):
        cfg["mode"] = args.mode
    if getattr(args, "seasons", None):
        cfg["seasons"] = parse_range(args.seasons)
    handlers = {"config": cmd_config, "selftest": cmd_selftest, "ingest": cmd_ingest, "static": cmd_static,
                "label": cmd_label, "features": cmd_features, "backtest": cmd_backtest, "report": cmd_report,
                "run": cmd_run, "publish": cmd_publish, "contract": cmd_contract}
    if args.command in handlers:
        return handlers[args.command](cfg, args)
    raise NotBuiltYet(args.command)


if __name__ == "__main__":
    sys.exit(main())
