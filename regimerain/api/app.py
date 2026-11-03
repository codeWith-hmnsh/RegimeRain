"""`regimerain serve`: the exported runs and the built web app on one port (BACKEND_BUILD_PLAN B2, TRD §15).

The API never computes a forecast or a metric (TRD §4.14). It lists, reads and serves the files the exporter
wrote (`runs/<run_id>/web/*.json`) and the web app built into `mvp/web/dist`.

    GET /api/health                      {ok, version, runs_dir, n_runs}
    GET /api/runs                        run summaries, newest first
    GET /api/runs/latest                 newest run, preferring model > replay > mock
    GET /api/runs/{run_id}/{file}.json   one export file (gzip, ETag, immutable caching)
    GET /run/{name}/{file}.json          the same files on the path the static web app reads; runs published
                                         into the build (mvp/web/dist/run/<name>) win over runs_dir
    GET /*                               the web app, with a single-page-app fallback to index.html
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from regimerain import __version__

FILES = ("manifest", "grid", "places", "qm_curves", "verification")
KIND_ORDER = {"model": 0, "replay": 1, "mock": 2}
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIST = REPO_ROOT / "mvp" / "web" / "dist"
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _summary(run_id: str, manifest: dict) -> dict:
    return {"run_id": run_id, "kind": manifest.get("kind"), "synthetic": manifest.get("synthetic"),
            "source": manifest.get("forecast_source"), "init": manifest.get("forecast_issue_date"),
            "created_utc": manifest.get("created_utc"), "layers": manifest.get("layers", [])}


def create_app(runs_dir: str | Path, dist: str | Path | None = DEFAULT_DIST) -> FastAPI:
    runs_dir = Path(runs_dir)
    dist = Path(dist) if dist else None
    app = FastAPI(title="RegimeRain API", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def runs() -> list[dict]:
        out = []
        for m in runs_dir.glob("*/web/manifest.json"):
            try:
                out.append(_summary(m.parent.parent.name, json.loads(m.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue                                  # a half-written export is skipped, not fatal
        return sorted(out, key=lambda r: r.get("created_utc") or "", reverse=True)

    def export_file(folder: Path, name: str) -> FileResponse:
        if name not in FILES:
            raise HTTPException(404, f"unknown file '{name}.json' (have: {', '.join(FILES)})")
        f = folder / f"{name}.json"
        if not f.is_file():
            raise HTTPException(404, f"{name}.json is not part of this run")
        return FileResponse(f, media_type="application/json", headers=IMMUTABLE)

    def check_id(run_id: str) -> str:
        if not SAFE_ID.match(run_id) or run_id in (".", ".."):
            raise HTTPException(404, "unknown run")
        return run_id

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "runs_dir": str(runs_dir), "n_runs": len(runs())}

    @app.get("/api/runs")
    def list_runs():
        return runs()

    @app.get("/api/runs/latest")
    def latest():
        rs = runs()
        if not rs:
            raise HTTPException(404, "no runs exported yet")
        best = min(rs, key=lambda r: (KIND_ORDER.get(r["kind"], 9), rs.index(r)))
        manifest = json.loads((runs_dir / best["run_id"] / "web" / "manifest.json").read_text(encoding="utf-8"))
        return {"run_id": best["run_id"], "manifest": manifest}

    @app.get("/api/runs/{run_id}/{name}.json")
    def run_file(run_id: str, name: str):
        folder = runs_dir / check_id(run_id) / "web"
        if not folder.is_dir():
            raise HTTPException(404, f"unknown run '{run_id}'")
        return export_file(folder, name)

    @app.get("/run/{run_id}/{name}.json")
    def static_run_file(run_id: str, name: str):
        check_id(run_id)
        for folder in ([dist / "run" / run_id] if dist else []) + [runs_dir / run_id / "web"]:
            if folder.is_dir():
                return export_file(folder, name)
        raise HTTPException(404, f"unknown run '{run_id}'")

    @app.get("/api/{rest:path}")
    def api_404(rest: str):
        raise HTTPException(404, f"no endpoint /api/{rest}")

    if dist and (dist / "index.html").is_file():
        root = dist.resolve()

        @app.get("/{path:path}")
        def web(path: str):
            f = (root / path).resolve()
            if path and f.is_file() and root in f.parents:
                return FileResponse(f)
            return FileResponse(root / "index.html")          # client-side routes: /forecast, /scorecard, ...
    else:
        @app.get("/")
        def no_web():
            return JSONResponse({"ok": True, "note": "web app not built: cd mvp/web && npm run build"})

    return app
