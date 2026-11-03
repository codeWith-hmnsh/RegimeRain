"""`regimerain serve` (BACKEND_BUILD_PLAN B2): read-only API over exported runs, plus the web app."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from regimerain.api.app import create_app  # noqa: E402
from regimerain.config import load_config  # noqa: E402
from regimerain.runs.export_web import export_run  # noqa: E402
from regimerain.runs.mock import mock_fields, mock_report  # noqa: E402


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    cfg = load_config()
    d = tmp_path_factory.mktemp("runs")
    export_run(mock_fields(cfg), cfg, d, report=mock_report(cfg))
    return d


@pytest.fixture()
def dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>RegimeRain</title>", encoding="utf-8")
    (d / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return d


def test_health_and_run_listing(runs_dir, dist):
    c = TestClient(create_app(runs_dir, dist))
    h = c.get("/api/health").json()
    assert h["ok"] and h["n_runs"] == 1
    runs = c.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["kind"] == "mock" and runs[0]["synthetic"] is True
    latest = c.get("/api/runs/latest").json()
    assert latest["run_id"] == runs[0]["run_id"] and latest["manifest"]["contract_version"] == 2


def test_export_files_are_served_exactly(runs_dir, dist):
    c = TestClient(create_app(runs_dir, dist))
    run_id = c.get("/api/runs").json()[0]["run_id"]
    for name in ("manifest", "grid", "places", "verification"):
        r = c.get(f"/api/runs/{run_id}/{name}.json")
        assert r.status_code == 200
        assert r.content == (runs_dir / run_id / "web" / f"{name}.json").read_bytes()      # bytes on disk, no recompute
        assert "immutable" in r.headers["cache-control"] and r.headers.get("etag")
    assert c.get(f"/run/{run_id}/grid.json").status_code == 200                           # path the web app reads
    assert c.get(f"/api/runs/{run_id}/grid.json", headers={"Accept-Encoding": "gzip"}).headers.get("content-encoding") == "gzip"


def test_unknown_things_are_404(runs_dir, dist):
    c = TestClient(create_app(runs_dir, dist))
    run_id = c.get("/api/runs").json()[0]["run_id"]
    assert c.get("/api/runs/nope/grid.json").status_code == 404
    assert c.get(f"/api/runs/{run_id}/secrets.json").status_code == 404
    assert c.get(f"/api/runs/{run_id}/qm_curves.json").status_code == 404                 # not exported for this run
    assert c.get("/api/nothing").status_code == 404


def test_paths_cannot_escape_the_run_folders(runs_dir, dist, tmp_path):
    (tmp_path / "private.json").write_text(json.dumps({"secret": 1}))
    c = TestClient(create_app(runs_dir, dist))
    for url in ("/api/runs/..%2F..%2Fprivate/manifest.json", "/run/..%2Fprivate/manifest.json",
                "/api/runs/%2E%2E/manifest.json", "/..%2Fprivate.json"):
        r = c.get(url)
        assert r.status_code in (404, 200) and b"secret" not in r.content, url


def test_web_app_and_spa_fallback(runs_dir, dist):
    c = TestClient(create_app(runs_dir, dist))
    assert c.get("/assets/app.js").text == "console.log(1)"
    for route in ("/", "/forecast", "/scorecard?run=sample"):
        r = c.get(route)
        assert r.status_code == 200 and "RegimeRain" in r.text
    no_web = TestClient(create_app(runs_dir, dist=None))
    assert "npm run build" in no_web.get("/").json()["note"]
