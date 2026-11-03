"""Run producers and the web contract v2 export (docs/BACKEND_BUILD_PLAN.md section 8)."""
import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from regimerain import cli
from regimerain.config import load_config
from regimerain.features.schema import SYNOPTIC
from regimerain.runs import contract as C
from regimerain.runs.derive import depression_track, wettest
from regimerain.runs.export_web import build_files, export_run, verification_from_report
from regimerain.runs.mock import MOCK_REGIME_DAYS, mock_curves, mock_fields, mock_report

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def fields(cfg):
    return mock_fields(cfg)


def _export(f, cfg, tmp_path, **kw):
    return export_run(f, cfg, tmp_path, report=mock_report(cfg), curves=mock_curves(cfg), regime_days=MOCK_REGIME_DAYS, **kw)


def test_mock_export_is_valid_v2(fields, cfg, tmp_path):
    web = _export(fields, cfg, tmp_path)
    assert C.check_folder(web) == []
    m = json.loads((web / "manifest.json").read_text())
    assert m["contract_version"] == 2 and m["synthetic"] is True and m["kind"] == "mock"
    assert m["thresholds_mm"] == {"heavy": 64.5, "very_heavy": 115.6}
    assert m["synoptic"] == list(SYNOPTIC)
    g = json.loads((web / "grid.json").read_text())
    n = g["nlat"] * g["nlon"]
    assert all(len(row) == n for row in g["layers"]["raw"])
    assert all(len(row) == n * len(SYNOPTIC) for row in g["regime_probs_pct"])
    assert (g["nlat"], g["nlon"]) == (129, 135)         # TRD 2.1 rain domain at 0.25 deg


def test_export_is_deterministic(fields, cfg, tmp_path):
    a = _export(fields, cfg, tmp_path / "a")
    b = _export(mock_fields(cfg), cfg, tmp_path / "b")   # fresh fields, same seed
    for name in C.FILES:
        ta, tb = (a / f"{name}.json").read_text(), (b / f"{name}.json").read_text()
        if name == "manifest":
            ta, tb = (re.sub(r'"created_utc":"[^"]*"', "", t) for t in (ta, tb))
        assert ta == tb, name


def test_missing_layers_are_absent_not_zero(fields, cfg, tmp_path):
    f = replace(fields, p_heavy=None, p_very_heavy=None, corrected_global=None)
    web = export_run(f, cfg, tmp_path)
    m = json.loads((web / "manifest.json").read_text())
    g = json.loads((web / "grid.json").read_text())
    p = json.loads((web / "places.json").read_text())
    assert not {"p_heavy", "p_very_heavy", "corrected_global"} & set(m["layers"])
    assert not {"p_heavy", "p_very_heavy", "corrected_global"} & set(g["layers"])
    assert "p_heavy" not in m["wettest"]
    assert all("p_heavy" not in lead for pl in p["places"] for lead in pl["leads"])
    assert not (web / "verification.json").exists() and not (web / "qm_curves.json").exists()


def test_synthetic_is_derived_from_kind(fields, cfg):
    files = build_files(replace(fields, kind="replay", provenance={"backtest_id": "20260930_abcdef12"}), cfg,
                        report=mock_report(cfg))
    assert all(getattr(m, "synthetic") is False for m in files.values())
    assert files["manifest"].run_id.endswith("_replay-20260930")


def test_producers_never_import_mock():
    """Replay and model code must not reach the synthetic generator (NFR-2)."""
    allowed = {"mock.py"}
    for py in (REPO / "regimerain").rglob("*.py"):
        if py.name in allowed or py.name == "cli.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(n.endswith("runs.mock") for n in names), f"{py} imports the mock producer"


def test_wettest_is_the_corrected_maximum(fields):
    w = wettest(fields)
    k, i, j = np.unravel_index(np.nanargmax(np.where(fields.land, fields.corrected, -np.inf)), fields.corrected.shape)
    assert (w["lead"], w["lat"], w["lon"]) == (k, float(fields.lat[i]), float(fields.lon[j]))
    assert w["corrected_mm"] == round(float(fields.corrected[k, i, j]), 1)


def test_depression_track_follows_the_low_and_skips_flat_fields(fields):
    track = depression_track(fields)
    assert len(track) == len(fields.leads)
    assert track[0]["lon"] > track[-1]["lon"]                       # the mock low moves west
    flat = replace(fields, mslp=np.full_like(fields.mslp, 1005.0))
    assert depression_track(flat) == []


def test_validate_rejects_bad_fields(fields):
    with pytest.raises(ValueError, match="sum to 1"):
        replace(fields, p_synoptic=fields.p_synoptic * 0.5).validate()
    with pytest.raises(ValueError, match="very heavy"):
        replace(fields, p_very_heavy=np.clip(fields.p_heavy + 0.1, 0, 1)).validate()


def test_verification_maps_variants_and_keeps_worse_results(cfg):
    v = verification_from_report(mock_report(cfg), {"heavy": 64.5, "very_heavy": 115.6}, 0.25, True)
    C.Verification.model_validate(v)
    assert [w["key"] for w in v["fss_windows"]] == [f"fss_{w}" for w in cfg["verify"]["fss_windows"]]
    assert v["fss_windows"][0]["km"] == 28
    pooled = [e for e in v["entries"] if e["fold"] == "pooled" and e["lead"] == v["headline_lead"] and e["threshold"] == "heavy"]
    assert {e["regime"] for e in pooled} == {None, *SYNOPTIC}
    brk = next(e for e in pooled if e["regime"] == "break")
    assert brk["corrected"]["ets"] < brk["baseline"]["ets"]          # NFR-4: a worse regime stays visible
    assert all(e["global"] is not None for e in pooled)


def test_cli_run_mock(tmp_path, capsys):
    assert cli.main(["run", "--source", "mock", "--data-root", str(tmp_path)]) == 0
    web = next((tmp_path / "runs").glob("*/web"))
    assert cli.main(["contract", "--check", str(web)]) == 0
    assert "OK: web contract v2" in capsys.readouterr().out


def test_cli_unbuilt_sources_exit_2():
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--source", "gfs"])
    assert e.value.code == 2


def test_web_types_cover_the_contract(tmp_path):
    """Every field the contract defines is named in the web's types.ts (drift check)."""
    schema = json.loads(C.write_schema(tmp_path / "s.json").read_text())
    ts = (REPO / "mvp" / "web" / "src" / "lib" / "types.ts").read_text(encoding="utf-8")
    names = set()

    def walk(node):
        if isinstance(node, dict):
            names.update(node.get("properties", {}).keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)
    missing = sorted(n for n in names if not re.search(rf"\b{re.escape(n)}\b", ts))
    assert missing == [], f"types.ts lacks: {missing}"
