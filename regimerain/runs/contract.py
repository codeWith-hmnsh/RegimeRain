"""Web contract v2: the JSON files the web app reads for one run (BACKEND_BUILD_PLAN section 3).

These pydantic models are the single source of truth for the shapes. The exporter builds them, so an
export that does not match cannot be written; `regimerain contract --check` validates a folder on disk;
`write_schema` emits JSON Schema for the web side.

Regimes follow MODEL_SPEC: four synoptic classes (active, break, depression, normal) predicted per cell,
plus a static geographic class (plains, coastal, orographic). Thresholds come from config (64.5 / 115.6).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = 2
FILES = ("manifest", "grid", "places", "qm_curves", "verification")

Synoptic = Literal["active", "break", "depression", "normal"]
GeoName = Literal["plains", "coastal", "orographic"]
ThresholdName = Literal["heavy", "very_heavy"]
Kind = Literal["mock", "replay", "model"]
Layer = Literal["raw", "corrected", "corrected_global", "regime", "geo", "wind850", "p_heavy", "p_very_heavy", "truth"]

Row = list[Optional[float]]          # one lead: nlat * nlon values, row-major, south to north, null off land


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Versioned(_M):
    contract_version: Literal[2] = CONTRACT_VERSION
    synthetic: bool


# ------------------------------------------------------------------------------------ manifest
class Thresholds(_M):
    heavy: float
    very_heavy: float


class TrackPoint(_M):
    lead: int = Field(description="index into grid.leads")
    lat: float
    lon: float
    mslp_hpa: float


class Wettest(_M):
    lead: int = Field(description="index into grid.leads")
    valid_date: str
    lat: float
    lon: float
    raw_mm: float
    corrected_mm: float
    p_heavy: Optional[float] = None
    p_very_heavy: Optional[float] = None
    regime: Synoptic
    geo: GeoName


class Provenance(_M):
    model_set_id: Optional[str] = None
    model_hashes: dict[str, str] = {}
    config_sha256: Optional[str] = None
    git_commit: Optional[str] = None
    backtest_id: Optional[str] = None
    held_out_season: Optional[int] = None
    seed: Optional[int] = None
    note: Optional[str] = None


class Manifest(_Versioned):
    kind: Kind
    run_id: str
    created_utc: str
    forecast_issue_date: str
    forecast_source: str
    truth_source: Optional[str] = None
    grid_step_deg: float
    unit: Literal["mm/day"] = "mm/day"
    thresholds_mm: Thresholds
    synoptic: list[Synoptic]
    geo_classes: list[GeoName]
    layers: list[Layer]
    regime_days: Optional[dict[str, int]] = Field(None, description="training days behind each synoptic regime's curves")
    depression_track: list[TrackPoint] = []
    wettest: Wettest
    provenance: Provenance


# ------------------------------------------------------------------------------------ grid
class LeadInfo(_M):
    index: int
    day: int = Field(description="rain day L (1-based)")
    valid_date: str = Field(description="start date of the rain day (03 UTC to 03 UTC)")
    lead_hours: int = Field(description="forecast hour at the end of the rain day")
    hours: tuple[int, int]


class GridLayers(_M):
    raw: list[Row]
    corrected: list[Row]
    regime: list[list[int]] = Field(description="most probable synoptic regime index, -1 off land")
    corrected_global: Optional[list[Row]] = None
    p_heavy: Optional[list[Row]] = None
    p_very_heavy: Optional[list[Row]] = None
    truth: Optional[list[Row]] = None
    u850: Optional[list[list[float]]] = None
    v850: Optional[list[list[float]]] = None
    wind850: Optional[list[list[float]]] = None


class Grid(_Versioned):
    lat0: float
    lon0: float
    step: float
    nlat: int
    nlon: int
    row_order: Literal["south_to_north"] = "south_to_north"
    synoptic: list[Synoptic]
    geo_classes: list[GeoName]
    leads: list[LeadInfo]
    geo: list[int] = Field(description="static geo class index per cell, -1 off land")
    layers: GridLayers
    regime_probs_pct: list[list[int]] = Field(description="per lead: nlat * nlon * len(synoptic), percent")


# ------------------------------------------------------------------------------------ places
class Cell(_M):
    lat: float
    lon: float


class PlaceLead(_M):
    raw_mm: float
    corrected_mm: float
    corrected_global_mm: Optional[float] = None
    p_heavy: Optional[float] = None
    p_very_heavy: Optional[float] = None
    truth_mm: Optional[float] = None
    regime: Synoptic
    regime_probs: dict[str, float]
    qm_curve: Optional[str] = None
    qm_curve_days: Optional[int] = None


class Place(_M):
    name: str
    lat: float
    lon: float
    cell: Cell
    geo: GeoName
    leads: list[PlaceLead]


class Places(_Versioned):
    places: list[Place]


# ------------------------------------------------------------------------------------ QM curves
class QmCurve(_M):
    id: str
    variant: Literal["A", "B"]
    regime: Synoptic | Literal["global"]
    geo: GeoName | Literal["all"] = "all"
    lead: Optional[int] = None
    n_days: int
    w_shrink: Optional[float] = None
    quantiles: list[float]
    forecast_mm: list[float]
    truth_mm: list[float]


class QmCurves(_Versioned):
    curves: list[QmCurve]


# ------------------------------------------------------------------------------------ verification
Scores = dict[str, Optional[float]]  # rmse, ets, csi, pod, far, fss_<window>; null when undefined


class FssWindow(_M):
    key: str
    cells: int
    km: int


class Entry(_M):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    fold: str = Field(description='"pooled" or a held-out season')
    lead: int = Field(description="rain day L (1-based)")
    threshold: ThresholdName
    regime: Optional[Synoptic] = None
    baseline: Scores = Field(description="raw NWP")
    corrected: Scores = Field(description="variant B, regime-aware")
    global_: Optional[Scores] = Field(None, alias="global", description="variant A, one global curve")
    n_samples: int
    n_events: Optional[int] = None


class Delta(_M):
    variant: str
    vs: str
    lead: int
    threshold: Optional[ThresholdName] = None
    metric: str
    delta: Optional[float] = None
    ci90: Optional[tuple[Optional[float], Optional[float]]] = None
    significant: Optional[bool] = None


class ReliabilityBin(_M):
    p_forecast: float
    p_observed: float
    n: int


class Verification(_Versioned):
    backtest_id: str
    cv: str
    seasons: list[int]
    leads: list[int]
    headline_lead: int
    variants: list[str]
    fss_windows: list[FssWindow]
    entries: list[Entry]
    deltas: list[Delta] = []
    reliability: Optional[dict[ThresholdName, list[ReliabilityBin]]] = None
    classifier: Optional[dict] = None


MODELS: dict[str, type[_Versioned]] = {"manifest": Manifest, "grid": Grid, "places": Places,
                                       "qm_curves": QmCurves, "verification": Verification}


def dump(model: BaseModel) -> str:
    """Compact JSON, optional fields left out (absent = not built)."""
    return json.dumps(model.model_dump(mode="json", exclude_none=True, by_alias=True), separators=(",", ":"))


def check_folder(folder: str | Path) -> list[str]:
    """Validate a web export folder. Returns problems (empty = valid). Missing verification/qm_curves is allowed."""
    folder = Path(folder)
    problems = []
    for name, model in MODELS.items():
        f = folder / f"{name}.json"
        if not f.exists():
            if name in ("manifest", "grid", "places"):
                problems.append(f"{name}.json missing")
            continue
        try:
            model.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception as e:  # pydantic.ValidationError, json errors
            problems.append(f"{name}.json: {str(e).splitlines()[0]} ({e.__class__.__name__})")
    return problems


def write_schema(path: str | Path) -> Path:
    """JSON Schema for all five files, keyed by file name (used by the web-side type check)."""
    path = Path(path)
    schema = {"contract_version": CONTRACT_VERSION,
              "files": {name: m.model_json_schema(by_alias=True) for name, m in MODELS.items()}}
    path.write_text(json.dumps(schema, indent=1) + "\n", encoding="utf-8")
    return path
