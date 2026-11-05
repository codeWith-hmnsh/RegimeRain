"""Live GFS helpers (MODEL_SPEC 18.2): rain-day totals, humidity conversion, rain-day means, fetch layout."""
import numpy as np
import pytest

from regimerain.config import load_config
from regimerain.ingest.gfs import (fetch_gfs, instant_fields, instant_hours, q_from_rh, rain_hours,
                                   rainday_from_cumulative, rainday_means)
from regimerain.ingest.hres import COLUMN_LEVELS


def test_hours_match_the_hres_rain_day():
    assert rain_hours([1, 2, 3, 4, 5]) == [3, 27, 51, 75, 99, 123]
    assert instant_hours([1, 2]) == [6, 12, 18, 24, 30, 36, 42, 48]


def test_rainday_is_difference_of_running_totals():
    cum = {h: np.full((2, 2), h * 0.5) for h in (3, 27, 51)}           # 0.5 mm per hour
    out = rainday_from_cumulative(cum, [1, 2])
    assert np.allclose(out[1], 12.0) and np.allclose(out[2], 12.0)
    cum[27] = np.full((2, 2), 0.0)                                      # a decreasing total never gives negative rain
    assert (rainday_from_cumulative(cum, [1])[1] >= 0).all()


def test_q_from_rh_saturated_and_dry():
    q = q_from_rh(np.array(100.0), np.array(300.0), 1000.0)
    assert q == pytest.approx(0.0222, abs=5e-4)                        # saturated air at 27 C, 1000 hPa
    assert q_from_rh(np.array(0.0), np.array(300.0), 1000.0) == 0.0
    prof = q_from_rh(np.full((8, 1, 1), 80.0), np.full((8, 1, 1), 280.0), COLUMN_LEVELS)
    assert prof.shape == (8, 1, 1) and prof[0, 0, 0] > prof[-1, 0, 0]  # same T and RH: more q at lower pressure


def test_instants_average_to_rain_day_means():
    def inst(value):
        lev = np.full((len(COLUMN_LEVELS), 2, 3), value)
        return instant_fields(lev * 0 + 0.01, lev, lev * 0, np.full((2, 3), 1e5), np.full((2, 3), -0.1))
    data = {h: inst(float(h)) for h in instant_hours([1])}
    m = rainday_means(data, [1])[1]
    assert set(m) == {"u850", "v850", "mslp", "w500", "pw", "ivtx", "ivty"}
    assert np.allclose(m["u850"], np.mean([6, 12, 18, 24]))


class FakeReader:
    """Constant 1 mm/h rain, uniform wind: the shape of what Gfs returns, without the network."""

    def cumulative(self, h, lat, lon):
        return np.full((len(lat), len(lon)), float(h))

    def instant(self, h, lat, lon):
        lev = np.ones((len(COLUMN_LEVELS), len(lat), len(lon)))
        return instant_fields(lev * 0.01, lev * 10, lev * 0, np.full((len(lat), len(lon)), 1e5), np.zeros((len(lat), len(lon))))


def test_fetch_gfs_layout_on_working_grids():
    cfg = load_config()
    out = fetch_gfs(cfg, "2026-07-14", [1, 2], reader=FakeReader(), log=lambda *_: None)
    assert out["lat"][0] == cfg["grid"]["rain_box"][0] and np.all(np.diff(out["lat"]) > 0)
    assert out["rain"][1].shape == (len(out["lat"]), len(out["lon"])) and np.allclose(out["rain"][2], 24.0)
    assert out["dyn"][1]["u850"].shape == (len(out["latd"]), len(out["lond"]))
    assert np.allclose(out["dyn"][2]["u850"], 10.0)


def test_season_note_flags_inits_outside_the_monsoon():
    from datetime import date
    from regimerain.runs.live import season_note
    assert season_note(date(2026, 7, 15)) == ""
    assert "OUTSIDE THE MONSOON" in season_note(date(2026, 11, 5))
