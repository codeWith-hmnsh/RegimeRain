import numpy as np
import pandas as pd
import pytest
import xarray as xr

from regimerain.ingest.align import cumulative_from_buckets, lead_hours, rainday_totals, rainday_window, to_hours
from regimerain.label.active_break import (ACTIVE, BREAK, DEPRESSION, NORMAL, active_break, circular_smooth,
                                           day_level_label, runs_at_least, synoptic_labels)
from regimerain.label.depression import depression_mask, haversine_km


def buckets(values_m):
    """6-hourly bucket accumulations (metres) ending at 6, 12, ... ; lead 0 is NaN like WeatherBench2."""
    hours = np.arange(0, 6 * (len(values_m) + 1), 6)
    data = np.concatenate([[np.nan], values_m])[None, :]
    return xr.DataArray(data, dims=("init", "prediction_timedelta"),
                        coords={"init": [np.datetime64("2019-07-01T00")],
                                "prediction_timedelta": hours.astype("timedelta64[h]")})


def test_undecoded_integer_lead_hours_work_too():
    """Recent xarray leaves prediction_timedelta as int64 + units attr (Phase 0 run 2)."""
    tp6 = buckets(np.full(22, 0.012))
    raw = tp6.assign_coords(prediction_timedelta=("prediction_timedelta",
                                                  (tp6.prediction_timedelta.values / np.timedelta64(1, "h")).astype("int64"),
                                                  {"units": "hours"}))
    a = rainday_totals(cumulative_from_buckets(to_hours(tp6)))
    b = rainday_totals(cumulative_from_buckets(to_hours(raw)))
    assert np.allclose(a.values, b.values) and np.allclose(b.values, 48.0)
    assert lead_hours(raw.prediction_timedelta)[:3].tolist() == [0.0, 6.0, 12.0]


def test_rainday_windows():
    assert rainday_window(1) == (3.0, 27.0) and rainday_window(5) == (99.0, 123.0)


def test_constant_rate_gives_exact_daily_total():
    rate_mm_h = 2.0
    tp6 = buckets(np.full(22, rate_mm_h * 6 / 1000))            # up to 132 h
    rain = rainday_totals(cumulative_from_buckets(to_hours(tp6)))
    assert np.allclose(rain.values, 24 * rate_mm_h)
    assert str(rain.valid.sel(lead=1).values.ravel()[0])[:10] == "2019-07-01"
    assert str(rain.valid.sel(lead=5).values.ravel()[0])[:10] == "2019-07-05"


def test_boundary_blocks_are_split_in_half():
    b = np.zeros(22); b[0] = 0.010; b[4] = 0.020                 # 10 mm in 0-6 h, 20 mm in 24-30 h
    rain = rainday_totals(cumulative_from_buckets(to_hours(buckets(b))), leads=(1, 2))
    assert rain.sel(lead=1).item() == pytest.approx(5 + 10)      # half of each boundary block
    assert rain.sel(lead=2).item() == pytest.approx(10)


def test_too_short_forecast_raises():
    with pytest.raises(ValueError):
        rainday_totals(cumulative_from_buckets(to_hours(buckets(np.zeros(8)))), leads=(3,))


def test_runs_at_least():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], bool)
    assert runs_at_least(m, 3).tolist() == [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]


def test_active_break_labels_and_year_boundary():
    t = pd.to_datetime(["2018-12-30", "2018-12-31", "2019-01-01", "2019-07-01", "2019-07-02", "2019-07-03",
                        "2019-07-04", "2019-07-05", "2019-07-06"])
    z = xr.DataArray([1.5, 1.5, 1.5, 1.2, 1.3, 1.1, -1.2, -1.5, -2.0], coords={"time": t}, dims="time")
    lab = active_break(z).values.tolist()
    assert lab[:3] == [NORMAL] * 3                                 # split by year boundary, then by the Jan->Jul gap
    assert lab[3:6] == [ACTIVE] * 3 and lab[6:] == [BREAK] * 3


def test_circular_smooth_preserves_length_and_constant():
    v = np.full(366, 7.0)
    assert np.allclose(circular_smooth(v), 7.0) and len(circular_smooth(v)) == 366


def test_synoptic_labels_depression_overrides():
    dep = np.zeros((2, 2, 2), bool); dep[1, 0, 0] = True
    lab = synoptic_labels(np.array([ACTIVE, BREAK]), dep)
    assert lab[0].tolist() == [[ACTIVE, ACTIVE], [ACTIVE, ACTIVE]]
    assert lab[1, 0, 0] == DEPRESSION and lab[1, 1, 1] == BREAK
    assert day_level_label(BREAK, 0.10) == DEPRESSION and day_level_label(BREAK, 0.01) == BREAK


def test_haversine_one_degree_latitude():
    assert haversine_km(20, 80, 21, 80) == pytest.approx(111.2, rel=0.01)


def test_depression_mask_radius_and_time_window():
    lat = np.arange(10, 30.01, 0.25); lon = np.arange(75, 95.01, 0.25)
    tracks = pd.DataFrame({"time": pd.to_datetime(["2019-08-10 06:00", "2019-08-11 06:00"]),
                           "lat": [20.0, 25.0], "lon": [85.0, 80.0]})
    m = depression_mask(tracks, pd.to_datetime(["2019-08-10"]), lat, lon, radius_km=500)
    iy, ix = np.searchsorted(lat, 20.0), np.searchsorted(lon, 85.0)
    assert m[0, iy, ix]
    assert not m[0, np.searchsorted(lat, 20.0), np.searchsorted(lon, 94.0)]       # ~940 km away
    assert not m[0, np.searchsorted(lat, 25.0), np.searchsorted(lon, 80.0)]       # on the 2nd fix, but it's next rain day
