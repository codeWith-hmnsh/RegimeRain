"""Error regimes (MODEL_SPEC 10): descriptors with known answers, clustering that finds planted groups."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from regimerain.errreg.errreg import DESCRIPTORS, assign, day_descriptors, fit_error_regimes, name_clusters  # noqa: E402


def one_day(f, o, lat, lon, zone=0, init="2020-07-01"):
    n = len(f)
    return pd.DataFrame({"init": pd.Timestamp(init), "lead": 1, "zone": zone, "lat": lat, "lon": lon,
                         "f_rain": np.asarray(f, float), "o_rain": np.asarray(o, float)}, index=range(n))


def test_descriptors_known_values():
    lat = np.array([10.0, 11.0, 12.0, 13.0]); lon = np.full(4, 75.0)
    o = np.array([0.0, 100.0, 0.0, 0.0])                       # heavy rain observed at 11 N
    f = np.array([0.0, 0.0, 0.0, 200.0])                       # forecast puts twice as much at 13 N
    d = day_descriptors(one_day(f, o, lat, lon), n_zones=1).iloc[0]
    assert d["bias_z0"] == pytest.approx((200 - 100) / 101)
    assert d["larea_z0"] == pytest.approx(0.0)                 # one heavy cell each
    assert d["disp_north_km_z0"] == pytest.approx(2 * 111.2)
    assert d["disp_km_z0"] == pytest.approx(222.4, rel=0.01)
    assert d["corr_z0"] < 0                                    # wrong place: anti-correlated pattern
    same = day_descriptors(one_day(o, o, lat, lon), n_zones=1).iloc[0]
    assert same["bias_z0"] == 0 and same["disp_km_z0"] == 0 and same["corr_z0"] == pytest.approx(1.0)


def test_missing_zone_is_zero_not_nan():
    lat = np.array([10.0, 11.0]); lon = np.full(2, 75.0)
    d = day_descriptors(one_day([1, 2], [2, 1], lat, lon, zone=2), n_zones=4)
    assert list(d.columns[:6]) == [f"{x}_z0" for x in DESCRIPTORS] and not d.isna().any().any()
    assert (d.filter(like="_z0") == 0).all().all()


def test_clustering_recovers_planted_groups():
    rng = np.random.default_rng(1)
    centres = np.array([[3, 0, 0, 0], [-3, 0, 0, 0], [0, 3, 0, 0], [0, -3, 0, 0], [0, 0, 3, 0]], float)
    truth = np.repeat(np.arange(5), 60)
    E = centres[truth] + rng.normal(0, 0.4, (len(truth), 4))
    m = fit_error_regimes(E, k_range=(4, 7), seed=0, n_boot=5)
    assert m["k"] == 5 and m["stability"] > 0.9
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(truth, m["labels"]) > 0.95
    assert (assign(m, E[:10]) == m["labels"][:10]).all()       # new days go to the nearest learned cluster


def test_cluster_names_follow_the_descriptor():
    E = pd.DataFrame({"bias_z1": [5.0, 5.0, -1, -1, -1, -1], "lint_z1": [0, 0, 0, 0, -4.0, -4.0]})
    names = name_clusters(E, np.array([0, 0, 1, 1, 2, 2]), 3)
    assert names[0] == "too wet (central)" and names[2] == "extremes too weak (central)"
