"""Single source of truth for feature columns (TRD section 3.3, MODEL_SPEC section 7.5).

Features are selected by allow-list. Truth (`o_*`) and label (`y_*`) columns must never reach a model.
"""
from __future__ import annotations

STATIC = ["lat", "lon", "geo", "zone", "elev_mean", "elev_std", "slope_mean", "windward", "dist_coast_km"]
FORECAST = [
    "f_rain", "f_rain_nbr3_mean", "f_rain_nbr3_max", "f_rain_nbr5_mean", "f_rain_nbr5_max",
    "f_u850", "f_v850", "f_ws850", "f_llj_index", "f_trough_lat", "f_bob_vort_max",
    "f_vort850_max500km", "f_dist_mslp_min_km", "f_mslp_anom", "f_pw", "f_ivt", "f_imfc", "f_w500",
    "f_cmz_rain", "f_cmz_wetfrac",          # forecast's own core-monsoon-zone rain: direct active/break signal
]
OTHER = ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_phase", "doy_sin", "doy_cos", "lead"]
BASE_FEATURES = STATIC + FORECAST + OTHER
CATEGORICAL = ["geo", "zone", "mjo_phase"]

SYNOPTIC = ["active", "break", "depression", "normal"]          # y_synoptic codes 0..3
GEO = ["plains", "coastal", "orographic"]                        # geo codes 0..2
ZONES = ["northwest", "central", "south_peninsula", "east_northeast"]

REGIME_PROB_FEATURES = [f"p_{s}" for s in SYNOPTIC]
EXCEED_FEATURES = BASE_FEATURES + ["qm_rain"] + REGIME_PROB_FEATURES

INDEX_COLUMNS = ["init", "valid", "lead", "lat_idx", "lon_idx"]
TRUTH_COLUMNS = ["o_rain"]
LABEL_COLUMNS = ["y_synoptic"]
TABLE_COLUMNS = list(dict.fromkeys(INDEX_COLUMNS + BASE_FEATURES + TRUTH_COLUMNS + LABEL_COLUMNS))

FORBIDDEN_PREFIXES = ("o_", "y_")


class LeakageError(AssertionError):
    pass


def assert_no_leak(features) -> None:
    bad = [f for f in features if str(f).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise LeakageError(f"Truth/label columns in a feature set: {bad}")


def missing_table_columns(columns) -> list[str]:
    have = set(columns)
    return [c for c in TABLE_COLUMNS if c not in have]
