"""Quick structural checks: config, schema/leakage, folds, CLI (run by `selftest --quick`)."""
import json

import pytest

from regimerain import cli
from regimerain.config import by_mode, config_sha256, deep_merge, lgb_params, load_config
from regimerain.features.schema import (BASE_FEATURES, EXCEED_FEATURES, LeakageError, TABLE_COLUMNS,
                                        assert_no_leak, missing_table_columns)
from regimerain.folds import (SEASONS, check_poolable, crossfit_groups, is_done, lomo_folds, mark_done)

pytestmark = pytest.mark.quick


# ---- config -------------------------------------------------------------------------------------
def test_default_config_loads():
    cfg = load_config()
    assert cfg["mode"] == "fast"
    assert cfg["exceed"]["thresholds"] == [64.5, 115.6]
    assert cfg["seasons"] == SEASONS


def test_deep_merge_overrides_leaves_only():
    out = deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"c": 9}})
    assert out == {"a": {"b": 1, "c": 9}, "d": 3}


def test_config_hash_ignores_paths_but_not_settings(tmp_path):
    base = load_config()
    moved = load_config(data_root=tmp_path)
    assert config_sha256(base) == config_sha256(moved)
    changed = deep_merge(base, {"num_threads": 2})
    assert config_sha256(changed) != config_sha256(base)


def test_layered_config_file(tmp_path):
    over = tmp_path / "o.yaml"
    over.write_text("num_threads: 2\nexceed: {per_lead: true}\n")
    cfg = load_config([cli.REPO_ROOT / "config" / "default.yaml", over])
    assert cfg["num_threads"] == 2 and cfg["exceed"]["per_lead"] is True
    assert cfg["exceed"]["thresholds"] == [64.5, 115.6]


def test_by_mode_and_lgb_params():
    cfg = load_config()
    assert by_mode(cfg, cfg["classifier"]["cell_stride"]) == 2
    assert by_mode(cfg, 7) == 7
    p = lgb_params(cfg, cfg["classifier"]["params"])
    assert p["seed"] == cfg["seed"] and p["num_threads"] == cfg["num_threads"] and p["deterministic"]


# ---- schema / leakage ---------------------------------------------------------------------------
def test_feature_sets_have_no_truth_or_labels():
    assert_no_leak(BASE_FEATURES)
    assert_no_leak(EXCEED_FEATURES)


def test_leak_is_detected():
    with pytest.raises(LeakageError):
        assert_no_leak(BASE_FEATURES + ["o_rain"])
    with pytest.raises(LeakageError):
        assert_no_leak(["y_synoptic"])


def test_table_columns_contract():
    assert "o_rain" in TABLE_COLUMNS and "y_synoptic" in TABLE_COLUMNS
    assert missing_table_columns(TABLE_COLUMNS) == []
    assert missing_table_columns(["f_rain"])  # something is missing


# ---- folds --------------------------------------------------------------------------------------
def test_lomo_folds_isolation():
    folds = list(lomo_folds())
    assert [f["test"] for f in folds] == SEASONS
    for f in folds:
        assert f["test"] not in f["train"]
        assert f["inner_val"] in f["train"] and f["inner_val"] not in f["fit"]
        assert sorted(f["fit"] + [f["inner_val"]]) == sorted(f["train"])
        assert len(f["train"]) == 6


def test_lomo_only_subset():
    assert [f["test"] for f in lomo_folds(only=[2018, 2021])] == [2018, 2021]


def test_crossfit_groups_partition():
    train = [2016, 2017, 2019, 2020, 2021, 2022]
    groups = crossfit_groups(train, 3)
    assert sorted(s for g in groups for s in g) == train and len(groups) == 3


def test_check_poolable(tmp_path):
    with pytest.raises(RuntimeError, match="not done"):
        check_poolable(tmp_path)
    for s in SEASONS:
        mark_done(tmp_path, s, {"config_sha256": "x", "git_commit": "c"})
    assert is_done(tmp_path, 2016)
    assert len(check_poolable(tmp_path)) == 7
    mark_done(tmp_path, 2019, {"config_sha256": "y", "git_commit": "c"})
    with pytest.raises(RuntimeError, match="config_sha256"):
        check_poolable(tmp_path)


# ---- CLI ----------------------------------------------------------------------------------------
def test_parse_range():
    assert cli.parse_range("2016-2018") == [2016, 2017, 2018]
    assert cli.parse_range("1,3,5") == [1, 3, 5]


def test_cli_config_prints_hash(capsys):
    assert cli.main(["config"]) == 0
    assert "config_sha256:" in capsys.readouterr().out


def test_cli_unbuilt_command_exits_2():
    with pytest.raises(SystemExit) as e:
        cli.main(["fit-final"])
    assert e.value.code == 2


def test_cli_report_refuses_missing_folds(tmp_path):
    with pytest.raises(RuntimeError, match="not done"):
        cli.main(["report", "--data-root", str(tmp_path)])
