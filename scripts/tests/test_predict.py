"""Tests for predict.py: trend tags, RL guardrail, CLI validation, BABIP, YoY, Phase3 gates."""
import sys
import os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from predict import compute_trend_tags


def test_compute_trend_tags_bullpen_slump_home():
    tags = compute_trend_tags({
        "home_bullpen_era": 5.2, "away_bullpen_era": 4.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "home-bullpen-slump" in tags
    assert "away-bullpen-slump" not in tags
    assert "home-bullpen-strong" not in tags


def test_compute_trend_tags_bullpen_strong_away():
    tags = compute_trend_tags({
        "home_bullpen_era": 4.0, "away_bullpen_era": 2.8,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "away-bullpen-strong" in tags
    assert "home-bullpen-slump" not in tags


def test_compute_trend_tags_bullpen_neutral_no_tag():
    tags = compute_trend_tags({
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert not any("bullpen" in t for t in tags)


def test_compute_trend_tags_bullpen_boundary_exact_5_is_slump():
    """5.0 應計入 slump（spec: `>= 5.0`）"""
    tags = compute_trend_tags({
        "home_bullpen_era": 5.0, "away_bullpen_era": 3.0,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "home_season_rs": 4.5, "home_season_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
        "away_season_rs": 4.5, "away_season_ra": 4.5,
    })
    assert "home-bullpen-slump" in tags
    assert "away-bullpen-strong" in tags  # 3.0 應計入 strong（spec: `<= 3.0`）


# === Task 4: apply_rl_guardrail 8 fixture tests ===
from predict import apply_rl_guardrail


def _rl(**overrides):
    """Shorthand: build apply_rl_guardrail kwargs with sensible defaults.

    Plan B 2026-04-22（W1）：`user_rl_rec` / `user_rl_stars` kwargs 廢除；
    RL 全走 auto override。
    """
    base = dict(
        adj_home=5.0,
        adj_away=3.0,
        trend_tags=[],
        predicted_winner="HOME",
        home_team="Atlanta Braves",
        away_team="New York Mets",
    )
    base.update(overrides)
    return base


def test_rl1b_mid_diff_strong_tag_1star():
    """Fixture 1: LOW + diff=1.8 + home-bullpen-slump → mid-diff+strong-tag, ATL 1★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.4, adj_away=3.6,                  # diff = 1.8
        trend_tags=["home-bullpen-slump"],
    ))
    assert rec == "ATL"
    assert stars == 1
    assert ov["active"] is True
    assert ov["path"] == "mid-diff+strong-tag"
    assert ov["diff"] == 1.8
    assert ov["stars"] == 1
    assert ov["tags"] == ["home-bullpen-slump"]
    assert ov["warnings"] == []
    assert ov["thresholds"] == {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0}


def test_rl1b_big_diff_no_tag_2star():
    """Fixture 2: LOW + diff=2.3 + 無強 tag → big-diff, ATL 2★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff = 2.3
        trend_tags=[],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"
    assert ov["tags"] == []  # pure-diff 路徑，tags 為空 list


def test_rl1b_mid_diff_strong_tag_just_over_star_boundary():
    """Fixture 3: LOW + diff=2.1 + home-pitching-slump → mid-diff+strong-tag, ATL 2★"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.6, adj_away=3.5,                  # diff = 2.1
        trend_tags=["home-pitching-slump"],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "mid-diff+strong-tag"
    assert ov["stars"] == 2


def test_rl1b_diff_below_min_not_triggered():
    """Fixture 4: LOW + diff=1.4 + home-bullpen-slump → 不觸發（diff<1.5）"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.2, adj_away=3.8,                  # diff = 1.4
        trend_tags=["home-bullpen-slump"],
    ))
    assert rec == "PASS"
    # user_rl_stars is None + rec 已是 PASS → RL-2 不觸發；stars 保持 None
    assert stars is None
    assert ov["active"] is False
    assert ov["path"] is None


def test_rl1b_mid_diff_without_strong_tag_not_triggered():
    """Fixture 5: LOW + diff=1.8 + 無強 tag → 不觸發（中分差需 tag）"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=5.4, adj_away=3.6,                  # diff = 1.8
        trend_tags=["home-hot-offense"],             # 非 strong tag
    ))
    assert rec == "PASS"
    assert ov["active"] is False


def test_rl1b_auto_big_diff_triggers_without_user_input():
    """Fixture 6: diff=2.3 + home-pitching-slump → 觸發 big-diff 2★（自動路徑，無 user input）。"""
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff = 2.3
        trend_tags=["home-pitching-slump"],
    ))
    assert rec == "ATL"
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"


def test_rl1b_defensive_direction_mismatch_still_triggers():
    """Fixture 8: diff_side=HOME 但 predicted_winner=AWAY + diff=2.3
    → 仍觸發 big-diff path，HOME abbr 2★，warnings 記 pw_diff_direction_mismatch。

    Q4 defensive check 保留（spec 2026-04-21 §現況觀察）：方向不一致時記錄
    warning 但不阻擋 override（diff_side 是 adj_home/adj_away 的 ground truth）。
    """
    rec, stars, ov = apply_rl_guardrail(**_rl(
        adj_home=6.0, adj_away=3.7,                  # diff_side=HOME, diff=2.3
        predicted_winner="AWAY",                     # 刻意不一致
    ))
    assert rec == "ATL"                              # 仍按 diff_side 推 HOME abbr
    assert stars == 2
    assert ov["active"] is True
    assert ov["path"] == "big-diff"
    assert "pw_diff_direction_mismatch" in ov["warnings"]


# ============================================================================
# Plan B 2026-04-22 — W1: 廢除 --run-line-rec / --run-line-stars
# ============================================================================
import subprocess


def _predict_py_path():
    return os.path.join(os.path.dirname(__file__), "..", "predict.py")


def test_w1_run_line_rec_arg_removed():
    """W1（Plan B §4.2）：--run-line-rec argparse 已廢除，應 reject。"""
    result = subprocess.run(
        [sys.executable, _predict_py_path(), "--test", "--run-line-rec", "NYY"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout)
    assert "unrecognized arguments" in combined or "--run-line-rec" in combined


def test_w1_run_line_stars_arg_removed():
    """W1（Plan B §4.2）：--run-line-stars argparse 已廢除，應 reject。"""
    result = subprocess.run(
        [sys.executable, _predict_py_path(), "--test", "--run-line-stars", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout)
    assert "unrecognized arguments" in combined or "--run-line-stars" in combined


# ============================================================================
# Plan B 2026-04-22 — W2: --ml-rec schema validation
# ============================================================================

def test_w2_ml_rec_accepts_valid_abbr():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("NYY", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_accepts_pass():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec("PASS", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_accepts_none():
    from predict import validate_ml_rec, TEAM_ABBREV
    validate_ml_rec(None, set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_home():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("HOME", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_literal_away():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("AWAY", set(TEAM_ABBREV.values()))


def test_w2_ml_rec_rejects_bogus():
    from predict import validate_ml_rec, TEAM_ABBREV
    with pytest.raises(SystemExit):
        validate_ml_rec("ZZZ", set(TEAM_ABBREV.values()))


# ============================================================================
# Plan B 2026-04-22 — W4: --game-data path regex validation
# ============================================================================

def test_w4_game_data_valid_unix_path():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/merged.json")
    validate_game_data_path("/abs/path/to/analysis-data/2026-04-23/NYY@BOS/merged.json")


def test_w4_game_data_valid_windows_path():
    from predict import validate_game_data_path
    validate_game_data_path(r"C:\projects\analysis-data\2026-04-23\NYY@BOS\merged.json")


def test_w4_game_data_valid_doubleheader():
    from predict import validate_game_data_path
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G1/merged.json")
    validate_game_data_path("analysis-data/2026-04-23/NYY@BOS-G2/merged.json")


def test_w4_game_data_rejects_missing_date():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/NYY@BOS/merged.json")


def test_w4_game_data_rejects_wrong_filename():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("analysis-data/2026-04-23/NYY@BOS/foo.json")


def test_w4_game_data_rejects_bogus_tmp_path():
    from predict import validate_game_data_path
    with pytest.raises(SystemExit):
        validate_game_data_path("/tmp/foo.json")


# ============================================================================
# Plan B 2026-04-22 — W3: signal_adjustments allowlist warning
# ============================================================================

def test_w3_signal_adjustments_known_prefix_silent(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bullpen_il_home": 0.3, "weather_mild_hr": 0.1})
    captured = capsys.readouterr()
    assert "unknown signal key" not in captured.err.lower()


def test_w3_signal_adjustments_unknown_warns(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"zzz_totally_bogus_xyz": 0.5})
    captured = capsys.readouterr()
    assert "zzz_totally_bogus_xyz" in captured.err


def test_w3_signal_adjustments_mixed(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({
        "bullpen_il_home": 0.3,
        "zzz_bogus": 0.2,
        "park_factor_adj": 0.1,
    })
    captured = capsys.readouterr()
    assert "zzz_bogus" in captured.err
    assert "bullpen_il_home" not in captured.err
    assert "park_factor_adj" not in captured.err


def test_w3_signal_adjustments_empty_or_none(capsys):
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys(None)
    warn_unknown_signal_keys({})
    captured = capsys.readouterr()
    assert captured.err == ""


def test_w3_never_exits():
    """W3: 即便全部未知，也只警告不 exit。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"bogus1": 1, "bogus2": 2, "bogus3": 3})  # should not raise


def test_w3_team_abbr_prefix_known(capsys):
    """現有 94 個 signal key 中，team abbr prefix（如 nyy_, bal_）應被識別。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"nyy_lhp_disadvantage": 0.2, "bal_bullpen_depth": -0.1})
    captured = capsys.readouterr()
    assert "nyy_lhp_disadvantage" not in captured.err
    assert "bal_bullpen_depth" not in captured.err


def test_w3_pitcher_suffix_known(capsys):
    """pitcher 個人化 signal（如 castillo_velo_decline）應被識別。"""
    from predict import warn_unknown_signal_keys
    warn_unknown_signal_keys({"castillo_velo_decline": 0.3, "rocker_new_arsenal": 0.2})
    captured = capsys.readouterr()
    assert "castillo_velo_decline" not in captured.err
    assert "rocker_new_arsenal" not in captured.err


# ============================================================================
# Plan B 2026-04-22 — Y-new-2: close game (|adj_diff| < 0.5) cap to 1
# ============================================================================

def test_ynew2_close_game_caps_to_1():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 4.5, current_cap=5)
    assert new_cap == 1
    assert reason is not None
    assert "近身戰" in reason


def test_ynew2_wide_game_no_cap():
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.2, 6.8, current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew2_respects_tighter_existing_cap():
    """current_cap 已更緊時不回升。"""
    from predict import apply_close_game_cap
    new_cap, _ = apply_close_game_cap(4.2, 4.5, current_cap=0)
    assert new_cap == 0


def test_ynew2_boundary_0_5_not_triggered():
    """|diff| == 0.5 正好不觸發（strictly less than）。"""
    from predict import apply_close_game_cap
    new_cap, reason = apply_close_game_cap(4.0, 4.5, current_cap=5)
    assert new_cap == 5
    assert reason is None


# ============================================================================
# Plan B 2026-04-22 — Y-new-3: divergent user tag caps to 2
# ============================================================================

def test_ynew3_divergent_caps_to_2():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["divergent"], current_cap=5)
    assert new_cap == 2
    assert reason is not None
    assert "divergent" in reason


def test_ynew3_no_divergent_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap(["early-season", "weather"], current_cap=5)
    assert new_cap == 5
    assert reason is None


def test_ynew3_mixed_tags_with_divergent():
    from predict import apply_divergent_user_tag_cap
    new_cap, _ = apply_divergent_user_tag_cap(["early-season", "divergent", "bullpen-il"], current_cap=5)
    assert new_cap == 2


def test_ynew3_respects_tighter_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, _ = apply_divergent_user_tag_cap(["divergent"], current_cap=1)
    assert new_cap == 1


def test_ynew3_empty_user_tags_no_cap():
    from predict import apply_divergent_user_tag_cap
    new_cap, reason = apply_divergent_user_tag_cap([], current_cap=5)
    assert new_cap == 5
    assert reason is None


# ============================================================================
# Plan B 2026-04-22 — Y-new-1: home 2-star audit tag
# ============================================================================

def test_ynew1_home_2star_triggers_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "NYY") is True


def test_ynew1_away_2star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("AWAY", 2, "BOS") is False


def test_ynew1_home_3star_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 3, "NYY") is False


def test_ynew1_home_2star_pass_no_tag():
    """final_ml_rec == PASS → tag 無意義（推薦已消）。"""
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, "PASS") is False


def test_ynew1_home_2star_none_rec_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", 2, None) is False


def test_ynew1_home_none_stars_no_tag():
    from predict import should_add_home_2star_tag
    assert should_add_home_2star_tag("HOME", None, "NYY") is False


# ============================================================================
# Plan B 2026-04-22 — B7 YoY / B10 BABIP trigger helpers
# ============================================================================

def test_pitcher_yoy_triggered_by_era_xera_gap():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.50, "xera": 1.88, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_triggered_by_small_ip_era_drop():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": 3.80}}) is True


def test_pitcher_yoy_not_triggered_normal():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 3.80, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 3.90}}) is False


def test_pitcher_yoy_boundary_1_5_triggers():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.50, "ip": 45.0, "prior_year": {"era": 4.00}}) is True


def test_pitcher_yoy_boundary_just_under_not_triggered():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.51, "ip": 45.0, "prior_year": {"era": 4.00}}) is False


def test_pitcher_yoy_none_tolerant():
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy(None) is False
    assert pitcher_triggers_yoy({}) is False
    assert pitcher_triggers_yoy({"era": None}) is False


def test_pitcher_yoy_no_prior_year_era_gap_still_triggers():
    """era-xera gap ≥ 1.5 路徑獨立於 prior_year。"""
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 5.00, "xera": 3.40, "ip": 45.0, "prior_year": {"era": None}}) is True


def test_pitcher_yoy_no_prior_year_small_ip_not_triggered():
    """小 IP 但無 prior year 比較對象 → 不觸發 IP 路徑（era gap 獨立路徑若不符也 False）。"""
    from predict import pitcher_triggers_yoy
    assert pitcher_triggers_yoy({"era": 2.50, "xera": 2.40, "ip": 25.0, "prior_year": {"era": None}}) is False


def test_lineup_babip_low_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.250}) is True


def test_lineup_babip_high_extreme_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.380}) is True


def test_lineup_babip_normal_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.300}) is False


def test_lineup_babip_boundary_260_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.260}) is True


def test_lineup_babip_boundary_370_triggers():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": 0.370}) is True


def test_lineup_babip_none_no_trigger():
    from predict import lineup_triggers_babip
    assert lineup_triggers_babip({"recent_babip": None}) is False
    assert lineup_triggers_babip({}) is False
    assert lineup_triggers_babip(None) is False


# ============================================================================
# Plan B 2026-04-22 Tasks 5.2 / 5.3 — B7 YoY file check + phase3_summary grep
# ============================================================================

def _minimal_merged_json(
    home_era=3.8, home_xera=3.5, home_ip=50.0, home_prior_era=3.9,
    away_era=3.8, away_xera=3.5, away_ip=50.0, away_prior_era=3.9,
    home_babip=0.300, away_babip=0.300,
):
    """產最小合法 merged.json content（dict）。
    預設不觸發任何 B7/B10，測試時覆蓋 fields 來觸發。
    """
    return {
        "_meta": {
            "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "home_sp": "Max Fried", "away_sp": "Garrett Crochet",
            "home_sp_starts": 3, "away_sp_starts": 3,
            "game_date": "2026-04-23T23:05:00Z", "venue": "Yankee Stadium",
            "game_pk": 999,
        },
        "home_starter_fip": 3.50, "home_starter_k_bb": 18.0, "home_starter_whip": 1.20,
        "away_starter_fip": 3.80, "away_starter_k_bb": 22.0, "away_starter_whip": 1.15,
        "home_batting_xwoba": 0.330, "home_batting_ops": 0.780, "home_batting_k_pct": 20.0,
        "away_batting_xwoba": 0.340, "away_batting_ops": 0.800, "away_batting_k_pct": 21.0,
        "home_pitcher": {
            "era": home_era, "xera": home_xera, "ip": home_ip,
            "era_xera_delta": abs(home_era - home_xera) if (home_era and home_xera) else None,
            "prior_year": {"era": home_prior_era},
        },
        "away_pitcher": {
            "era": away_era, "xera": away_xera, "ip": away_ip,
            "era_xera_delta": abs(away_era - away_xera) if (away_era and away_xera) else None,
            "prior_year": {"era": away_prior_era},
        },
        "home_lineup": {"recent_babip": home_babip},
        "away_lineup": {"recent_babip": away_babip},
        "home_bullpen_era": 4.0, "away_bullpen_era": 4.0, "park_factor": 100,
        "home_season_games": 22, "away_season_games": 22,
        "home_recent_rs": 4.5, "home_recent_ra": 4.5,
        "away_recent_rs": 4.5, "away_recent_ra": 4.5,
    }


def _setup_game_dir(tmp_path, date="2026-04-23", matchup="NYY@BOS", **merged_overrides):
    """建 analysis-data/<date>/<matchup>/merged.json，回傳 game_dir + merged_path。"""
    game_dir = tmp_path / "analysis-data" / date / matchup
    game_dir.mkdir(parents=True)
    merged_path = game_dir / "merged.json"
    merged_path.write_text(json.dumps(_minimal_merged_json(**merged_overrides)))
    return game_dir, merged_path


def test_yoy_check_triggered_missing_file_exits(tmp_path):
    """B7 觸發（home era-xera≥1.5）+ 缺 prior year file → sys.exit with hint。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "B7" in result.stderr or "YoY" in result.stderr
    assert "pitcher_stats.py" in result.stderr


def test_yoy_check_triggered_with_file_proceeds_past_yoy(tmp_path):
    """B7 觸發 + prior year file 存在 → YoY check 通過（後續可能因 phase3 check 卡住但不在 B7）。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    # 建 prior year file
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # 要麼 returncode == 0 或 stderr 不含 B7 錯誤
    assert "B7 YoY" not in result.stderr, f"should pass B7 check; stderr: {result.stderr}"


def test_yoy_skip_flag_bypasses(tmp_path):
    """--skip-yoy-check 即便觸發 + 缺檔也不阻擋。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    # 寫一個 phase3_summary.md 讓 phase3 check 通過
    (game_dir / "phase3_summary.md").write_text("# test summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-yoy-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # 應跑到 predict.py 後半段（可能成功或因其他原因失敗，但不應是 B7）
    assert "B7 YoY" not in result.stderr


def test_yoy_no_trigger_no_check(tmp_path):
    """無 YoY 觸發 + 缺 prior year file → 也不 exit（因 trigger 不成立）。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)  # 預設不觸發
    (game_dir / "phase3_summary.md").write_text("# test summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert "B7 YoY" not in result.stderr


def test_phase3_missing_file_exits(tmp_path):
    """phase3_summary.md 不存在 → exit。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "phase3_summary" in result.stderr


def test_phase3_yoy_trigger_missing_section_exits(tmp_path):
    """B7 觸發 + phase3_summary.md 存在但缺 '## YoY 對比結論' → exit。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0
    )
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    (game_dir / "phase3_summary.md").write_text(
        "# basic summary\n\n## 投打對決\n無 YoY section\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "phase3_summary" in result.stderr
    assert "YoY 對比結論" in result.stderr


def test_phase3_babip_trigger_missing_section_exits(tmp_path):
    """B10 觸發（BABIP≤.260）+ 缺 '## BABIP 回歸判定' → exit。"""
    game_dir, merged_path = _setup_game_dir(tmp_path, home_babip=0.250)
    (game_dir / "phase3_summary.md").write_text("# basic summary\n")
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "BABIP 回歸判定" in result.stderr


def test_phase3_skip_flag_bypasses(tmp_path):
    """--skip-phase3-check 即便缺 phase3_summary.md 也放行。"""
    game_dir, merged_path = _setup_game_dir(tmp_path)
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save", "--skip-phase3-check"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # phase3 check 不會阻擋；可能 returncode 非 0 但錯誤不是 phase3_summary
    assert "phase3_summary" not in result.stderr


def test_phase3_all_sections_present_passes(tmp_path):
    """所有必要 section 都在 → phase3 check 通過。"""
    game_dir, merged_path = _setup_game_dir(
        tmp_path, home_era=5.0, home_xera=3.0, home_ip=45.0, home_prior_era=4.0,
        home_babip=0.250,
    )
    (game_dir / "home_pitcher_2025.json").write_text(json.dumps({"season": {"era": 4.0}}))
    (game_dir / "phase3_summary.md").write_text(
        "# summary\n\n## YoY 對比結論\n OK\n\n## BABIP 回歸判定\n OK\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, _predict_py_path(),
         "--game-data", str(merged_path), "--save"],
        capture_output=True, text=True, encoding="utf-8",
    )
    # phase3 check 不卡；整體可能成功
    assert "phase3_summary.md 缺必要 section" not in result.stderr


# ============================================================================
# Task 1: _format_pct_with_flip
# ============================================================================

def test_format_pct_no_adjusted_home_winner():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "HOME", 0.0, 0.0, has_adjusted=False)
    assert result == "Formula log5: **51.9% (HOME)**"


def test_format_pct_no_adjusted_away_winner():
    """pct 永遠是 home 勝率；side label 為 HOME (主隊勝率視角)"""
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(44.2, "AWAY", 0.0, 0.0, has_adjusted=False)
    assert result == "Formula log5: **44.2% (HOME)**"


def test_format_pct_adjusted_no_flip():
    """adjusted 比分仍與 formula 同方向 → 維持 formula log5 顯示"""
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "HOME", 5.0, 4.0, has_adjusted=True)
    assert result == "Formula log5: **51.9% (HOME)**"


def test_format_pct_adjusted_flip_home_to_away():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(51.9, "AWAY", 4.4, 4.85, has_adjusted=True)
    assert result.startswith("⚠️")
    assert "51.9% (HOME)" in result
    assert "4.4 < 4.85" in result
    assert "AWAY 勝" in result
    assert "pct 未隨翻轉重算" in result


def test_format_pct_adjusted_flip_away_to_home():
    from predict import _format_pct_with_flip
    result = _format_pct_with_flip(44.2, "HOME", 5.0, 4.0, has_adjusted=True)
    assert result.startswith("⚠️")
    assert "44.2% (HOME)" in result
    assert "5.0 > 4.0" in result
    assert "HOME 勝" in result


def test_format_signal_table_both_populated():
    from predict import format_signal_table_md
    auto = [
        {"signal": "Park Factor 106（修正 +0.30）", "run_value": 0.30},
        {"signal": "雙方先發 FIP ≤ 3.0（Ace 級）", "run_value": -1.0},
    ]
    user = {"bullpen_il_away": 0.5, "pitcher_yoy_home": 0.3}
    result = format_signal_table_md(auto, user)
    assert "### Auto signals" in result
    assert "### User-supplied signals" in result
    assert "Park Factor 106" in result
    assert "+0.30" in result
    assert "-1.00" in result
    assert "`bullpen_il_away`" in result
    # auto 總和 = -0.70；user 總和 = +0.80
    assert "**-0.70**" in result
    assert "**+0.80**" in result


def test_format_signal_table_auto_empty():
    from predict import format_signal_table_md
    result = format_signal_table_md([], {"foo": 0.5})
    assert "### Auto signals" in result
    # auto empty → 「（無）」一行
    assert "（無）" in result
    assert "`foo`" in result


def test_format_signal_table_user_empty():
    from predict import format_signal_table_md
    auto = [{"signal": "X", "run_value": 0.5}]
    result = format_signal_table_md(auto, {})
    assert "X" in result
    assert "（無）" in result


def test_format_signal_table_both_empty():
    from predict import format_signal_table_md
    result = format_signal_table_md([], {})
    # 兩段都顯示「（無）」
    assert result.count("（無）") == 2


def _make_minimal_record(**overrides):
    """Phase 4 test record factory."""
    base = {
        "date": "2026-04-26",
        "home_team": "Kansas City Royals",
        "away_team": "Los Angeles Angels",
        "predicted_winner": "HOME",
        "predicted_home_pct": 51.9,
        "predicted_home_score": 3.1,
        "predicted_away_score": 2.6,
        "formula_home_score": 3.1,
        "formula_away_score": 2.6,
        "adjusted_total": 5.7,
        "signal_adjustments": {},
        "ou_line": 9.0,
        "ou_rec": "UNDER",
        "ou_stars": 2,
        "ml_rec": "KC",
        "ml_stars": 2,
        "original_ml_stars": 2,
        "run_line_rec": "PASS",
        "run_line_stars": None,
        "rl_override": {
            "active": False, "path": None, "diff": None, "stars": None,
            "tags": None, "warnings": None, "thresholds": None,
        },
        "tags": [],
        "temperature_f": None,
        "wind_mph": None,
        "wind_direction": None,
        "umpire_name": None,
        "umpire_ou_rate": None,
    }
    base.update(overrides)
    return base


def test_format_recommendation_rows_full_pass():
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        ml_rec="PASS", ml_stars=None,
        ou_rec="PASS", ou_stars=None,
        run_line_rec="PASS",
    )
    tldr, full = format_recommendation_rows(record, [])
    # TL;DR 表頭存在
    assert "| 市場 |" in tldr
    # 三行都 PASS
    assert tldr.count("PASS") >= 3
    assert full.count("PASS") >= 3


def test_format_recommendation_rows_ml_with_audit_tag():
    from predict import format_recommendation_rows
    record = _make_minimal_record(tags=["home-2star-risk"])
    tldr, full = format_recommendation_rows(record, [])
    assert "audit" in tldr
    assert "`home-2star-risk`" in tldr
    assert "audit" in full
    # ml direction = KC
    assert "| ML | KC |" in tldr or "| KC |" in tldr


def test_format_recommendation_rows_ml_cap_appended_in_full():
    """ml_stars < original_ml_stars → full rows 附加降級原因"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(ml_stars=2, original_ml_stars=4)
    cap_reasons = ["formula 勝率 51.9%（50-55%）上限 2"]
    _tldr, full = format_recommendation_rows(record, cap_reasons)
    assert "原" in full or "降為" in full or "上限" in full
    # cap reason text 必須出現
    assert "50-55%" in full


def test_format_recommendation_rows_rl_inactive():
    """rl_override.active=False → reason 提及 RL_DIFF_MIN"""
    from predict import format_recommendation_rows
    record = _make_minimal_record()
    tldr, full = format_recommendation_rows(record, [])
    assert "RL_DIFF_MIN" in tldr or "1.5" in tldr
    assert "PASS" in tldr


def test_format_recommendation_rows_rl_active_big_diff():
    """rl_override.active=True big-diff → reason 含 path + diff + tags"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        run_line_rec="LAA",
        run_line_stars=2,
        predicted_home_score=2.0,
        predicted_away_score=4.6,
        rl_override={
            "active": True,
            "path": "big-diff",
            "diff": 2.6,
            "stars": 2,
            "tags": ["home-bullpen-slump", "home-pitching-slump"],
            "warnings": [],
            "thresholds": {"diff_min": 1.5, "diff_big": 2.2, "diff_star": 2.0},
        },
    )
    tldr, full = format_recommendation_rows(record, [])
    assert "big-diff" in tldr
    assert "2.6" in tldr
    # tags 折進 RL row 一句話理由
    assert "home-bullpen-slump" in tldr or "home-pitching-slump" in tldr


def test_format_recommendation_rows_ou_pass_due_to_small_gap():
    """ou_rec=PASS 且 |adj_total - ou_line| < 1.5 → reason 提及差距"""
    from predict import format_recommendation_rows
    record = _make_minimal_record(
        ou_rec="PASS", ou_stars=None,
        adjusted_total=8.8, ou_line=9.0,
    )
    tldr, _full = format_recommendation_rows(record, [])
    assert "0.2" in tldr  # gap
    assert "PASS" in tldr


# ============================================================================
# Task 4: format_discipline_check
# ============================================================================

def test_format_discipline_check_all_pass():
    from predict import format_discipline_check
    record = _make_minimal_record()
    result = format_discipline_check(record)
    # 4 行（D1/D2/D3/D5；D4 已棄用）
    assert result.count("\n") == 3
    assert "✅ D1" in result
    assert "✅ D2" in result
    assert "✅ D3" in result
    assert "✅ D5" in result


def test_format_discipline_check_d1_direction_override():
    from predict import format_discipline_check
    record = _make_minimal_record(tags=["direction-override"])
    result = format_discipline_check(record)
    assert "⚠️ D1" in result
    assert "direction-override" in result


def test_format_discipline_check_d1_ml_pass():
    from predict import format_discipline_check
    record = _make_minimal_record(ml_rec="PASS", ml_stars=None)
    result = format_discipline_check(record)
    assert "✅ D1" in result
    assert "PASS" in result


def test_format_discipline_check_d3_violation():
    """ml 推主隊 + run_line 推客隊 → D3 ⚠️"""
    from predict import format_discipline_check
    record = _make_minimal_record(
        ml_rec="KC", run_line_rec="LAA", run_line_stars=2,
    )
    result = format_discipline_check(record)
    assert "⚠️ D3" in result


def test_format_discipline_check_d5_violation():
    """ou_rec=OVER 但 adj_total <= ou_line → D5 ⚠️"""
    from predict import format_discipline_check
    record = _make_minimal_record(
        ou_rec="OVER", adjusted_total=8.5, ou_line=9.0,
    )
    result = format_discipline_check(record)
    assert "⚠️ D5" in result


def test_format_discipline_check_d5_ou_pass_skipped():
    """ou_rec=PASS → D5 永遠 ✅"""
    from predict import format_discipline_check
    record = _make_minimal_record(ou_rec="PASS", ou_stars=None)
    result = format_discipline_check(record)
    assert "✅ D5" in result
