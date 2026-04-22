"""Tests for merge_game_data Plan B 2026-04-22 §4.6 extension."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_nested_pitcher_from_pitcher_stats_output():
    """extract_pitcher_nested 從 pitcher_stats.py 輸出取 era/xera/ip + delta。"""
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {
        "season": {"era": 3.50, "xera": 2.80, "ip": 45.2, "fip": 3.10},
    }
    result = extract_pitcher_nested(pitcher_data, prior_pitcher_data=None, prefix="home")
    assert "home_pitcher" in result
    p = result["home_pitcher"]
    assert p["era"] == 3.50
    assert p["xera"] == 2.80
    assert p["ip"] == 45.2
    assert abs(p["era_xera_delta"] - 0.70) < 0.001
    assert p["prior_year"]["era"] is None


def test_nested_pitcher_with_prior_year():
    from merge_game_data import extract_pitcher_nested
    pitcher_data = {"season": {"era": 3.50, "xera": 2.80, "ip": 45.2}}
    prior = {"season": {"era": 4.20, "xera": 3.90, "ip": 180.0}}
    result = extract_pitcher_nested(pitcher_data, prior, prefix="home")
    assert result["home_pitcher"]["prior_year"]["era"] == 4.20


def test_nested_pitcher_missing_fields_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {}}, None, prefix="home")
    assert result["home_pitcher"]["era"] is None
    assert result["home_pitcher"]["xera"] is None
    assert result["home_pitcher"]["ip"] is None
    assert result["home_pitcher"]["era_xera_delta"] is None
    assert result["home_pitcher"]["prior_year"]["era"] is None


def test_nested_pitcher_season_error_treats_as_empty():
    """pitcher_stats.py 回 {season: {error: ...}} 時視為缺失。"""
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {"error": "lookup_failed"}}, None, prefix="away")
    assert result["away_pitcher"]["era"] is None


def test_nested_pitcher_none_data_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested(None, None, prefix="home")
    assert result["home_pitcher"]["era"] is None


def test_nested_pitcher_prior_error_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested(
        {"season": {"era": 3.5, "xera": 3.0, "ip": 40.0}},
        {"season": {"error": "not_found"}},
        prefix="home",
    )
    assert result["home_pitcher"]["prior_year"]["era"] is None


def test_nested_lineup_from_lineup_analyzer_output():
    from merge_game_data import extract_lineup_nested
    lineup_data = {"last7_babip": 0.320, "avg_babip": 0.290}
    result = extract_lineup_nested(lineup_data, prefix="home")
    assert result["home_lineup"]["recent_babip"] == 0.320


def test_nested_lineup_missing_last7_babip_returns_none():
    from merge_game_data import extract_lineup_nested
    result = extract_lineup_nested({"avg_babip": 0.290}, prefix="home")
    assert result["home_lineup"]["recent_babip"] is None


def test_nested_lineup_none_data_tolerant():
    from merge_game_data import extract_lineup_nested
    result = extract_lineup_nested(None, prefix="away")
    assert result["away_lineup"]["recent_babip"] is None


# ============================================================================
# Plan B 2026-04-22 Task 4.3 — --home-pitcher-prior / --away-pitcher-prior args
# ============================================================================

def test_cli_accepts_prior_pitcher_args():
    """argparse 接受 --home-pitcher-prior / --away-pitcher-prior。"""
    import subprocess
    merge_py = os.path.join(os.path.dirname(__file__), "..", "merge_game_data.py")
    # --test mode 會跳過所有實際工作，只驗 argparse 不報錯
    result = subprocess.run(
        [sys.executable, merge_py, "--test",
         "--home-pitcher-prior", "/tmp/not_exist.json",
         "--away-pitcher-prior", "/tmp/not_exist.json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "unrecognized" not in result.stderr


def test_prior_pitcher_extract_chain():
    """extract_pitcher_nested 傳 prior_data 會正確填 prior_year.era（端到端）。"""
    from merge_game_data import extract_pitcher_nested
    pitcher = {"season": {"era": 3.50, "xera": 2.80, "ip": 45.0}}
    prior = {"season": {"era": 4.20, "xera": 3.90, "ip": 180.0}}
    result = extract_pitcher_nested(pitcher, prior, "home")
    assert result["home_pitcher"]["era"] == 3.50
    assert result["home_pitcher"]["prior_year"]["era"] == 4.20


def test_prior_pitcher_none_no_prior():
    """無 prior data → prior_year.era = None（正常情況，非 bug）。"""
    from merge_game_data import extract_pitcher_nested
    pitcher = {"season": {"era": 3.50, "xera": 2.80, "ip": 45.0}}
    result = extract_pitcher_nested(pitcher, None, "home")
    assert result["home_pitcher"]["prior_year"]["era"] is None
