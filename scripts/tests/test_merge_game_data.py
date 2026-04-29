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
    result = extract_pitcher_nested(pitcher_data, prefix="home")
    assert "home_pitcher" in result
    p = result["home_pitcher"]
    assert p["era"] == 3.50
    assert p["xera"] == 2.80
    assert p["ip"] == 45.2
    assert abs(p["era_xera_delta"] - 0.70) < 0.001
    assert "prior_year" not in p


def test_nested_pitcher_missing_fields_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {}}, prefix="home")
    assert result["home_pitcher"]["era"] is None
    assert result["home_pitcher"]["xera"] is None
    assert result["home_pitcher"]["ip"] is None
    assert result["home_pitcher"]["era_xera_delta"] is None


def test_nested_pitcher_season_error_treats_as_empty():
    """pitcher_stats.py 回 {season: {error: ...}} 時視為缺失。"""
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested({"season": {"error": "lookup_failed"}}, prefix="away")
    assert result["away_pitcher"]["era"] is None


def test_nested_pitcher_none_data_tolerant():
    from merge_game_data import extract_pitcher_nested
    result = extract_pitcher_nested(None, prefix="home")
    assert result["home_pitcher"]["era"] is None


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
# 2026-04-26 — Park Factor JSON 化 + alias 解析（spec §7.1.2）
# ============================================================================

def test_resolve_park_factor_canonical_name():
    """正式球場名直接命中 JSON 表。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Coors Field") == 131.0
    assert resolve_park_factor("T-Mobile Park") == 82.0


def test_resolve_park_factor_alias_old_name():
    """舊球場名透過 alias 解析到新名 — 向後相容。"""
    from merge_game_data import resolve_park_factor
    # Tropicana → Steinbrenner（Rays 臨時主場）
    assert resolve_park_factor("Tropicana Field") == 100.0
    # Oakland Coliseum → Sutter Health Park
    assert resolve_park_factor("Oakland Coliseum") == 109.0
    # Minute Maid → Daikin
    assert resolve_park_factor("Minute Maid Park") == 98.0
    # Dodger Stadium → UNIQLO Field at Dodger Stadium
    assert resolve_park_factor("Dodger Stadium") == 98.0
    # Guaranteed Rate → Rate Field
    assert resolve_park_factor("Guaranteed Rate Field") == 97.0
    # Camden Yards → Oriole Park at Camden Yards
    assert resolve_park_factor("Camden Yards") == 96.0


def test_resolve_park_factor_unknown_returns_default():
    """未知球場名回傳 100.0（聯盟平均，安全 fallback）。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor("Nonexistent Stadium") == 100.0


def test_resolve_park_factor_none_returns_default():
    """None venue 回傳 100.0。"""
    from merge_game_data import resolve_park_factor
    assert resolve_park_factor(None) == 100.0


def test_resolve_park_factor_returns_float():
    """回傳型別必為 float（predict.py 後續做 PF / 100 浮點除法）。"""
    from merge_game_data import resolve_park_factor
    result = resolve_park_factor("Coors Field")
    assert isinstance(result, float)
