"""Tests for fetch_game_data summary helpers (Phase 1 context slimming)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_team_abbr_known_team_id():
    from fetch_game_data import team_abbr
    assert team_abbr(118, "Kansas City Royals") == "KC"


def test_team_abbr_team_id_priority_over_name():
    """team_id 優先於 team_name；name 不影響結果"""
    from fetch_game_data import team_abbr
    assert team_abbr(108, "Wrong Name") == "LAA"


def test_team_abbr_team_id_none_lookup_full_name():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "Los Angeles Angels") == "LAA"


def test_team_abbr_unknown_fallback():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "Unknown Team Name") == "UNK"


def test_team_abbr_empty_name_fallback():
    from fetch_game_data import team_abbr
    assert team_abbr(None, "") == ""


def test_compute_trend_arrows_offense_up_defense_worse():
    """KC 範例：RS 上升 → 攻↑；RA 上升 → 守↓"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(5.10, 6.00, 3.79, 4.54)
    assert result["off_arrow"] == "↑"
    assert result["def_arrow"] == "↓"
    assert abs(result["off_delta"] - 1.31) < 0.01
    assert abs(result["def_delta"] - 1.46) < 0.01


def test_compute_trend_arrows_offense_down_flat_defense():
    """LAA 範例：RS −0.64 → 攻↓；RA −0.29 → 守→（未達 0.5）"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.00, 4.50, 4.64, 4.79)
    assert result["off_arrow"] == "↓"
    assert result["def_arrow"] == "→"


def test_compute_trend_arrows_offense_down_defense_better():
    """RS 下降 → 攻↓；RA 下降 → 守↑"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(3.50, 3.50, 4.50, 4.50)
    assert result["off_arrow"] == "↓"
    assert result["def_arrow"] == "↑"


def test_compute_trend_arrows_threshold_exact_50():
    """Δ = 0.5 邊界值應觸發箭頭（≥ 0.5）"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(5.00, 4.50, 4.50, 5.00)
    assert result["off_arrow"] == "↑"  # +0.50
    assert result["def_arrow"] == "↑"  # RA −0.50 → 守↑


def test_compute_trend_arrows_threshold_just_below():
    """Δ = ±0.49 應為 →"""
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.99, 4.50, 4.50, 4.99)
    assert result["off_arrow"] == "→"
    assert result["def_arrow"] == "→"


def test_compute_trend_arrows_zero_delta():
    from fetch_game_data import compute_trend_arrows
    result = compute_trend_arrows(4.50, 4.50, 4.50, 4.50)
    assert result["off_arrow"] == "→"
    assert result["def_arrow"] == "→"
    assert result["off_delta"] == 0.00
    assert result["def_delta"] == 0.00


def test_detect_current_series_g3_two_prev_games():
    """前 2 場連續對 LAA → 返回 [G1, G2]，升序排列"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-24", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
        {"date": "2026-04-22", "is_home": True, "opponent": "Baltimore Orioles",
         "team_score": 6, "opp_score": 8, "is_winner": False},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert len(result) == 2
    assert result[0]["date"] == "2026-04-24"
    assert result[0]["label"] == "G1"
    assert result[1]["date"] == "2026-04-25"
    assert result[1]["label"] == "G2"


def test_detect_current_series_first_game():
    """games[0] 對手不同 → 返回空 list（本系列首戰）"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert result == []


def test_detect_current_series_empty_games():
    from fetch_game_data import detect_current_series
    result = detect_current_series([], "Los Angeles Angels", "2026-04-26")
    assert result == []


def test_detect_current_series_doubleheader():
    """同日對同對手 2 場 → label 包含 (DH-1) / (DH-2)，G 編號連續遞增"""
    from fetch_game_data import detect_current_series
    games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
        {"date": "2026-04-22", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    result = detect_current_series(games, "Los Angeles Angels", "2026-04-26")
    assert len(result) == 2
    # 兩場同日，皆有 DH 標記；G 編號連續
    labels = [g["label"] for g in result]
    assert "G1 (DH-1)" in labels
    assert "G2 (DH-2)" in labels


def test_format_streak_context_winning_streak():
    """連勝 → '連勝對手 → ABBR (MM-DD), ...'，升序排列"""
    from fetch_game_data import format_streak_context
    games = [
        {"date": "2026-04-25", "opponent": "Los Angeles Angels", "is_winner": True},
        {"date": "2026-04-24", "opponent": "Los Angeles Angels", "is_winner": True},
        {"date": "2026-04-22", "opponent": "Baltimore Orioles", "is_winner": False},
    ]
    result = format_streak_context(games, 2)
    assert result is not None
    assert "連勝對手" in result
    assert "LAA" in result
    assert "04-24" in result
    assert "04-25" in result
    # 升序：04-24 應在 04-25 前
    assert result.index("04-24") < result.index("04-25")


def test_format_streak_context_losing_streak():
    """連敗 → '連敗對手 → ABBR (MM-DD), ...'"""
    from fetch_game_data import format_streak_context
    games = [
        {"date": "2026-04-25", "opponent": "Kansas City Royals", "is_winner": False},
        {"date": "2026-04-24", "opponent": "Kansas City Royals", "is_winner": False},
        {"date": "2026-04-22", "opponent": "Toronto Blue Jays", "is_winner": False},
    ]
    result = format_streak_context(games, -3)
    assert result is not None
    assert "連敗對手" in result
    assert "KC" in result
    assert "TOR" in result
    # TOR (04-22) 應排在最前（升序）
    assert result.index("TOR") < result.index("KC")


def test_format_streak_context_streak_zero_returns_none():
    from fetch_game_data import format_streak_context
    games = [{"date": "2026-04-25", "opponent": "X", "is_winner": True}]
    assert format_streak_context(games, 0) is None


def test_format_streak_context_empty_games_returns_none():
    from fetch_game_data import format_streak_context
    assert format_streak_context([], 2) is None
