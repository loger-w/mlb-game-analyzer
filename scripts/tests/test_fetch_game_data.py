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


def _make_minimal_result(home_games=None, away_games=None, series_prev=None):
    """測試用 result_dict 工廠"""
    return {
        "game": {
            "gamePk": 824122,
            "date": "2026-04-26T23:20:00Z",
            "status": "Preview",
            "venue": "Kauffman Stadium",
            "home": {"team": "Kansas City Royals", "team_id": 118, "probable_pitcher": "Seth Lugo"},
            "away": {"team": "Los Angeles Angels", "team_id": 108, "probable_pitcher": "Reid Detmers"},
        },
        "home_recent": {"record": "3-7", "rs_per_game": 5.10, "ra_per_game": 6.00,
                        "run_diff": -9, "streak": 2, "games": home_games or []},
        "away_recent": {"record": "3-7", "rs_per_game": 4.00, "ra_per_game": 4.50,
                        "run_diff": -5, "streak": -3, "games": away_games or []},
        "home_recent_30": {"record": "10-18", "rs_per_game": 3.79, "ra_per_game": 4.54,
                           "run_diff": -21, "streak": 2, "games": []},
        "away_recent_30": {"record": "12-16", "rs_per_game": 4.64, "ra_per_game": 4.79,
                           "run_diff": -4, "streak": -3, "games": []},
        "home_season": {"record": "10-18", "rs_per_game": 3.79, "ra_per_game": 4.54,
                        "run_diff": -21, "streak": 2, "games": []},
        "away_season": {"record": "12-16", "rs_per_game": 4.64, "ra_per_game": 4.79,
                        "run_diff": -4, "streak": -3, "games": []},
        "home_season_games_count": 28,
        "away_season_games_count": 28,
        "series_prev": series_prev,
    }


def test_format_summary_md_smoke_full_game():
    """完整 result_dict → markdown 含所有 hard sections + 標題"""
    from fetch_game_data import format_summary_md
    home_games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 12, "opp_score": 1, "is_winner": True},
        {"date": "2026-04-24", "is_home": True, "opponent": "Los Angeles Angels",
         "team_score": 6, "opp_score": 3, "is_winner": True},
    ]
    away_games = [
        {"date": "2026-04-25", "is_home": False, "opponent": "Kansas City Royals",
         "team_score": 1, "opp_score": 12, "is_winner": False},
        {"date": "2026-04-24", "is_home": False, "opponent": "Kansas City Royals",
         "team_score": 3, "opp_score": 6, "is_winner": False},
        {"date": "2026-04-22", "is_home": True, "opponent": "Toronto Blue Jays",
         "team_score": 2, "opp_score": 4, "is_winner": False},
    ]
    md = format_summary_md(_make_minimal_result(home_games, away_games))
    assert "# Game Data Summary — LAA @ KC (2026-04-26)" in md
    assert "## 比賽資訊" in md
    assert "## 戰績摘要" in md
    assert "## 趨勢" in md
    assert "## 當前系列賽" in md
    assert "## Streak 脈絡" in md
    assert "Reid Detmers" in md
    assert "Seth Lugo" in md
    # 系列累計：KC 2-0 LAA
    assert "KC 2-0 LAA" in md or "**KC 2-0 LAA**" in md


def test_format_summary_md_first_game_of_series():
    """無前場 → 系列賽 section 顯示「本系列首戰」"""
    from fetch_game_data import format_summary_md
    home_games = [
        {"date": "2026-04-25", "is_home": True, "opponent": "Detroit Tigers",
         "team_score": 5, "opp_score": 3, "is_winner": True},
    ]
    md = format_summary_md(_make_minimal_result(home_games=home_games))
    assert "本系列首戰" in md


def test_format_summary_md_empty_games_omits_soft_sections():
    """games 空 → 系列賽 + Streak 脈絡 sections 整段省略；hard sections 仍存在"""
    from fetch_game_data import format_summary_md
    md = format_summary_md(_make_minimal_result())
    assert "## 戰績摘要" in md  # hard section 保留
    assert "## 當前系列賽" not in md  # soft section 省略
    assert "## Streak 脈絡" not in md


def test_format_summary_md_raises_on_missing_game():
    from fetch_game_data import format_summary_md
    import pytest
    with pytest.raises(ValueError):
        format_summary_md({})


def test_format_summary_md_raises_on_missing_team_id():
    from fetch_game_data import format_summary_md
    import pytest
    bad = _make_minimal_result()
    bad["game"]["home"]["team_id"] = None
    with pytest.raises(ValueError):
        format_summary_md(bad)


def test_extract_game_info_includes_probable_pitcher_id():
    """schedule API hydrate=probablePitcher 已含 .id；extract_game_info 應寫入 probable_pitcher_id"""
    from fetch_game_data import extract_game_info
    game = {
        "gamePk": 12345,
        "gameDate": "2026-04-28T22:10:00Z",
        "status": {"abstractGameState": "Preview"},
        "venue": {"name": "Progressive Field"},
        "teams": {
            "home": {
                "team": {"name": "Cleveland Guardians", "id": 114},
                "probablePitcher": {"fullName": "Tanner Bibee", "id": 676440},
            },
            "away": {
                "team": {"name": "Tampa Bay Rays", "id": 139},
                "probablePitcher": {"fullName": "Nick Martínez", "id": 607259},
            },
        },
    }
    result = extract_game_info(game)
    assert result["home"]["probable_pitcher_id"] == 676440
    assert result["away"]["probable_pitcher_id"] == 607259


def test_extract_game_info_missing_probable_pitcher_id_is_none():
    """無 probablePitcher（TBD 先發）→ probable_pitcher_id = None"""
    from fetch_game_data import extract_game_info
    game = {
        "gamePk": 12345,
        "gameDate": "2026-04-28T22:10:00Z",
        "status": {"abstractGameState": "Preview"},
        "venue": {"name": "Progressive Field"},
        "teams": {
            "home": {"team": {"name": "Cleveland Guardians", "id": 114}},
            "away": {"team": {"name": "Tampa Bay Rays", "id": 139}},
        },
    }
    result = extract_game_info(game)
    assert result["home"]["probable_pitcher_id"] is None
    assert result["away"]["probable_pitcher_id"] is None
