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
