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
