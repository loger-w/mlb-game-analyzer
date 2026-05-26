"""Tests for scripts/lib/parse_summary.py"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.parse_summary import parse_summary

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def test_parse_finished_extracts_direction_total_confidence():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert result is not None
    assert result["direction"] == "HOME"  # "Yankees 中度偏優" → home (NYY)
    assert result["total"] == 8.5
    assert result["confidence"] == "MEDIUM"
    assert result["parse_failed"] is False


def test_parse_template_returns_parse_failed():
    result = parse_summary(
        FIXTURES / "sample_summary_template.md",
        home_team_abbr="SF",
        away_team_abbr="ARI",
    )
    assert result is not None
    assert result["parse_failed"] is True
    assert result["direction"] is None


def test_parse_extracts_flags():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert isinstance(result.get("has_reverse_platoon"), bool)
    assert isinstance(result.get("has_chain_break_300"), bool)
    assert isinstance(result.get("has_bullpen_il_2plus"), bool)


def test_parse_extracts_park_factor():
    result = parse_summary(
        FIXTURES / "sample_summary_finished.md",
        home_team_abbr="NYY",
        away_team_abbr="BAL",
    )
    assert result["park_factor"] is not None
    assert isinstance(result["park_factor"], float)


def test_direction_phrasing_pure_team_name():
    """ '**Yankees 中度偏優**' → HOME (when home=NYY)"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("**Yankees 中度偏優**。三大核心訊號", "NYY", "BAL") == "HOME"


def test_direction_phrasing_with_marker():
    """ '**AWAY (ATL) 顯著有利**' → AWAY"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("**AWAY (ATL) 顯著有利**。Quintana 崩盤", "COL", "ATL") == "AWAY"


def test_direction_phrasing_abbr_only():
    """ 'CHC 略佔優' → away/home depending on which is CHC"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("CHC 略佔優。三條獨立邊", "CHC", "ARI") == "HOME"
    assert _resolve_direction("CHC 略佔優。三條獨立邊", "ARI", "CHC") == "AWAY"


def test_direction_phrasing_pingpan():
    """ '持平' / '勢均力敵' / '無明顯方向' → 持平"""
    from lib.parse_summary import _resolve_direction
    assert _resolve_direction("持平 — 兩邊投打勢均", "NYY", "BAL") == "持平"
