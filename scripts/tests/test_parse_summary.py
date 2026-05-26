"""Tests for scripts/lib/parse_summary.py"""
import sys
from pathlib import Path
import pytest

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


def test_confidence_percentage_single_value():
    """`**方向信心**: 62%` → confidence_pct=0.62"""
    import tempfile
    from lib.parse_summary import parse_summary
    md = "## 整體判斷\n\n- **方向（基本面）**：**Yankees 中度偏優**。三大訊號\n- **方向信心**：62%\n- **總分（基本面）**：8.0\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="NYY", away_team_abbr="BAL")
    assert r["confidence_pct"] == pytest.approx(0.62)


def test_confidence_percentage_range_midpoint():
    """`**方向信心**: 62-65%` → confidence_pct = 0.635 (midpoint)"""
    import tempfile
    from lib.parse_summary import parse_summary
    md = "## 整體判斷\n\n- **方向（基本面）**：HOME 顯著有利\n- **方向信心**：62-65%\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="NYY", away_team_abbr="BAL")
    assert r["confidence_pct"] == pytest.approx(0.635)


def test_confidence_percentage_with_tilde():
    """`**方向信心**: ~70%` → confidence_pct = 0.70"""
    import tempfile
    from lib.parse_summary import parse_summary
    md = "## 整體判斷\n\n- **方向（基本面）**：AWAY (NYY) 顯著\n- **方向信心**：~70%\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="BAL", away_team_abbr="NYY")
    assert r["confidence_pct"] == pytest.approx(0.70)


def test_chain_break_finds_max_not_first():
    """When first chain break < 0.300 but later one ≥ 0.300, flag should be True."""
    import tempfile
    from lib.parse_summary import parse_summary
    md = ("## 整體判斷\n\n- **方向（基本面）**：HOME 偏優\n- **總分（基本面）**：8.0\n\n"
          "### 額外信號\n- 🟠 AWAY chain breaks at #4-5：OPS 落差 0.150\n"
          "- 🔴 HOME chain breaks at #8-9：OPS 落差 0.450\n")
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="NYY", away_team_abbr="BAL")
    assert r["has_chain_break_300"] is True


def test_total_extracts_with_tilde():
    """`**總分（基本面）**: ~7.5` → total=7.5"""
    import tempfile
    from lib.parse_summary import parse_summary
    md = "## 整體判斷\n\n- **方向（基本面）**：HOME 偏優\n- **總分（基本面）**：~7.5\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="NYY", away_team_abbr="BAL")
    assert r["total"] == 7.5


def test_park_factor_bolded():
    """`Park Factor: **82.0** →` → park_factor=82.0"""
    import tempfile
    from lib.parse_summary import parse_summary
    md = ("## 整體判斷\n\n- **方向（基本面）**：HOME 偏優\n- **總分（基本面）**：6.0\n\n"
          "## 條件修正\n\n- Park Factor: **82.0** → **-0.90 run** (T-Mobile pitcher friendly)\n")
    with tempfile.NamedTemporaryFile(mode='w', suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(md); path = t.name
    r = parse_summary(Path(path), home_team_abbr="NYY", away_team_abbr="BAL")
    assert r["park_factor"] == 82.0
