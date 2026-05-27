"""Tests for scripts/lib/load.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import lib.load as load_mod
from lib.load import build_dataframe_for_month, _matchup_to_abbrs


def test_matchup_to_abbrs_basic():
    assert _matchup_to_abbrs("BAL@NYY") == ("BAL", "NYY")
    assert _matchup_to_abbrs("ARI@SF") == ("ARI", "SF")


def test_matchup_to_abbrs_with_suffix():
    # Doubleheader suffix forms: -1, -2, -G2
    assert _matchup_to_abbrs("BAL@NYY-1") == ("BAL", "NYY")
    assert _matchup_to_abbrs("BAL@NYY-G2") == ("BAL", "NYY")
    assert _matchup_to_abbrs("STL@CIN-G2") == ("STL", "CIN")


def test_build_dataframe_real_data_smoke(tmp_path):
    """Smoke test against real 2026-05-02 data (1 day, expect ~12-14 rows)."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02"})
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = {
        "date", "matchup", "game_pk",
        "skill_direction", "skill_total", "skill_confidence", "skill_confidence_pct",
        "skill_prob_mapped",
        "market_home_winprob_no_vig", "market_total_line",
        "actual_winner", "actual_total",
        "parse_failed", "closing_missing", "result_missing",
        "park_factor", "has_reverse_platoon",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_build_dataframe_marks_template_as_parse_failed():
    """All 5/25 summaries are template state; all rows should be parse_failed=True."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-25"})
    assert len(df) > 0, "Expected at least 1 row for 5/25"
    assert df["parse_failed"].all(), \
        f"Expected all rows to be parse_failed=True, got {df['parse_failed'].sum()}/{len(df)}"


def test_load_handles_doubleheader_g_suffix_summary(tmp_path, monkeypatch):
    """When matchup dir has summary-G1.md instead of summary.md (doubleheader convention),
    load should pick it up and point dossier_path at dossier-G1.md."""
    date_dir = tmp_path / "2026-05-23"
    matchup = date_dir / "STL@CIN-G1"
    matchup.mkdir(parents=True)
    (matchup / "game_data.json").write_text(json.dumps({"game": {
        "gamePk": 824518,
        "home": {"team": "Cincinnati Reds"},
        "away": {"team": "St. Louis Cardinals"},
    }}), encoding="utf-8")
    # Doubleheader convention: -G1 suffix on summary + dossier
    (matchup / "summary-G1.md").write_text(
        "## 整體判斷\n\n"
        "- **方向（基本面）**：CIN 略佔優。三條獨立邊\n"
        "- **總分（基本面）**：8.0\n"
        "- **方向信心**：62%\n",
        encoding="utf-8",
    )
    (matchup / "result.json").write_text(json.dumps({
        "game_pk": 824518, "winner": "HOME", "total": 7,
        "home_score": 4, "away_score": 3,
        "final_score": [4, 3], "status": "Final", "postponed": False,
    }), encoding="utf-8")

    monkeypatch.setattr(load_mod, "ANALYSIS_DATA_DIR", tmp_path)

    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-23"})
    assert len(df) == 1
    row = df.iloc[0]
    assert row["matchup"] == "STL@CIN-G1"
    assert row["dossier_path"] == "../2026-05-23/STL@CIN-G1/dossier-G1.md"
    assert row["parse_failed"] == False
    assert row["actual_winner"] == "HOME"


def test_load_falls_back_to_summary_md_when_both_exist(tmp_path, monkeypatch):
    """Prefer summary.md over summary-G*.md when both are present (non-DH dir)."""
    date_dir = tmp_path / "2026-05-23"
    matchup = date_dir / "STL@CIN"
    matchup.mkdir(parents=True)
    (matchup / "game_data.json").write_text(json.dumps({"game": {
        "gamePk": 824520,
        "home": {"team": "Cincinnati Reds"},
        "away": {"team": "St. Louis Cardinals"},
    }}), encoding="utf-8")
    (matchup / "summary.md").write_text(
        "## 整體判斷\n\n- **方向（基本面）**：CIN\n- **總分（基本面）**：8.0\n- **方向信心**：62%\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(load_mod, "ANALYSIS_DATA_DIR", tmp_path)

    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-23"})
    assert len(df) == 1
    assert df.iloc[0]["dossier_path"] == "../2026-05-23/STL@CIN/dossier.md"


def test_skill_prob_mapped_prefers_pct_when_available():
    """When confidence_pct is set, skill_prob_mapped should use that value;
    otherwise fall back to bucket mapping (LOW=0.55 / MEDIUM=0.62 / HIGH=0.72)."""
    # Use real data — pick a day where some rows have pct, some have bucket
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02", "2026-05-15"})
    valid = df[~df["parse_failed"]]

    pct_rows = valid[valid["skill_confidence_pct"].notna()]
    bucket_only = valid[valid["skill_confidence_pct"].isna() & valid["skill_confidence"].notna()]

    # Both paths must be exercised in current data — fail loudly if data changes
    assert len(pct_rows) > 0, "Expected ≥1 row with confidence_pct set (post-5/4 format)"
    assert len(bucket_only) > 0, "Expected ≥1 row with only bucket confidence (pre-5/4 format)"

    # pct path
    for _, r in pct_rows.iterrows():
        assert r["skill_prob_mapped"] == r["skill_confidence_pct"]

    # bucket path
    expected_map = {"LOW": 0.55, "MEDIUM": 0.62, "HIGH": 0.72}
    for _, r in bucket_only.iterrows():
        assert r["skill_prob_mapped"] == expected_map[r["skill_confidence"]]
