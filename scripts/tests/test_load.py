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
        "market_home_winprob_no_vig", "market_total_line",
        "actual_winner", "actual_total",
        "closing_missing", "result_missing",
        "park_factor", "has_reverse_platoon",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_load_handles_doubleheader_g_suffix_summary(tmp_path, monkeypatch):
    """When matchup dir uses the -G1 doubleheader convention,
    load should detect summary-G1.md and point dossier_path at dossier-G1.md.
    Predictions come from merged.json (deterministic path)."""
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
    # merged.json required for deterministic prediction
    (matchup / "merged.json").write_text(json.dumps({
        "home_batting_xwoba": 0.320, "away_batting_xwoba": 0.310,
        "home_starter_fip": 4.0, "away_starter_fip": 4.2, "park_factor": 100,
    }), encoding="utf-8")
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
    assert row["actual_winner"] == "HOME"


def test_load_falls_back_to_summary_md_when_both_exist(tmp_path, monkeypatch):
    """When summary.md is present (non-DH dir), dossier_path should point to dossier.md.
    Predictions come from merged.json (deterministic path); summary.md content is not used."""
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
    # merged.json required for deterministic prediction
    (matchup / "merged.json").write_text(json.dumps({
        "home_batting_xwoba": 0.320, "away_batting_xwoba": 0.310,
        "home_starter_fip": 4.0, "away_starter_fip": 4.2, "park_factor": 100,
    }), encoding="utf-8")

    monkeypatch.setattr(load_mod, "ANALYSIS_DATA_DIR", tmp_path)

    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-23"})
    assert len(df) == 1
    assert df.iloc[0]["dossier_path"] == "../2026-05-23/STL@CIN/dossier.md"


def test_all_rows_have_confidence_pct_from_predict():
    """Deterministic path: every loaded row carries a confidence_pct from predict()
    (it always returns one, even for 持平)."""
    df = build_dataframe_for_month(month="2026-05", days_filter={"2026-05-02", "2026-05-15"})

    assert len(df) > 0, "Expected ≥1 row"
    assert df["skill_confidence_pct"].notna().all(), \
        "All rows should have confidence_pct from predict()"


def test_build_row_prediction_from_merged(tmp_path):
    """_build_row computes predictions from merged.json via predict(), not summary.md."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import load

    matchup = tmp_path / "2026-05-02" / "BAL@NYY"
    matchup.mkdir(parents=True)
    (matchup / "game_data.json").write_text(json.dumps({
        "game": {"gamePk": 1, "home": {"team": "New York Yankees"},
                 "away": {"team": "Baltimore Orioles"}}}), encoding="utf-8")
    (matchup / "merged.json").write_text(json.dumps({
        "home_batting_xwoba": 0.340, "away_batting_xwoba": 0.300,
        "home_starter_fip": 3.5, "away_starter_fip": 4.5, "park_factor": 100}), encoding="utf-8")
    (matchup / "signals.json").write_text(json.dumps({
        "signals": [{"name": "reverse_platoon", "fired": True, "side": "HOME",
                     "value": 0.4, "severity": "high", "label": "", "details": {},
                     "confidence": "data", "half_life": "medium"}],
        "fired_count": 1}), encoding="utf-8")
    (matchup / "result.json").write_text(json.dumps({
        "winner": "HOME", "total": 9, "home_score": 5, "away_score": 4}), encoding="utf-8")

    row = load._build_row("2026-05-02", matchup, "BAL", "NYY")
    assert row is not None
    assert row["skill_direction"] in ("HOME", "AWAY", "持平")
    assert row["skill_confidence_pct"] is not None
    assert row["has_reverse_platoon"] is True
    assert row["park_factor"] == 100
