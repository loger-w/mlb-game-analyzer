"""Tests for scripts/fetch_results.py"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fetch_results
from fetch_results import fetch_final_scores, build_result_record, find_matchup_dir_by_pk


FIXTURE = Path(__file__).parent / "fixtures" / "backtest" / "sample_mlb_schedule.json"


def test_fetch_final_scores_filters_to_final_regular_season():
    fake_resp = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with patch("fetch_results.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_resp
        mock_get.return_value.raise_for_status = lambda: None
        results = fetch_final_scores("2026-05-02")
    assert len(results) == 1
    r = results[0]
    assert r["game_pk"] == 823554
    assert r["home_team"] == "New York Yankees"
    assert r["away_team"] == "Baltimore Orioles"
    assert r["home_score"] == 5
    assert r["away_score"] == 8


def test_build_result_record_winner_away():
    record = build_result_record({
        "game_pk": 823554,
        "home_team": "New York Yankees",
        "away_team": "Baltimore Orioles",
        "home_score": 5,
        "away_score": 8,
    })
    assert record["game_pk"] == 823554
    assert record["winner"] == "AWAY"
    assert record["home_score"] == 5
    assert record["away_score"] == 8
    assert record["total"] == 13
    assert record["status"] == "Final"
    assert record["postponed"] is False


def test_build_result_record_winner_home():
    record = build_result_record({
        "game_pk": 1,
        "home_team": "H",
        "away_team": "A",
        "home_score": 7,
        "away_score": 4,
    })
    assert record["winner"] == "HOME"
    assert record["total"] == 11


def test_build_result_record_winner_tie():
    record = build_result_record({
        "game_pk": 1,
        "home_team": "H",
        "away_team": "A",
        "home_score": 5,
        "away_score": 5,
    })
    assert record["winner"] == "TIE"
    assert record["total"] == 10


def test_find_matchup_dir_by_pk_distinguishes_doubleheader(tmp_path, monkeypatch):
    """When 2 dirs share home/away team names (doubleheader), gamePk picks the right one."""
    date_dir = tmp_path / "2026-05-23"
    g1 = date_dir / "STL@CIN-G1"
    g2 = date_dir / "STL@CIN-G2"
    g1.mkdir(parents=True)
    g2.mkdir(parents=True)
    (g1 / "game_data.json").write_text(json.dumps({"game": {
        "gamePk": 824518,
        "home": {"team": "Cincinnati Reds"},
        "away": {"team": "St. Louis Cardinals"},
    }}), encoding="utf-8")
    (g2 / "game_data.json").write_text(json.dumps({"game": {
        "gamePk": 824519,
        "home": {"team": "Cincinnati Reds"},
        "away": {"team": "St. Louis Cardinals"},
    }}), encoding="utf-8")

    monkeypatch.setattr(fetch_results, "ANALYSIS_DATA_DIR", tmp_path)

    assert find_matchup_dir_by_pk("2026-05-23", 824518) == g1
    assert find_matchup_dir_by_pk("2026-05-23", 824519) == g2
    assert find_matchup_dir_by_pk("2026-05-23", 999999) is None


def test_find_matchup_dir_by_pk_returns_none_when_date_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_results, "ANALYSIS_DATA_DIR", tmp_path)
    assert find_matchup_dir_by_pk("2026-05-23", 824518) is None
