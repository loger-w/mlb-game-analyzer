"""Tests for scripts/fetch_results.py"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from fetch_results import fetch_final_scores, build_result_record


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
