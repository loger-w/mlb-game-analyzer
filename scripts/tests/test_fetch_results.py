"""Unit tests for fetch_results.py — MLB Stats API 比分抓取與回填。"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from fetch_results import (
    apply_scores_to_predictions,
    fetch_final_scores,
)


def test_fetch_final_scores_parses_linescore():
    """MLB API schedule+linescore response → list of (home_team, away_team, home_score, away_score)"""
    mock_response = {
        "dates": [{
            "games": [{
                "teams": {
                    "home": {"team": {"name": "New York Yankees"}, "score": 7},
                    "away": {"team": {"name": "Boston Red Sox"}, "score": 3},
                },
                "status": {"abstractGameState": "Final"},
                "gameType": "R",
            }]
        }]
    }
    with patch("fetch_results.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_final_scores("2026-04-21")
    assert len(result) == 1
    g = result[0]
    assert g["home_team"] == "New York Yankees"
    assert g["away_team"] == "Boston Red Sox"
    assert g["home_score"] == 7
    assert g["away_score"] == 3


def test_fetch_final_scores_skips_non_final():
    """Non-Final games excluded"""
    mock_response = {
        "dates": [{
            "games": [
                {"teams": {"home": {"team": {"name": "A"}, "score": 0},
                           "away": {"team": {"name": "B"}, "score": 0}},
                 "status": {"abstractGameState": "Preview"},
                 "gameType": "R"},
                {"teams": {"home": {"team": {"name": "C"}, "score": 5},
                           "away": {"team": {"name": "D"}, "score": 2}},
                 "status": {"abstractGameState": "Final"},
                 "gameType": "R"},
            ]
        }]
    }
    with patch("fetch_results.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_final_scores("2026-04-21")
    assert len(result) == 1
    assert result[0]["home_team"] == "C"


def test_fetch_final_scores_only_regular_season():
    """gameType != R excluded (排除春訓)"""
    mock_response = {
        "dates": [{
            "games": [{
                "teams": {"home": {"team": {"name": "X"}, "score": 4},
                          "away": {"team": {"name": "Y"}, "score": 2}},
                "status": {"abstractGameState": "Final"},
                "gameType": "S",  # Spring
            }]
        }]
    }
    with patch("fetch_results.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_final_scores("2026-03-15")
    assert result == []


def test_apply_scores_writes_actual_to_prediction_json(tmp_path):
    """給 Final 比分，對應到 per-game prediction.json，寫入 actual_* + verified=true"""
    date = "2026-04-21"
    date_dir = tmp_path / "analysis-data" / date / "BOS@NYY"
    date_dir.mkdir(parents=True)
    pred_path = date_dir / "prediction.json"
    pred_path.write_text(json.dumps({
        "date": date,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "actual_home_score": None,
        "actual_away_score": None,
        "actual_winner": None,
        "actual_total": None,
        "verified": False,
    }), encoding="utf-8")

    scores = [{
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_score": 7,
        "away_score": 3,
    }]
    count = apply_scores_to_predictions(date, scores, analysis_data_dir=tmp_path / "analysis-data")
    assert count == 1

    rec = json.loads(pred_path.read_text(encoding="utf-8"))
    assert rec["actual_home_score"] == 7
    assert rec["actual_away_score"] == 3
    assert rec["actual_winner"] == "HOME"
    assert rec["actual_total"] == 10
    assert rec["verified"] is True


def test_apply_scores_skips_missing_prediction(tmp_path):
    """沒有對應 prediction.json 的比賽略過，不 error"""
    date = "2026-04-21"
    (tmp_path / "analysis-data" / date).mkdir(parents=True)
    scores = [{
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "home_score": 7,
        "away_score": 3,
    }]
    count = apply_scores_to_predictions(date, scores, analysis_data_dir=tmp_path / "analysis-data")
    assert count == 0
