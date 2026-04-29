"""Tests for prepare_game.py Steps A-G."""
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_fake_run(returncode=0, stdout="", stderr=""):
    def fake_run(*a, **k):
        return FakeResult(returncode=returncode, stdout=stdout, stderr=stderr)
    return fake_run

# 11a: Step A
# ---------------------------------------------------------------------------

def test_step_a_extracts_pitcher_ids(monkeypatch, tmp_path):
    """step_a 從 game_data.json 提取 pitcher IDs 與名字。"""
    from prepare_game import step_a

    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "_meta": {},
        "home": {
            "team": "CLE",
            "team_id": 114,
            "probable_pitcher": "Tanner Bibee",
            "probable_pitcher_id": 676440,
        },
        "away": {
            "team": "TB",
            "team_id": 139,
            "probable_pitcher": "Nick Martínez",
            "probable_pitcher_id": 607259,
        },
    }), encoding="utf-8")

    monkeypatch.setattr("prepare_game.subprocess.run", make_fake_run())
    result = step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)

    assert result["home_id"] == 676440
    assert result["away_id"] == 607259
    assert result["home_name"] == "Tanner Bibee"
    assert result["away_name"] == "Nick Martínez"


def test_step_a_non_regular_season_exits_2(monkeypatch, tmp_path):
    """step_a gameType != 'R' → sys.exit(2)。"""
    from prepare_game import step_a

    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "gameType": "P",
        "home": {"probable_pitcher": "X", "probable_pitcher_id": 1},
        "away": {"probable_pitcher": "Y", "probable_pitcher_id": 2},
    }), encoding="utf-8")

    monkeypatch.setattr("prepare_game.subprocess.run", make_fake_run())
    with pytest.raises(SystemExit) as exc:
        step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)
    assert exc.value.code == 2


def test_step_a_regular_season_passes(monkeypatch, tmp_path):
    """step_a gameType == 'R' 不 exit。"""
    from prepare_game import step_a

    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "gameType": "R",
        "home": {"probable_pitcher": "P1", "probable_pitcher_id": 100},
        "away": {"probable_pitcher": "P2", "probable_pitcher_id": 200},
    }), encoding="utf-8")

    monkeypatch.setattr("prepare_game.subprocess.run", make_fake_run())
    result = step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)
    assert result["home_id"] == 100
    assert result["away_id"] == 200


def test_step_a_no_gametype_field_passes(monkeypatch, tmp_path):
    """step_a 沒有 gameType 欄位時不 exit（視為 acceptable）。"""
    from prepare_game import step_a

    game_data_path = tmp_path / "game_data.json"
    game_data_path.write_text(json.dumps({
        "home": {"probable_pitcher": "P1", "probable_pitcher_id": 100},
        "away": {"probable_pitcher": "P2", "probable_pitcher_id": 200},
    }), encoding="utf-8")

    monkeypatch.setattr("prepare_game.subprocess.run", make_fake_run())
    result = step_a(date="2026-04-28", team_abbr="TB", output_dir=tmp_path)
    assert result["home_id"] == 100


