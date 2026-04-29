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

# 11c: Step C + pitcher_stats --mlbam-id
# ---------------------------------------------------------------------------

def test_step_c_runs_both_sides_with_mlbam_id(monkeypatch, tmp_path):
    """step_c 對 home + away 各跑一次，且含 --mlbam-id 參數。"""
    from prepare_game import step_c

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_c(
        home_id=676440, away_id=607259,
        home_name="Tanner Bibee", away_name="Nick Martínez",
        season=2026, output_dir=tmp_path,
    )

    assert len(call_args) == 2
    # Both calls should pass --mlbam-id
    for args in call_args:
        assert "--mlbam-id" in args
    # Verify specific IDs appear in the combined args
    all_args = [a for cmd in call_args for a in cmd]
    assert "676440" in all_args
    assert "607259" in all_args


def test_step_c_no_mlbam_id_omits_flag(monkeypatch, tmp_path):
    """step_c: mlbam_id=None → --mlbam-id フラグ省略（只用 --name）。"""
    from prepare_game import step_c

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_c(
        home_id=None, away_id=None,
        home_name="Unknown", away_name="Unknown",
        season=2026, output_dir=tmp_path,
    )

    for args in call_args:
        assert "--mlbam-id" not in args


def test_pitcher_stats_main_accepts_mlbam_id_arg_skipping_lookup(monkeypatch, tmp_path):
    """pitcher_stats --mlbam-id provided => lookup_pitcher_id NOT called."""
    import pitcher_stats

    lookup_calls = []

    def fake_lookup(name):
        lookup_calls.append(name)
        return 99999  # should not be called

    monkeypatch.setattr(pitcher_stats, "lookup_pitcher_id", fake_lookup)

    # Monkeypatch all the fetch functions to avoid real API calls
    monkeypatch.setattr(pitcher_stats, "fetch_player_info", lambda pid: {"age": 28, "birth_date": "1996-01-01", "pitch_hand": "R"})
    monkeypatch.setattr(pitcher_stats, "fetch_mlb_api_stats", lambda pid, yr: {"era": 3.50})
    monkeypatch.setattr(pitcher_stats, "fetch_statcast_expected", lambda pid, yr: {})
    monkeypatch.setattr(pitcher_stats, "fetch_statcast_stats", lambda pid, yr: {})
    monkeypatch.setattr(pitcher_stats, "fetch_statcast_barrels", lambda pid, yr: {"error": "no data"})
    monkeypatch.setattr(pitcher_stats, "fetch_game_log", lambda pid, yr, limit=3: [])
    monkeypatch.setattr(pitcher_stats, "fetch_platoon_splits", lambda pid, yr: {})
    monkeypatch.setattr(pitcher_stats, "fetch_whiff_csw", lambda pid, yr: {"error": "no data"})
    monkeypatch.setattr(pitcher_stats, "fetch_prior_year_stats", lambda pid, yr: {})

    out_file = tmp_path / "pitcher_out.json"

    # Invoke main() with sys.argv set, writing to file (avoids StringIO reconfigure issue)
    old_argv = sys.argv
    try:
        sys.argv = ["pitcher_stats.py", "--name", "Tanner Bibee", "--mlbam-id", "676440",
                    "--no-md", "-o", str(out_file)]
        pitcher_stats.main()
    finally:
        sys.argv = old_argv

    # lookup should NOT have been called
    assert len(lookup_calls) == 0
    # Output file should contain the mlbam_id
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["mlbam_id"] == 676440


