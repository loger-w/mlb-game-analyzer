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

# 11d: Step D
# ---------------------------------------------------------------------------

def test_step_d_home_lineup_vs_away_pitcher(monkeypatch, tmp_path):
    """step_d: home 打線 vs away 投手（opposing-pitcher-id = away_id）。"""
    from prepare_game import step_d

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_d(
        home="CLE", away="TB",
        home_id=676440, away_id=607259,
        season=2026, output_dir=tmp_path,
    )

    assert len(call_args) == 2
    # Find which call is for home (CLE) and which is for away (TB)
    home_call = next(c for c in call_args if "CLE" in c)
    away_call = next(c for c in call_args if "TB" in c)

    # Home lineup should face away pitcher (607259)
    assert "607259" in home_call
    # Away lineup should face home pitcher (676440)
    assert "676440" in away_call


def test_step_d_opposing_pitcher_id_arg_present(monkeypatch, tmp_path):
    """step_d: --opposing-pitcher-id フラグが両側に渡される。"""
    from prepare_game import step_d

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_d(
        home="CLE", away="TB",
        home_id=100, away_id=200,
        season=2026, output_dir=tmp_path,
    )

    for cmd in call_args:
        assert "--opposing-pitcher-id" in cmd


