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

# 11e: Step E
# ---------------------------------------------------------------------------

def test_step_e_runs_merge_game_data(monkeypatch, tmp_path):
    """step_e: merge_game_data.py が呼ばれる。"""
    from prepare_game import step_e

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_e(output_dir=tmp_path)

    assert len(call_args) == 1
    assert any("merge_game_data" in str(a) for a in call_args[0])


def test_step_e_includes_all_inputs(monkeypatch, tmp_path):
    """step_e: merge_game_data には game/pitcher/lineup すべてが渡される。"""
    from prepare_game import step_e

    captured_cmd = []

    def fake_run(cmd, **k):
        captured_cmd.extend(cmd)
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_e(output_dir=tmp_path)

    cmd_str = " ".join(str(a) for a in captured_cmd)
    assert "game_data.json" in cmd_str
    assert "home_pitcher.json" in cmd_str
    assert "away_pitcher.json" in cmd_str
    assert "home_lineup.json" in cmd_str
    assert "away_lineup.json" in cmd_str
    assert "merged.json" in cmd_str


