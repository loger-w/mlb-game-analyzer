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

# 11b: Step B
# ---------------------------------------------------------------------------

def test_step_b_runs_both_sides_parallel(monkeypatch, tmp_path):
    """step_b 對 home + away 各跑一次 subprocess.run（共 2 次）。"""
    from prepare_game import step_b

    call_args = []

    def fake_run(cmd, **k):
        call_args.append(cmd)
        # Write stub output files so step succeeds
        for arg in cmd:
            if arg.endswith(".json"):
                Path(arg).write_text("{}", encoding="utf-8")
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    step_b(
        home="CLE", away="TB", season=2026,
        home_pitcher="Tanner Bibee", away_pitcher="Nick Martínez",
        output_dir=tmp_path,
    )

    assert len(call_args) == 2
    # Both calls should include roster_checker.py
    for args in call_args:
        assert any("roster_checker" in str(a) for a in args)


def test_step_b_starter_not_active_exits_5(monkeypatch, tmp_path):
    """step_b: STARTER_NOT_ACTIVE 在 stderr → sys.exit(5)。"""
    from prepare_game import step_b

    def fake_run(cmd, **k):
        return FakeResult(returncode=1, stderr="STARTER_NOT_ACTIVE: pitcher not on roster")

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        step_b(
            home="CLE", away="TB", season=2026,
            home_pitcher="Unknown Pitcher", away_pitcher="Nick Martínez",
            output_dir=tmp_path,
        )
    assert exc.value.code == 5


def test_step_b_starter_not_active_in_stdout_exits_5(monkeypatch, tmp_path):
    """step_b: STARTER_NOT_ACTIVE 在 stdout → sys.exit(5)。"""
    from prepare_game import step_b

    def fake_run(cmd, **k):
        return FakeResult(returncode=1, stdout="STARTER_NOT_ACTIVE", stderr="")

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        step_b(
            home="CLE", away="TB", season=2026,
            home_pitcher="Bad Pitcher", away_pitcher="Nick Martínez",
            output_dir=tmp_path,
        )
    assert exc.value.code == 5


def test_step_b_non_zero_non_starter_propagates_code(monkeypatch, tmp_path):
    """step_b: 其他錯誤碼不含 STARTER_NOT_ACTIVE → propagate exit code。"""
    from prepare_game import step_b

    def fake_run(cmd, **k):
        return FakeResult(returncode=7, stderr="API failure")

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        step_b(
            home="CLE", away="TB", season=2026,
            home_pitcher="Pitcher A", away_pitcher="Pitcher B",
            output_dir=tmp_path,
        )
    assert exc.value.code == 7


