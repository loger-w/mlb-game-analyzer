"""Tests for prepare_game.py main script.

Strategy: 不真的呼叫子腳本（會打 API），而是 monkeypatch subprocess.run
與 Path.exists 等 I/O，測試 CLI parsing、exit code、step 順序。
"""
import os
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_parse_args_defaults():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE"])
    assert args.date == "2026-04-28"
    assert args.away == "TB"
    assert args.home == "CLE"
    assert args.season == 2026
    assert args.game_suffix is None
    assert not args.force


def test_parse_args_explicit_season_overrides():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE",
                       "--season", "2025"])
    assert args.season == 2025


def test_parse_args_doubleheader_g1():
    from prepare_game import parse_args
    args = parse_args(["--date", "2026-04-28", "--away", "TB", "--home", "CLE",
                       "--game-suffix", "G1"])
    assert args.game_suffix == "G1"


def test_compute_output_dir_default():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix=None, override=None)
    assert p == Path("analysis-data/2026-04-28/TB@CLE")


def test_compute_output_dir_doubleheader_g2():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix="G2", override=None)
    assert p == Path("analysis-data/2026-04-28/TB@CLE-G2")


def test_compute_output_dir_explicit_override():
    from prepare_game import compute_output_dir
    p = compute_output_dir(date="2026-04-28", away="TB", home="CLE",
                          game_suffix=None, override="/tmp/foo")
    assert p == Path("/tmp/foo")


def test_dossier_filename_no_suffix():
    from prepare_game import dossier_filename, skeleton_filename
    assert dossier_filename(None) == "dossier.md"
    assert skeleton_filename(None) == "phase3_skeleton.md"


def test_dossier_filename_with_suffix():
    from prepare_game import dossier_filename, skeleton_filename
    assert dossier_filename("G1") == "dossier-G1.md"
    assert skeleton_filename("G2") == "phase3_skeleton-G2.md"


def test_run_step_subprocess_failure_exits_with_propagated_code(monkeypatch, tmp_path):
    """子腳本 exit non-zero → prepare_game.py exit non-zero（傳遞 stderr）"""
    from prepare_game import run_step

    class FakeResult:
        def __init__(self):
            self.returncode = 5
            self.stdout = ""
            self.stderr = "⛔ 先發不在 active"

    def fake_run(*a, **k):
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        run_step("B", ["python", "scripts/roster_checker.py"])
    assert exc.value.code == 5


def test_tw_to_et_converts_minus_one_day():
    """spec 2026-04-29 §2: et_date = tw_date − 1 day."""
    from prepare_game import _tw_to_et
    assert _tw_to_et("2026-04-30") == "2026-04-29"
    assert _tw_to_et("2026-05-01") == "2026-04-30"
    # 跨月
    assert _tw_to_et("2026-05-01") == "2026-04-30"
    # 跨年
    assert _tw_to_et("2027-01-01") == "2026-12-31"
