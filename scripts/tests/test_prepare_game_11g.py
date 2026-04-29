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

# 11g: _print_risk_notes + main()
# ---------------------------------------------------------------------------

def test_print_risk_notes_flag13_detected(tmp_path, capsys):
    """_print_risk_notes: era_xera_delta ≥ 1.5 → Flag 13 警告出力。"""
    from prepare_game import _print_risk_notes

    merged = {
        "home_pitcher": {"era_xera_delta": 2.0},
        "away_pitcher": {},
        "home_lineup": {},
        "away_lineup": {},
    }
    (tmp_path / "merged.json").write_text(json.dumps(merged), encoding="utf-8")

    _print_risk_notes(tmp_path)
    err = capsys.readouterr().err
    assert "Flag 13" in err
    assert "2.00" in err


def test_print_risk_notes_flag3_detected(tmp_path, capsys):
    """_print_risk_notes: recent_babip ≥ 0.370 → Flag 3 警告出力。"""
    from prepare_game import _print_risk_notes

    merged = {
        "home_pitcher": {},
        "away_pitcher": {},
        "home_lineup": {"recent_babip": 0.390},
        "away_lineup": {},
    }
    (tmp_path / "merged.json").write_text(json.dumps(merged), encoding="utf-8")

    _print_risk_notes(tmp_path)
    err = capsys.readouterr().err
    assert "Flag 3" in err
    assert "0.390" in err


def test_print_risk_notes_no_flags_silent(tmp_path, capsys):
    """_print_risk_notes: 正常値 → Risk Notes 出力なし。"""
    from prepare_game import _print_risk_notes

    merged = {
        "home_pitcher": {"era_xera_delta": 0.5},
        "away_pitcher": {"era_xera_delta": 0.3},
        "home_lineup": {"recent_babip": 0.310},
        "away_lineup": {"recent_babip": 0.290},
    }
    (tmp_path / "merged.json").write_text(json.dumps(merged), encoding="utf-8")

    _print_risk_notes(tmp_path)
    err = capsys.readouterr().err
    assert "Flag 13" not in err
    assert "Flag 3" not in err


def test_print_risk_notes_missing_merged_json(tmp_path, capsys):
    """_print_risk_notes: merged.json 欠如 → 何も出力しない。"""
    from prepare_game import _print_risk_notes

    _print_risk_notes(tmp_path)
    err = capsys.readouterr().err
    assert err == ""


def test_main_full_integration(monkeypatch, tmp_path):
    """main(): Steps A-G が正しい順序で呼ばれる（subprocess モック）。"""
    import types

    # Set up fake dossier/skeleton modules
    fake_dossier = types.ModuleType("dossier_renderer")
    fake_dossier.render_dossier = lambda bundle, game_dir="": "# Dossier"
    monkeypatch.setitem(sys.modules, "dossier_renderer", fake_dossier)

    fake_skel = types.ModuleType("phase3_skeleton_renderer")
    fake_skel.render_skeleton = lambda bundle, formula_pred: "# Skeleton"
    monkeypatch.setitem(sys.modules, "phase3_skeleton_renderer", fake_skel)

    fake_pred = types.ModuleType("predict")
    fake_pred.predict_with_formula = lambda merged: {}
    monkeypatch.setitem(sys.modules, "predict", fake_pred)

    step_order = []
    call_count = {"n": 0}

    def fake_run(cmd, **k):
        call_count["n"] += 1
        script = next((a for a in cmd if ".py" in str(a)), "")
        step_order.append(str(script))

        # Write expected output files based on which script is called
        for i, a in enumerate(cmd):
            if str(a) == "-o" and i + 1 < len(cmd):
                out_file = Path(cmd[i + 1])
                if not out_file.exists():
                    out_file.write_text(json.dumps({
                        "home": {
                            "team": "CLE", "team_id": 114,
                            "probable_pitcher": "Tanner Bibee",
                            "probable_pitcher_id": 676440,
                        },
                        "away": {
                            "team": "TB", "team_id": 139,
                            "probable_pitcher": "Nick Martinez",
                            "probable_pitcher_id": 607259,
                        },
                    }) if "game_data" in str(out_file) else json.dumps({}), encoding="utf-8")
        return FakeResult()

    monkeypatch.setattr("prepare_game.subprocess.run", fake_run)

    from prepare_game import main
    result = main([
        "--date", "2026-04-28",
        "--away", "TB",
        "--home", "CLE",
        "--output-dir", str(tmp_path),
    ])

    assert result == 0
    # Verify all expected scripts were called
    scripts_called = " ".join(step_order)
    assert "fetch_game_data" in scripts_called
    assert "roster_checker" in scripts_called
    assert "pitcher_stats" in scripts_called
    assert "lineup_analyzer" in scripts_called
    assert "merge_game_data" in scripts_called

    # fetch_game_data must come first
    first_script = step_order[0]
    assert "fetch_game_data" in first_script

    # merge must come after roster/pitcher/lineup
    merge_idx = next(i for i, s in enumerate(step_order) if "merge" in s)
    fetch_idx = next(i for i, s in enumerate(step_order) if "fetch_game_data" in s)
    assert merge_idx > fetch_idx

