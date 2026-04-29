"""Tests for prepare_game.py Steps A-G.

Strategy: 不真的呼叫子腳本（會打 API），而是 monkeypatch prepare_game.subprocess.run
與 Path I/O，測試各 step 的行為、exit code、並行呼叫數量。
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 11f: _load_bundle + step_f + step_g
# ---------------------------------------------------------------------------

def test_load_bundle_reads_existing_files(tmp_path):
    """_load_bundle: 存在するファイルだけ読み込む。"""
    from prepare_game import _load_bundle

    (tmp_path / "game_data.json").write_text('{"key": "gd"}', encoding="utf-8")
    (tmp_path / "home_pitcher.json").write_text('{"era": 3.5}', encoding="utf-8")
    # away_pitcher.json は存在しない

    bundle = _load_bundle(tmp_path)
    assert "game_data" in bundle
    assert bundle["game_data"] == {"key": "gd"}
    assert "home_pitcher" in bundle
    assert "away_pitcher" not in bundle


def test_load_bundle_missing_dir_returns_empty_bundle(tmp_path):
    """_load_bundle: 何もない dir → 空 bundle。"""
    from prepare_game import _load_bundle

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    bundle = _load_bundle(empty_dir)
    assert bundle == {}


def test_step_f_writes_dossier_md(monkeypatch, tmp_path):
    """step_f: render_dossier() の結果を dossier_path に書く。"""
    from prepare_game import step_f

    # Create minimal bundle files
    (tmp_path / "game_data.json").write_text('{}', encoding="utf-8")

    monkeypatch.setattr("prepare_game.sys.path", list(sys.path))

    # Mock dossier_renderer.render_dossier
    import types
    fake_module = types.ModuleType("dossier_renderer")
    fake_module.render_dossier = lambda bundle, game_dir="": "# Dossier\nContent here"
    monkeypatch.setitem(sys.modules, "dossier_renderer", fake_module)

    dossier_path = tmp_path / "dossier.md"
    step_f(output_dir=tmp_path, dossier_path=dossier_path)

    assert dossier_path.exists()
    assert "Dossier" in dossier_path.read_text(encoding="utf-8")


def test_step_g_writes_skeleton_md(monkeypatch, tmp_path):
    """step_g: render_skeleton() の結果を skeleton_path に書く。"""
    from prepare_game import step_g

    (tmp_path / "merged.json").write_text('{}', encoding="utf-8")

    monkeypatch.setattr("prepare_game.sys.path", list(sys.path))

    import types
    fake_skel = types.ModuleType("phase3_skeleton_renderer")
    fake_skel.render_skeleton = lambda bundle, formula_pred: "# Skeleton\nContent"
    monkeypatch.setitem(sys.modules, "phase3_skeleton_renderer", fake_skel)

    fake_pred = types.ModuleType("predict")
    fake_pred.predict_with_formula = lambda merged: {"home_score": 4, "away_score": 3}
    monkeypatch.setitem(sys.modules, "predict", fake_pred)

    skeleton_path = tmp_path / "phase3_skeleton.md"
    step_g(output_dir=tmp_path, skeleton_path=skeleton_path)

    assert skeleton_path.exists()
    assert "Skeleton" in skeleton_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
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
