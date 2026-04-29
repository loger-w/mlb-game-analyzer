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


