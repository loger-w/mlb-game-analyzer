"""E2E: predict.py --save produces prediction.json with recommendation_snapshot + line_movement."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"


@pytest.fixture
def tmp_analysis_dir(tmp_path):
    """Stage a minimal analysis-data/{date}/{game} dir with merged.json + odds snapshots."""
    fixtures = Path(__file__).parent / "fixtures"
    date_dir = tmp_path / "analysis-data" / "2026-04-18" / "NYM_CHC"
    date_dir.mkdir(parents=True)
    shutil.copy(fixtures / "sample_merged.json", date_dir / "merged.json")
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(fixtures / "sample_snapshot_open.json", snap_dir / "2026-04-18_00-00-ET.json")
    shutil.copy(fixtures / "sample_snapshot.json",      snap_dir / "2026-04-18_16-00-ET.json")
    return {"root": tmp_path, "game_dir": date_dir, "snap_dir": snap_dir}


def test_predict_writes_rec_snapshot_block(tmp_analysis_dir):
    game_dir = tmp_analysis_dir["game_dir"]
    env = {**os.environ, "MLB_SNAPSHOT_DIR_OVERRIDE": str(tmp_analysis_dir["snap_dir"])}
    subprocess.run(
        [sys.executable, str(SCRIPTS / "predict.py"),
         "--game-data", str(game_dir / "merged.json"), "--save"],
        check=True, env=env, cwd=str(SCRIPTS),
    )
    pred_path = game_dir / "prediction.json"
    assert pred_path.exists()
    with open(pred_path) as f:
        pred = json.load(f)
    assert "recommendation_snapshot" in pred
    assert pred["recommendation_snapshot"] is not None
    assert pred["recommendation_snapshot"]["source"] == "2026-04-18_16-00-ET.json"
    assert pred["recommendation_snapshot"]["ml"]["home"]["decimal"] == 1.74
    assert "line_movement" in pred
    assert pred["line_movement"]["open_snapshot"] == "2026-04-18_00-00-ET.json"
    assert pred["line_movement"]["close_snapshot"] is None
    assert pred["line_movement"]["open_to_rec"] is not None


def test_predict_without_snapshot(tmp_analysis_dir):
    """No snapshots at all -> recommendation_snapshot = null; line_movement still present."""
    for p in tmp_analysis_dir["snap_dir"].glob("*.json"):
        p.unlink()
    game_dir = tmp_analysis_dir["game_dir"]
    env = {**os.environ, "MLB_SNAPSHOT_DIR_OVERRIDE": str(tmp_analysis_dir["snap_dir"])}
    subprocess.run(
        [sys.executable, str(SCRIPTS / "predict.py"),
         "--game-data", str(game_dir / "merged.json"), "--save"],
        check=True, env=env, cwd=str(SCRIPTS),
    )
    with open(game_dir / "prediction.json") as f:
        pred = json.load(f)
    assert pred["recommendation_snapshot"] is None
    # line_movement must still be present and non-crashing
    assert "line_movement" in pred


# ============================================================================
# Task 7: _enrich_record_with_clv tests (flat record shape)
# ============================================================================


def test_enrich_writes_clv_fields(tmp_path):
    """After predict.py emits rec blocks, _enrich_record_with_clv writes closing_line + clv."""
    fixtures = Path(__file__).parent / "fixtures"
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(fixtures / "sample_snapshot.json",       snap_dir / "2026-04-18_16-00-ET.json")
    shutil.copy(fixtures / "sample_snapshot_close.json", snap_dir / "2026-04-18_22-00-ET.json")

    from upload_results import _enrich_record_with_clv

    record = {
        "verified": True,
        "date": "2026-04-18",
        "game_time": "2026-04-18T23:00:00Z",
        "home_team": "CHC",
        "away_team": "NYM",
        "ml_rec": "HOME",
        "ou_rec": "OVER",
        "run_line_rec": "PASS",
        "recommendation_snapshot": {
            "source": "2026-04-18_16-00-ET.json",
            "snapshot_time_et": "2026-04-18 16:00 ET",
            "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
            "commence_utc": "2026-04-18T23:00:00Z",
            "minutes_before_first_pitch": 180,
            "ml": {"home": {"decimal": 1.74, "american": -135, "implied_pct": 57.5},
                   "away": {"decimal": 2.24, "american": 124, "implied_pct": 44.6}},
            "ou": {"point": 8.0,
                   "over":  {"decimal": 1.93, "american": -108, "implied_pct": 51.8},
                   "under": {"decimal": 1.94, "american": -106, "implied_pct": 51.5}},
            "rl": {"favorite_side": "HOME",
                   "home": {"decimal": 1.56, "american": -179, "implied_pct": 64.1, "point": -1.5},
                   "away": {"decimal": 2.58, "american": 158, "implied_pct": 38.8, "point": 1.5}},
        },
        "kelly": {
            "ml": {"direction": "HOME", "units": 1.5, "decimal_odds": 1.74},
            "ou": None,
            "rl": None,
        },
    }
    _enrich_record_with_clv(record, str(snap_dir), home_full="Chicago Cubs", away_full="New York Mets")
    assert record["closing_line_source"] == "2026-04-18_22-00-ET.json"
    assert record["closing_line"]["ml"]["home"]["decimal"] == 1.69
    assert record["clv"]["ml"]["cents"] != 0
    assert record["clv"]["ml"]["bet_placed"] is True
    assert record["clv"]["ou"]["bet_placed"] is False  # kelly.ou is None -> units 0 -> False
    assert record["rec_to_close"]["ml_home_cents"] != 0


def test_enrich_idempotent(tmp_path):
    """Re-running on a record with existing clv does NOT overwrite without --force."""
    from upload_results import _enrich_record_with_clv
    snap_dir = tmp_path / "odds_snapshots"; snap_dir.mkdir()
    rec = {"verified": True, "clv": {"ml": {"cents": 5}}, "recommendation_snapshot": {}, "kelly": {}}
    _enrich_record_with_clv(rec, str(snap_dir), home_full="H", away_full="A")
    assert rec["clv"]["ml"]["cents"] == 5  # unchanged


def test_enrich_force_overwrites(tmp_path):
    from upload_results import _enrich_record_with_clv
    fixtures = Path(__file__).parent / "fixtures"
    snap_dir = tmp_path / "odds_snapshots"; snap_dir.mkdir()
    shutil.copy(fixtures / "sample_snapshot_close.json", snap_dir / "2026-04-18_22-00-ET.json")
    rec = {
        "verified": True,
        "clv": {"ml": {"cents": 99}},
        "date": "2026-04-18",
        "game_time": "2026-04-18T23:00:00Z",
        "home_team": "CHC",
        "away_team": "NYM",
        "ml_rec": "HOME",
        "ou_rec": "PASS",
        "run_line_rec": "PASS",
        "recommendation_snapshot": {
            "source": "r", "snapshot_time_et": "x", "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
            "commence_utc": "2026-04-18T23:00:00Z", "minutes_before_first_pitch": 180,
            "ml": {"home": {"decimal": 1.74, "american": -135, "implied_pct": 57.5},
                   "away": {"decimal": 2.24, "american": 124, "implied_pct": 44.6}},
            "ou": None, "rl": None,
        },
        "kelly": {"ml": {"units": 1.0}, "ou": None, "rl": None},
    }
    _enrich_record_with_clv(rec, str(snap_dir), home_full="Chicago Cubs", away_full="New York Mets", force=True)
    assert rec["clv"]["ml"]["cents"] != 99


def test_enrich_no_closing_snapshot(tmp_path):
    from upload_results import _enrich_record_with_clv
    snap_dir = tmp_path / "odds_snapshots"; snap_dir.mkdir()
    rec = {
        "verified": True,
        "date": "2026-04-18",
        "game_time": "2026-04-18T23:00:00Z",
        "home_team": "CHC",
        "away_team": "NYM",
        "ml_rec": "HOME",
        "ou_rec": "PASS",
        "run_line_rec": "PASS",
        "recommendation_snapshot": {"ml": {"home": {"decimal": 1.74}, "away": {"decimal": 2.24}}, "ou": None, "rl": None},
        "kelly": {"ml": {"units": 1.0}, "ou": None, "rl": None},
    }
    _enrich_record_with_clv(rec, str(snap_dir), home_full="Chicago Cubs", away_full="New York Mets")
    assert rec["clv"] is None
    assert "no_closing_snapshot" in rec.get("clv_warnings", [])


def test_enrich_legacy_no_kelly(tmp_path):
    from upload_results import _enrich_record_with_clv
    fixtures = Path(__file__).parent / "fixtures"
    snap_dir = tmp_path / "odds_snapshots"; snap_dir.mkdir()
    shutil.copy(fixtures / "sample_snapshot_close.json", snap_dir / "2026-04-18_22-00-ET.json")
    rec = {"verified": True, "date": "2026-04-18", "game_time": "2026-04-18T23:00:00Z",
           "home_team": "CHC", "away_team": "NYM",
           "ml_rec": "HOME", "ou_rec": "PASS", "run_line_rec": "PASS"}
    _enrich_record_with_clv(rec, str(snap_dir), home_full="Chicago Cubs", away_full="New York Mets")
    assert rec.get("clv") is None
    assert "no_kelly_block" in rec.get("clv_warnings", []) or "no_rec_snapshot" in rec.get("clv_warnings", [])
