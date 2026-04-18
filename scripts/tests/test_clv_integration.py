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
