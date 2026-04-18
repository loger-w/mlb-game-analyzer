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
def staged_jsonl(tmp_path):
    """analysis-data/2026-04-18/predictions.jsonl with one verified record missing CLV."""
    fixtures = Path(__file__).parent / "fixtures"
    date_dir = tmp_path / "analysis-data" / "2026-04-18"
    date_dir.mkdir(parents=True)
    record = {
        "verified": True,
        "date": "2026-04-18",
        "game_time": "2026-04-18T23:00:00Z",
        "home_team": "CHC",
        "away_team": "NYM",
        "ml_rec": "HOME",
        "ou_rec": "PASS",
        "run_line_rec": "PASS",
        "recommendation_snapshot": {
            "source": "2026-04-18_16-00-ET.json",
            "snapshot_time_et": "2026-04-18 16:00 ET",
            "snapshot_time_utc": "2026-04-18T20:00:00+00:00",
            "commence_utc": "2026-04-18T23:00:00Z",
            "minutes_before_first_pitch": 180,
            "ml": {"home": {"decimal": 1.74, "american": -135, "implied_pct": 57.5},
                   "away": {"decimal": 2.24, "american": 124, "implied_pct": 44.6}},
            "ou": None, "rl": None,
        },
        "kelly": {"ml": {"direction": "HOME", "units": 1.5, "decimal_odds": 1.74},
                  "ou": None, "rl": None},
    }
    (date_dir / "predictions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    snap_dir = tmp_path / "odds_snapshots"
    snap_dir.mkdir()
    shutil.copy(fixtures / "sample_snapshot_close.json", snap_dir / "2026-04-18_22-00-ET.json")
    return {"root": tmp_path, "date_dir": date_dir, "snap_dir": snap_dir}


def _run_backfill(root, snap_dir, *args):
    env = {**os.environ,
           "MLB_ANALYSIS_ROOT_OVERRIDE": str(root / "analysis-data"),
           "MLB_SNAPSHOT_DIR_OVERRIDE": str(snap_dir)}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "backfill_clv.py"), *args],
        env=env, cwd=str(SCRIPTS),
        capture_output=True, text=True, check=False,
    )


def test_dry_run_default_no_write(staged_jsonl):
    result = _run_backfill(staged_jsonl["root"], staged_jsonl["snap_dir"], "--date", "2026-04-18")
    assert result.returncode == 0, result.stderr
    with open(staged_jsonl["date_dir"] / "predictions.jsonl") as f:
        rec = json.loads(f.readline())
    assert "clv" not in rec
    assert "dry-run" in result.stdout.lower() or "would update" in result.stdout.lower()


def test_no_dry_run_writes(staged_jsonl):
    result = _run_backfill(staged_jsonl["root"], staged_jsonl["snap_dir"],
                             "--date", "2026-04-18", "--no-dry-run")
    assert result.returncode == 0, result.stderr
    with open(staged_jsonl["date_dir"] / "predictions.jsonl") as f:
        rec = json.loads(f.readline())
    assert rec.get("clv") is not None
    assert rec["clv"]["ml"]["cents"] != 0


def test_skip_unverified(staged_jsonl):
    with open(staged_jsonl["date_dir"] / "predictions.jsonl") as f:
        rec = json.loads(f.readline())
    rec["verified"] = False
    (staged_jsonl["date_dir"] / "predictions.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    result = _run_backfill(staged_jsonl["root"], staged_jsonl["snap_dir"],
                             "--date", "2026-04-18", "--no-dry-run")
    assert result.returncode == 0, result.stderr
    with open(staged_jsonl["date_dir"] / "predictions.jsonl") as f:
        rec2 = json.loads(f.readline())
    assert "clv" not in rec2
    assert "skipped=1" in result.stdout


def test_summary_counts(staged_jsonl):
    result = _run_backfill(staged_jsonl["root"], staged_jsonl["snap_dir"],
                             "--date", "2026-04-18", "--no-dry-run")
    assert "processed=1" in result.stdout
    assert "updated=1" in result.stdout
