"""Tests for scripts/lib/closing_line.py"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.closing_line import (
    find_closing_snapshot_for_game,
    extract_pinnacle_no_vig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


def test_finds_latest_pregame_snapshot_excludes_inplay(tmp_path):
    # Two snapshots: one pre-game @ 12:00 ET, one in-play @ 15:00 ET
    pre = (FIXTURES / "sample_snapshot_pregame.json").read_text(encoding="utf-8")
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_09-00-ET.json").write_text(pre, encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(inp, encoding="utf-8")

    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is not None
    # Must pick the pre-game one (snapshot_time_utc < commence_utc)
    assert "12:00 ET" in snap.get("snapshot_time_et", "") or "09-00" in str(snap_ts)


def test_returns_none_when_no_pregame_snapshot(tmp_path):
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(inp, encoding="utf-8")
    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is None


def test_extract_pinnacle_no_vig_returns_complete_dict():
    pre = (FIXTURES / "sample_snapshot_pregame.json")
    import json
    data = json.loads(pre.read_text(encoding="utf-8"))
    game = data["games"][0]
    line = extract_pinnacle_no_vig(game)
    assert abs(line["home_winprob_no_vig"] - 0.608) < 0.001
    assert abs(line["away_winprob_no_vig"] - 0.392) < 0.001
    assert line["total_line"] == 8.5
    assert abs(line["over_no_vig"] - 0.510) < 0.001
    assert abs(line["under_no_vig"] - 0.490) < 0.001


def test_extract_returns_none_if_pinnacle_missing(tmp_path):
    line = extract_pinnacle_no_vig({"bookmakers": {}})
    assert line is None
