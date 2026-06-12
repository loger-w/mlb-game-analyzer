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


def test_finds_12_00_et_snapshot_for_matchup(tmp_path):
    """Loads {date}_12-00-ET.json and finds the matchup within."""
    pre = (FIXTURES / "sample_snapshot_pregame.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(pre, encoding="utf-8")

    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is not None
    assert snap_ts == "2026-05-02_12-00-ET.json"


def test_returns_none_when_12_00_et_file_missing(tmp_path):
    """Only non-12:00-ET snapshots exist → return None (we only consult 12:00 ET)."""
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(inp, encoding="utf-8")
    snap, snap_ts = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    assert snap is None


def test_returns_none_when_12_00_et_snapshot_is_after_commence(tmp_path):
    """Safety net: if 12:00 ET file's snapshot_time_utc >= commence_utc, skip (in-play)."""
    inp = (FIXTURES / "sample_snapshot_inplay.json").read_text(encoding="utf-8")
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(inp, encoding="utf-8")
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


def test_extract_returns_none_if_ml_no_vig_missing():
    """fetch_odds._pair_no_vig 在任一邊缺 implied_pct 時不會寫 no_vig_pct;
    ML 缺 no_vig_pct 而 O/U 完整時必須回 None,不能 KeyError。"""
    game = {
        "home_team": "New York Yankees",
        "away_team": "Baltimore Orioles",
        "bookmakers": {"pinnacle": {
            "ml": {
                "New York Yankees": {"odds": 1.61, "implied_pct": 62.1},  # no no_vig_pct
                "Baltimore Orioles": {"odds": 2.5, "implied_pct": 40.0},
            },
            "ou": {
                "Over": {"odds": 1.9, "point": 8.5, "no_vig_pct": 51.0},
                "Under": {"odds": 1.99, "point": 8.5, "no_vig_pct": 49.0},
            },
        }},
    }
    assert extract_pinnacle_no_vig(game) is None


def test_excludes_next_day_game_in_late_snapshot(tmp_path):
    """A snapshot file dated 5/02 can contain games for 5/03. Must not match
    when caller asks for 5/02 matchup."""
    import json
    # Build a snapshot that has snapshot_time_utc on 5/02 22:00 ET (= 5/03 02:00 UTC),
    # containing a game whose game_date_et is 5/03 (commence 5/03 17:36 UTC).
    cross_day = {
        "snapshot_time_utc": "2026-05-03T02:00:00+00:00",
        "snapshot_time_et": "2026-05-02 22:00 ET",
        "game_count": 1,
        "games": [
            {
                "game": "Baltimore Orioles @ New York Yankees",
                "away_team": "Baltimore Orioles",
                "home_team": "New York Yankees",
                "commence_utc": "2026-05-03T17:36:00Z",
                "commence_et": "2026-05-03 13:36 ET",
                "game_date_et": "2026-05-03",
                "bookmakers": {
                    "pinnacle": {
                        "title": "Pinnacle",
                        "ml": {
                            "Baltimore Orioles": {"odds": 2.5, "implied_pct": 40.0, "no_vig_pct": 39.2},
                            "New York Yankees": {"odds": 1.61, "implied_pct": 62.1, "no_vig_pct": 60.8}
                        },
                        "ou": {
                            "Over": {"odds": 1.9, "point": 8.5, "implied_pct": 52.6, "no_vig_pct": 51.0},
                            "Under": {"odds": 1.99, "point": 8.5, "implied_pct": 50.3, "no_vig_pct": 49.0}
                        }
                    }
                }
            }
        ]
    }
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(cross_day), encoding="utf-8")
    snap, _ = find_closing_snapshot_for_game(
        snapshots_dir=tmp_path,
        date="2026-05-02",  # caller asks for 5/02
        home_team="New York Yankees",
        away_team="Baltimore Orioles",
    )
    # Should NOT return the 5/03 game even though it's in a 5/02-named file
    assert snap is None


def _snap(snap_utc, snap_et, commence_utc, date_et="2026-05-02",
          home="New York Yankees", away="Baltimore Orioles", over_nv=51.0):
    return {
        "snapshot_time_utc": snap_utc, "snapshot_time_et": snap_et,
        "games": [{
            "away_team": away, "home_team": home, "game_date_et": date_et,
            "commence_utc": commence_utc,
            "bookmakers": {"pinnacle": {
                "ml": {away: {"no_vig_pct": 39.2}, home: {"no_vig_pct": 60.8}},
                "ou": {"Over": {"point": 8.5, "no_vig_pct": over_nv},
                       "Under": {"point": 8.5, "no_vig_pct": 100 - over_nv}},
                "rl": {home: {"point": -1.5, "no_vig_pct": 40.0},
                       away: {"point": 1.5, "no_vig_pct": 60.0}},
            }},
        }],
    }


def test_find_entry_close_picks_earliest_and_latest(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    import json
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", "2026-05-02T22:00:00Z", over_nv=50.0)), encoding="utf-8")
    (tmp_path / "2026-05-02_15-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T19:00:00Z", "2026-05-02 15:00 ET", "2026-05-02T22:00:00Z", over_nv=53.0)), encoding="utf-8")
    (tmp_path / "2026-05-02_18-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T21:00:00Z", "2026-05-02 18:00 ET", "2026-05-02T22:00:00Z", over_nv=55.0)), encoding="utf-8")
    entry, close = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    assert entry["snapshot_time_utc"] == "2026-05-02T16:00:00Z"   # earliest
    assert close["snapshot_time_utc"] == "2026-05-02T21:00:00Z"   # latest


def test_find_entry_close_excludes_post_commence(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    import json
    (tmp_path / "2026-05-02_12-00-ET.json").write_text(json.dumps(
        _snap("2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", "2026-05-02T22:00:00Z")), encoding="utf-8")
    (tmp_path / "2026-05-02_22-00-ET.json").write_text(json.dumps(
        _snap("2026-05-03T02:00:00Z", "2026-05-02 22:00 ET", "2026-05-02T22:00:00Z")), encoding="utf-8")
    entry, close = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    assert entry["snapshot_time_utc"] == "2026-05-02T16:00:00Z"
    assert close["snapshot_time_utc"] == "2026-05-02T16:00:00Z"   # only one qualifies → entry==close


def test_find_entry_close_none_when_no_match(tmp_path):
    from lib.closing_line import find_entry_close_snapshots
    assert find_entry_close_snapshots(tmp_path, "2026-05-02", "X", "Y") == (None, None)


def test_find_entry_close_caches_per_date_dir(tmp_path):
    """回測同一天 ~15 場重複掃同目錄;同 (dir, date) 只 glob+parse 一次。
    驗證方式:第一次呼叫後刪檔,第二次呼叫仍命中快取回傳結果。"""
    from lib.closing_line import find_entry_close_snapshots
    import json
    f = tmp_path / "2026-05-02_12-00-ET.json"
    f.write_text(json.dumps(
        _snap("2026-05-02T16:00:00Z", "2026-05-02 12:00 ET", "2026-05-02T22:00:00Z")), encoding="utf-8")
    entry, _ = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    assert entry is not None
    f.unlink()
    entry2, _ = find_entry_close_snapshots(tmp_path, "2026-05-02", "New York Yankees", "Baltimore Orioles")
    assert entry2 is not None
    assert entry2["snapshot_time_utc"] == "2026-05-02T16:00:00Z"
