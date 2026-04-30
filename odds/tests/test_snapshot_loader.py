"""Tests for odds.lib.snapshot_loader (ET-native)."""
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from snapshot_loader import (
    Snapshot,
    GameRecord,
    ET,
    load_snapshots_for_et_date,
    collect_game_timeline,
    select_anchor,
)


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ── load_snapshots_for_et_date ────────────────────────────────────────────────

def test_loads_all_snapshots_in_dir():
    """Loader 已不依賴檔名日期；呼叫應回目錄下所有合法 snapshot，按 utc 時間排序。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    assert len(snapshots) == 4
    times = [s.snapshot_time_utc for s in snapshots]
    assert times == sorted(times)


def test_load_returns_full_directory_regardless_of_date():
    """無論 et_date 為何，loader 都應回目錄下全部合法 snapshot。"""
    a = load_snapshots_for_et_date("2026-04-26", FIXTURES)
    b = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    c = load_snapshots_for_et_date("2026-04-29", FIXTURES)
    assert len(a) == len(b) == len(c) == 4


def test_loaded_snapshot_has_no_tw_field():
    """ET-native 後 Snapshot 不該有 snapshot_time_tw 屬性。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    assert len(snapshots) > 0
    s = snapshots[0]
    assert not hasattr(s, "snapshot_time_tw")
    assert s.snapshot_time_et is not None
    assert s.snapshot_time_utc is not None


def test_collect_timeline_missing_date_returns_empty():
    """不存在的 game_date_et → collect_game_timeline 回空 dict。"""
    snapshots = load_snapshots_for_et_date("2099-01-01", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2099-01-01")
    assert timelines == {}


# ── collect_game_timeline ─────────────────────────────────────────────────────

def test_collect_timeline_groups_by_game_key():
    """同一場跨兩 snapshot → 同一 game_key → list 兩筆按時間排序。

    Fixture 04-27 場次（commence_utc 2026-04-27T22:11Z = ET 18:11）→ game_date_et = 2026-04-27。
    """
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    assert len(timelines) == 3   # TBR@CLE / STL@PIT / BOS@TOR

    tbr_cle = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]
    assert len(tbr_cle) == 2
    # anchor odds 1.85, latest 1.60
    assert tbr_cle[0].pinnacle["ml"]["Cleveland Guardians"]["odds"] == 1.85
    assert tbr_cle[1].pinnacle["ml"]["Cleveland Guardians"]["odds"] == 1.60


def test_collect_timeline_filters_by_game_date_et():
    """commence_utc → game_date_et 不符的場應被跳過。"""
    s27 = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    s26 = load_snapshots_for_et_date("2026-04-26", FIXTURES)
    all_snaps = s27 + s26
    # 只要 ET 4/27 開打的場；ET 4/26 開打的 Detroit 不該出現
    timelines = collect_game_timeline(all_snaps, "2026-04-27")
    keys = list(timelines.keys())
    assert all(k[0] != "Detroit Tigers" for k in keys)


def test_cross_day_snapshot_contributes_to_later_date():
    """檔名為 04-28 的 snapshot 內含 ET 4/29 場次（TBR@CLE, commence_utc 2026-04-29T17:11Z = ET 13:11），
    在 game_date_et 2026-04-29 應能取到那場 timeline。
    """
    snapshots = load_snapshots_for_et_date("2026-04-29", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-29")
    keys = list(timelines.keys())
    assert any(k[0] == "Tampa Bay Rays" and k[1] == "Cleveland Guardians" for k in keys), \
        f"ET 4/29 TBR@CLE timeline 應出現，實際 keys={keys}"
    # 驗證 ET 4/28 的 Houston 場（commence_utc 2026-04-28T22:38Z = ET 18:38）不該誤入 ET 4/29 timeline
    assert not any(k[0] == "Houston Astros" for k in keys), \
        "ET 4/28 HOU@BAL 不該出現在 ET 4/29 timeline"


def test_early_et_game_groups_to_correct_et_day():
    """ET 早場（commence_utc UTC 13:30 = ET 09:30）→ game_date_et 自然為 ET 同日。

    驗證 ET-native 過濾邏輯對早場無歧義（不會像舊 TW 化版本誤分到隔日）。
    """
    g = _make_game(
        away="London A", home="London B",
        commence_utc="2026-04-29T13:30:00Z",     # ET 09:30
        game_date_et="2026-04-29",
    )
    snap = _make_synthetic_snapshot("2026-04-29 04:00 ET", [g])
    timelines = collect_game_timeline([snap], "2026-04-29")
    assert len(timelines) == 1
    # 不該誤分到隔日
    timelines_4_30 = collect_game_timeline([snap], "2026-04-30")
    assert timelines_4_30 == {}


def test_doubleheader_separate_keys():
    """同 matchup 同 game_date 但 commence_utc 不同 → 兩個獨立 timeline。

    commence_utc 2026-05-01T17:05Z = ET 13:05；2026-05-01T20:35Z = ET 16:35 → 都 game_date_et = 2026-05-01。
    """
    snap1 = _make_synthetic_snapshot("2026-05-01 00:00 ET", [
        _make_game(away="A", home="B", commence_utc="2026-05-01T17:05:00Z", game_date_et="2026-05-01"),
    ])
    snap2 = _make_synthetic_snapshot("2026-05-01 00:00 ET", [
        _make_game(away="A", home="B", commence_utc="2026-05-01T20:35:00Z", game_date_et="2026-05-01"),
    ])
    snap3 = _make_synthetic_snapshot("2026-05-01 04:00 ET", [
        _make_game(away="A", home="B", commence_utc="2026-05-01T17:05:00Z", game_date_et="2026-05-01"),
        _make_game(away="A", home="B", commence_utc="2026-05-01T20:35:00Z", game_date_et="2026-05-01"),
    ])
    timelines = collect_game_timeline([snap1, snap2, snap3], "2026-05-01")
    assert len(timelines) == 2
    early = [t for k, t in timelines.items() if k[2] == "2026-05-01T17:05:00Z"][0]
    late = [t for k, t in timelines.items() if k[2] == "2026-05-01T20:35:00Z"][0]
    assert len(early) == 2
    assert len(late) == 2


def test_team_name_mismatch_skips_game():
    """ml.keys 不等於 {away, home} → 跳該場、不 crash。"""
    bad_game = _make_game(away="Foo", home="Bar", commence_utc="2026-05-01T17:05:00Z", game_date_et="2026-05-01")
    bad_game["bookmakers"]["pinnacle"]["ml"] = {
        "Foo": {"odds": 1.80, "implied_pct": 55.6},
        "Wrong Team": {"odds": 2.10, "implied_pct": 47.6},
    }
    snap = _make_synthetic_snapshot("2026-05-01 00:00 ET", [bad_game])
    timelines = collect_game_timeline([snap], "2026-05-01")
    assert timelines == {}


def test_post_commence_snapshot_skipped():
    """snapshot_time_utc >= commence_utc → 視為賽中賠率、跳該 record。"""
    g = _make_game(away="A", home="B", commence_utc="2026-05-01T17:05:00Z", game_date_et="2026-05-01")
    snap_after = Snapshot(
        snapshot_time_et=datetime(2026, 5, 1, 14, 0),
        snapshot_time_utc=datetime(2026, 5, 1, 18, 5, tzinfo=timezone.utc),
        games=[g],
    )
    snap_before = Snapshot(
        snapshot_time_et=datetime(2026, 5, 1, 8, 0),
        snapshot_time_utc=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        games=[g],
    )
    timelines = collect_game_timeline([snap_before, snap_after], "2026-05-01")
    assert len(timelines) == 1
    timeline = list(timelines.values())[0]
    assert len(timeline) == 1   # 只剩 snap_before


def test_pinnacle_missing_skips_game():
    """bookmakers 沒 pinnacle → 跳。"""
    g = _make_game(away="Foo", home="Bar", commence_utc="2026-05-01T17:05:00Z", game_date_et="2026-05-01")
    g["bookmakers"] = {}
    snap = _make_synthetic_snapshot("2026-05-01 00:00 ET", [g])
    timelines = collect_game_timeline([snap], "2026-05-01")
    assert timelines == {}


def test_game_record_has_et_fields():
    """每筆 GameRecord 應有 ET 相關欄位，格式正確；TW 欄位完全消失。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    record = list(timelines.values())[0][0]
    assert record.commence_et_label.endswith(" ET")
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} ET$", record.commence_et_label)
    assert re.match(r"^\d{2}-\d{2} \d{2}:\d{2}$", record.snapshot_time_et_label)
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", record.game_date_et)
    assert record.game_date_et == "2026-04-27"
    # TW 欄位不該存在
    assert not hasattr(record, "commence_tw_label")
    assert not hasattr(record, "snapshot_time_tw")
    assert not hasattr(record, "snapshot_time_tw_label")
    assert not hasattr(record, "game_date_tw")


# ── select_anchor ─────────────────────────────────────────────────────────────

def test_select_anchor_returns_first():
    """select_anchor 應回 timeline[0]。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    a_timeline = list(timelines.values())[0]
    anchor = select_anchor(a_timeline)
    assert anchor is a_timeline[0]


def test_select_anchor_single_record():
    """timeline 只一筆 → anchor == latest 同筆。

    snapshots[1] = 04-27_00-00-ET（第一個含 ET 4/27 場次的 snapshot），game_date_et = 2026-04-27。
    """
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline([snapshots[1]], "2026-04-27")
    a_timeline = list(timelines.values())[0]
    assert len(a_timeline) == 1
    anchor = select_anchor(a_timeline)
    assert anchor is a_timeline[0]


# ── 輔助 ──────────────────────────────────────────────────────────────────────

def _make_synthetic_snapshot(snapshot_time_et_label: str, games: list[dict]) -> Snapshot:
    et_dt = datetime.strptime(snapshot_time_et_label.replace(" ET", ""), "%Y-%m-%d %H:%M")
    snap_utc = et_dt.replace(tzinfo=timezone.utc)
    return Snapshot(
        snapshot_time_et=et_dt,
        snapshot_time_utc=snap_utc,
        games=games,
    )


def _make_game(away: str, home: str, commence_utc: str, game_date_et: str) -> dict:
    return {
        "game": f"{away} @ {home}",
        "away_team": away,
        "home_team": home,
        "commence_utc": commence_utc,
        "commence_et": "synthetic",
        "game_date_et": game_date_et,
        "bookmakers": {
            "pinnacle": {
                "title": "Pinnacle",
                "ml": {
                    home: {"odds": 1.80, "implied_pct": 55.6},
                    away: {"odds": 2.10, "implied_pct": 47.6},
                },
                "ou": {
                    "Over":  {"odds": 1.91, "point": 8.5, "implied_pct": 52.4},
                    "Under": {"odds": 1.95, "point": 8.5, "implied_pct": 51.3},
                },
                "rl": {
                    home: {"odds": 2.55, "point": -1.5, "implied_pct": 39.2},
                    away: {"odds": 1.58, "point":  1.5, "implied_pct": 63.3},
                },
            }
        },
    }
