"""Tests for odds.lib.md_renderer."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from snapshot_loader import collect_game_timeline, load_snapshots_for_et_date
from movement import compute_game_movement, GameMovementReport
from md_renderer import render


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _build_reports(now_utc: datetime = None) -> list[GameMovementReport]:
    if now_utc is None:
        now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    return [compute_game_movement(t, now_utc) for t in timelines.values()]


def test_render_smoke_all_tiers():
    """fixtures 涵蓋 major / watch / quiet 三 tier；輸出應含對應 header。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["00:00", "04:00"],
        reports=reports,
        rendered_at_et="2026-04-27 04:00 ET (TW 16:00)",
    )
    assert "Smart Money Tracker — 2026-04-27 (ET)" in md
    assert "🔥 Major" in md
    assert "🔵 Watch" in md
    assert "⚪ Quiet" in md
    assert "ℹ️ Anchor Notes" in md
    # 沒 significant tier → 該 header 不出現
    assert "🟡 Significant" not in md


def test_render_skips_empty_tiers():
    """全部都是 quiet → 只出現 quiet header（其餘 tier 略）。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    # 只挑 BOS@TOR（quiet 場）
    quiet_timeline = [t for k, t in timelines.items() if k[0] == "Boston Red Sox"][0]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    rep = compute_game_movement(quiet_timeline, now_utc)
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["00:00", "04:00"],
        reports=[rep],
        rendered_at_et="2026-04-27 04:00 ET (TW 16:00)",
    )
    assert "🔥 Major" not in md
    assert "🟡 Significant" not in md
    assert "🔵 Watch" not in md
    assert "⚪ Quiet" in md


def test_render_direction_arrow_present():
    """major 場應在輸出中含 direction_label 文字（例 '→ CLE'）。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["00:00", "04:00"],
        reports=reports,
        rendered_at_et="2026-04-27 04:00 ET (TW 16:00)",
    )
    # CLE 是 major 的 home
    assert "→ CLE" in md or "CLE" in md
    # Total cross 標記
    assert "跨越 key 9" in md


def test_render_thin_market_marker():
    """is_thin_market=True 場應顯示 [薄盤]。"""
    # 構造一個 latest snapshot 距 commence < 4h 的 timeline
    from snapshot_loader import GameRecord
    home, away = "Boston Red Sox", "Toronto Blue Jays"
    commence_iso = "2026-05-01T21:00:00Z"
    pinnacle_anchor = _pinnacle(home, away, ml_home_imp=54.1, ml_away_imp=47.6)
    pinnacle_latest = _pinnacle(home, away, ml_home_imp=62.5, ml_away_imp=40.0)   # major
    anchor = GameRecord(
        game_key=(away, home, commence_iso),
        away=away, home=home,
        commence_utc=datetime.fromisoformat(commence_iso.replace("Z", "+00:00")),
        commence_et_label="2026-05-01 17:00 ET",
        pinnacle=pinnacle_anchor,
        snapshot_time_et=datetime(2026, 5, 1, 8, 0),
        snapshot_time_et_label="08:00",
    )
    latest = GameRecord(
        game_key=(away, home, commence_iso),
        away=away, home=home,
        commence_utc=datetime.fromisoformat(commence_iso.replace("Z", "+00:00")),
        commence_et_label="2026-05-01 17:00 ET",
        pinnacle=pinnacle_latest,
        snapshot_time_et=datetime(2026, 5, 1, 14, 0),   # 3h pre-commence
        snapshot_time_et_label="14:00",
    )
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    rep = compute_game_movement([anchor, latest], now_utc)
    md = render(
        et_date="2026-05-01",
        snapshot_count=2,
        snapshot_times_et=["08:00", "14:00"],
        reports=[rep],
        rendered_at_et="2026-05-01 14:00 ET",
    )
    assert "[薄盤]" in md


def _pinnacle(home: str, away: str, ml_home_imp: float, ml_away_imp: float) -> dict:
    return {
        "title": "Pinnacle",
        "ml": {
            home: {"odds": round(100/ml_home_imp, 2), "implied_pct": ml_home_imp},
            away: {"odds": round(100/ml_away_imp, 2), "implied_pct": ml_away_imp},
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


def test_render_time_series_table_rows():
    """主要場應有時間軸表格，行數 = snapshot 數。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["00:00", "04:00"],
        reports=reports,
        rendered_at_et="2026-04-27 04:00 ET",
    )
    # 表格 header 應含 ET 時間 / Total
    assert "| ET 時間 |" in md
    # major 場（CLE）有 2 row
    assert "| 00:00 |" in md
    assert "| 04:00 |" in md


def test_render_no_reports_produces_empty_message():
    """reports 空 list → 寫一段「無比賽」訊息，不 crash。"""
    md = render(
        et_date="2026-04-27",
        snapshot_count=0,
        snapshot_times_et=[],
        reports=[],
        rendered_at_et="2026-04-27 04:00 ET",
    )
    assert "Smart Money Tracker" in md
    # 應有「無資料」/「無比賽」字樣
    assert "無" in md or "no games" in md.lower()
