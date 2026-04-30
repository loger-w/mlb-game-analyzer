"""Tests for odds.lib.md_renderer (ET-native)."""
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from snapshot_loader import collect_game_timeline, load_snapshots_for_et_date, ET
from movement import compute_game_movement, GameMovementReport
from md_renderer import render


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _build_reports(now_utc: datetime = None) -> list[GameMovementReport]:
    if now_utc is None:
        now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    # ET 4/27 場次的 game_date_et = 2026-04-27
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    return [compute_game_movement(t, now_utc) for t in timelines.values()]


def test_render_smoke_all_tiers():
    """fixtures 涵蓋 major / watch / quiet 三 tier；輸出應含對應 header。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET (TW 04-27 16:00)",
    )
    assert "Smart Money Tracker — 2026-04-27 (ET)" in md
    assert "## 🔥 Major" in md
    assert "## 🔵 Watch" in md
    assert "## ⚪ Quiet" in md
    assert "ℹ️ Anchor Notes" in md
    assert "## 🟡 Significant" not in md


def test_render_skips_empty_tiers():
    """全部都是 quiet → 只出現 quiet header。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    quiet_timeline = [t for k, t in timelines.items() if k[0] == "Boston Red Sox"][0]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    rep = compute_game_movement(quiet_timeline, now_utc)
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=[rep],
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "## 🔥 Major" not in md
    assert "## 🟡 Significant" not in md
    assert "## 🔵 Watch" not in md
    assert "## ⚪ Quiet" in md


def test_render_direction_arrow_present():
    """major 場應在輸出中含 direction_label 文字（例 '→ CLE'）。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "→ CLE" in md or "CLE" in md
    assert "跨越 key 9" in md


def test_render_thin_market_marker():
    """is_thin_market=True 場應顯示 [薄盤]。"""
    home, away = "Boston Red Sox", "Toronto Blue Jays"
    commence_iso = "2026-05-01T21:00:00Z"   # ET 17:00 → game_date_et = 2026-05-01
    pinnacle_anchor = _pinnacle(home, away, ml_home_imp=54.1, ml_away_imp=47.6)
    pinnacle_latest = _pinnacle(home, away, ml_home_imp=62.5, ml_away_imp=40.0)
    anchor = _make_record(
        away=away, home=home, commence_iso=commence_iso,
        pinnacle=pinnacle_anchor,
        snap_et=datetime(2026, 5, 1, 8, 0),
    )
    latest = _make_record(
        away=away, home=home, commence_iso=commence_iso,
        pinnacle=pinnacle_latest,
        snap_et=datetime(2026, 5, 1, 14, 0),   # 3h pre-commence
    )
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    rep = compute_game_movement([anchor, latest], now_utc)
    md = render(
        et_date="2026-05-01",
        snapshot_count=2,
        snapshot_times_et=["05-01 08:00", "05-01 14:00"],
        reports=[rep],
        rendered_at="2026-05-01 14:00 ET",
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


def _make_record(away, home, commence_iso, pinnacle, snap_et):
    """Helper:構造帶完整 ET 欄位的 GameRecord。"""
    from snapshot_loader import GameRecord
    commence_utc = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    commence_et = commence_utc.astimezone(ET)
    return GameRecord(
        game_key=(away, home, commence_iso),
        away=away, home=home,
        commence_utc=commence_utc,
        commence_et_label=commence_et.strftime("%Y-%m-%d %H:%M ET"),
        pinnacle=pinnacle,
        snapshot_time_et=snap_et,
        snapshot_time_et_label=snap_et.strftime("%m-%d %H:%M"),
        game_date_et=commence_et.strftime("%Y-%m-%d"),
    )


def test_render_time_series_table_rows():
    """v2 主要場應有 7 欄時間軸表格，行數 = snapshot 數；cell 含 MM-DD 日期前綴。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "| ET 時間 |" in md
    assert "| Over |" in md
    assert "| Under |" in md
    assert "| Total |" not in md
    # snapshot_time_et_label 包含 MM-DD 前綴
    assert re.search(r"\| 04-27 \d{2}:\d{2} \|", md)
    # Over / Under cell 應含 "@ <point>"
    assert " @ " in md


def test_render_cell_has_date_prefix():
    """所有時間軸 cell 的時間欄都應符合 'MM-DD HH:MM' 格式。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert re.search(r"\| \d{2}-\d{2} \d{2}:\d{2} \| ", md)


def test_render_no_reports_produces_empty_message():
    """reports 空 list → 寫一段「無比賽」訊息 + AI footnote 仍輸出，不 crash。"""
    md = render(
        et_date="2026-04-27",
        snapshot_count=0,
        snapshot_times_et=[],
        reports=[],
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "Smart Money Tracker" in md
    assert "無" in md or "no games" in md.lower()
    assert "解讀說明(給 AI)" in md


def test_render_quiet_tier_has_full_detail():
    """v2: quiet tier 場應有 timeline 表格與 headline 位移行。"""
    reports = _build_reports()
    quiet_only = [r for r in reports if r.tier == "quiet"]
    assert len(quiet_only) > 0, "fixture 應至少有一場 quiet"
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=quiet_only,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "## ⚪ Quiet" in md
    assert "| ET 時間 |" in md
    assert "位移" in md


def test_render_ai_footnote_present():
    """v2: md 末段含「解讀說明(給 AI)」與關鍵解讀規則；時區一律 ET。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "解讀說明(給 AI)" in md
    assert "5pp" in md
    assert "3pp" in md
    assert "1pp" in md
    assert "7 / 9 / 11" in md
    assert "vig" in md.lower()
    assert "薄盤" in md
    assert "Pinnacle" in md
    # Footnote 時區應為 ET
    assert "ET" in md
    # 不該再出現 TW 字眼
    assert "UTC+8" not in md
    assert "TW (UTC" not in md


# ── No-vig 雙顯示 + footnote 更新 ────────────────────────────────────────────

def test_render_cell_shows_raw_and_no_vig():
    """timeline cell 應同時顯示 raw% 與 no-vig%，例 '1.91 (52.4% / 50.9%)'。"""
    reports = _build_reports()
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    assert re.search(r"\(\d+\.\d% / \d+\.\d%\)", md), "cell 應有 raw / no-vig 雙百分比"


def test_render_footnote_updated_for_no_vig():
    """footnote 應移除『vig < 0.5pp 可直接比較』錯誤陳述，改寫為 no-vig basis。"""
    md = render(
        et_date="2026-04-27",
        snapshot_count=0,
        snapshot_times_et=[],
        reports=[],
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "可直接以 pp 比較" not in md
    assert "no-vig" in md.lower()


def test_render_compresses_same_day_snapshot_times():
    """cover line：同日多個 snapshot 時間 → 第一個含 MM-DD，後續只顯 HH:MM。"""
    md = render(
        et_date="2026-04-29",
        snapshot_count=3,
        snapshot_times_et=["04-28 22:00", "04-29 00:00", "04-29 03:00"],
        reports=[],
        rendered_at="2026-04-29 03:00 ET",
    )
    # 三筆不同日 → 完整顯示（壓縮只在同日才生效）
    assert "04-28 22:00" in md and "04-29 00:00" in md and "04-29 03:00" in md


# ── F6: anchor 跨 ET 日描述修正 ────────────────────────────────────────────────

def test_footnote_no_longer_claims_no_cross_day():
    """Anchor footnote 應移除『0 天回溯(不跨日)』錯誤陳述，改說可跨 ET 日。

    舊文字與實作矛盾：collect_game_timeline 實際會跨日納入 snapshot（見 04-29 報告）。
    """
    md = render(
        et_date="2026-04-27",
        snapshot_count=0,
        snapshot_times_et=[],
        reports=[],
        rendered_at="2026-04-27 04:00 ET",
    )
    assert "0 天回溯" not in md
    assert "不跨日" not in md
    assert "可跨 ET 日" in md


def test_anchor_notes_flags_cross_day_anchor():
    """timeline anchor 跨 ET 日 → anchor notes 應提到「跨 ET 日 anchor」。"""
    home, away = "Atlanta Braves", "Detroit Tigers"
    commence_iso = "2026-05-02T16:16:00Z"   # ET 12:16, game_date_et = 2026-05-02
    pinnacle_anchor = _pinnacle(home, away, ml_home_imp=53.5, ml_away_imp=49.0)
    pinnacle_latest = _pinnacle(home, away, ml_home_imp=54.9, ml_away_imp=47.4)
    anchor = _make_record(
        away=away, home=home, commence_iso=commence_iso,
        pinnacle=pinnacle_anchor,
        snap_et=datetime(2026, 5, 1, 22, 0),    # ET 5/1 22:00 — 跨日於 game_date 5/2
    )
    latest = _make_record(
        away=away, home=home, commence_iso=commence_iso,
        pinnacle=pinnacle_latest,
        snap_et=datetime(2026, 5, 2, 0, 0),
    )
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    rep = compute_game_movement([anchor, latest], now_utc)
    md = render(
        et_date="2026-05-02",
        snapshot_count=2,
        snapshot_times_et=["05-01 22:00", "05-02 00:00"],
        reports=[rep],
        rendered_at="2026-05-02 00:00 ET",
    )
    assert "跨 ET 日" in md


def test_anchor_notes_no_cross_day_flag_when_same_day():
    """anchor 與 game_date_et 同日 → anchor notes 不應出現「跨 ET 日」字眼。"""
    reports = _build_reports()    # fixtures 04-27，anchor 與 game date 同日
    md = render(
        et_date="2026-04-27",
        snapshot_count=2,
        snapshot_times_et=["04-27 00:00", "04-27 04:00"],
        reports=reports,
        rendered_at="2026-04-27 04:00 ET",
    )
    # anchor notes 段不應 flag 跨日（footnote 仍會解釋「可跨 ET 日」是另一回事）
    anchor_notes_section = md.split("ℹ️ Anchor Notes")[1].split("---")[0]
    assert "跨 ET 日 anchor" not in anchor_notes_section
