"""Tests for odds.lib.movement."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from odds_math import decimal_to_implied
from snapshot_loader import GameRecord, Snapshot, ET, collect_game_timeline, load_snapshots_for_et_date
from movement import (
    FieldMovement,
    GameMovementReport,
    THRESHOLDS,
    TIER_ORDER,
    compute_game_movement,
    _tier_from_delta,
    _max_tier,
)


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ── decimal_to_implied (smoke) ────────────────────────────────────────────────

def test_decimal_to_implied_basic():
    assert decimal_to_implied(2.0) == 50.0
    assert decimal_to_implied(1.91) == 52.4
    assert decimal_to_implied(1.0) == 0.0
    assert decimal_to_implied(0.0) == 0.0
    assert decimal_to_implied(-1.0) == 0.0
    assert decimal_to_implied("invalid") == 0.0


# ── tier helpers ──────────────────────────────────────────────────────────────

def test_tier_from_delta():
    assert _tier_from_delta(0.5) == "quiet"
    assert _tier_from_delta(0.99) == "quiet"
    assert _tier_from_delta(1.0) == "watch"
    assert _tier_from_delta(2.5) == "watch"
    assert _tier_from_delta(3.0) == "significant"
    assert _tier_from_delta(4.9) == "significant"
    assert _tier_from_delta(5.0) == "major"
    assert _tier_from_delta(10.0) == "major"


def test_max_tier():
    assert _max_tier("quiet", "watch") == "watch"
    assert _max_tier("watch", "major") == "major"
    assert _max_tier("significant", "watch") == "significant"
    assert _max_tier("quiet", "quiet") == "quiet"


# ── compute_game_movement (real fixtures) ─────────────────────────────────────
# Fixture commence_utc 在 ET 4/27 → game_date_et = 2026-04-27

def test_tbr_cle_is_major_with_total_cross():
    """TBR@CLE: no-vig CLE ML 53.2→61.0 (+7.8pp) = major; Total 8.5→9.5 跨 9。

    註：raw delta 為 +8.4pp（54.1→62.5），no-vig 化後縮為 +7.8pp 但仍 ≥5 落 major。
    """
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]

    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)   # 22h 前
    report = compute_game_movement(timeline, now_utc)

    assert report.tier == "major"
    assert report.tier_downgraded is False
    ml_home = next(f for f in report.fields if f.field == "ml_home")
    assert ml_home.delta_pp >= 5.0          # tier=major 門檻
    assert ml_home.latest_value_raw > ml_home.anchor_value_raw   # raw 仍可比對
    assert any("跨越 key 9" in c for c in report.key_number_crosses)


def test_stl_pit_is_watch():
    """STL@PIT: PIT ML 56.2→57.5 (+1.3pp) = watch；最大 |delta_pp| 落 watch 帶。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "St. Louis Cardinals"][0]

    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.tier == "watch"
    assert report.key_number_crosses == []


def test_bos_tor_is_quiet():
    """BOS@TOR: 幾乎沒位移 → quiet。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Boston Red Sox"][0]

    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.tier == "quiet"


def test_thin_market_downgrades_tier():
    """latest snapshot 距 commence < 4h → tier 降一檔 + tier_downgraded=True。

    薄盤判斷基於 latest snapshot vs commence（市場流動性），不是 now_utc。
    """
    rec_anchor = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="08:00",
        ml_home_odds=1.85, ml_home_imp=54.1,
        ml_away_odds=2.10, ml_away_imp=47.6,
    )
    rec_latest = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="14:00",
        ml_home_odds=1.50, ml_home_imp=66.7,
        ml_away_odds=2.80, ml_away_imp=35.7,
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.is_thin_market is True
    assert report.tier_downgraded is True
    assert report.tier == "significant"


def test_thin_market_quiet_stays_quiet():
    """quiet 場薄盤不會降到負索引。"""
    rec_anchor = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="08:00",
        ml_home_imp=52.4, ml_away_imp=51.3,
    )
    rec_latest = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="14:00",
        ml_home_imp=52.5, ml_away_imp=51.2,
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.is_thin_market is True
    assert report.tier == "quiet"
    assert report.tier_downgraded is False


def test_thin_market_false_for_historical_data():
    """latest snapshot 取於 commence 之前但 now_utc 已過開球 → 不應視為薄盤。"""
    rec_anchor = _make_record(
        commence_utc_iso="2026-04-20T22:00:00Z",
        snap_time="00:00",
        ml_home_imp=52.4, ml_away_imp=51.3,
    )
    rec_latest = _make_record(
        commence_utc_iso="2026-04-20T22:00:00Z",
        snap_time="12:00",
        ml_home_imp=52.5, ml_away_imp=51.2,
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.is_thin_market is False
    assert report.tier_downgraded is False


def test_just_appeared_anchor_no_crash():
    """timeline 只 1 筆 → window_delta=0 / anchor_age=0、不 crash。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline([snapshots[1]], "2026-04-27")
    timeline = list(timelines.values())[0]
    assert len(timeline) == 1

    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.snapshot_count == 1
    assert report.anchor_age_hours == 0.0
    assert report.tier == "quiet"
    assert all(f.delta_pp == 0.0 for f in report.fields if f.field != "total_point")


def test_total_juice_shift_only():
    """同 point、僅 juice 變化 → 應落入相應 tier。"""
    rec_anchor = _make_record(over_imp=50.0, under_imp=50.0, snap_time="00:00")
    rec_latest = _make_record(over_imp=54.0, under_imp=46.0, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.tier == "significant"
    juice_over = next(f for f in report.fields if f.field == "total_juice_over")
    assert abs(juice_over.delta_pp - 4.0) < 0.05


def test_total_point_shift_drives_tier():
    """ML 完全不動，只有 total point 跳一整檔 → 至少 significant。"""
    rec_anchor = _make_record(total_point=8.5, snap_time="00:00")
    rec_latest = _make_record(total_point=9.5, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.tier in ("significant", "major")
    assert any("跨越 key 9" in c for c in report.key_number_crosses)


def test_total_point_no_key_cross():
    """8.5 → 8.0 移半檔但沒跨 key number → 沒 cross flag。"""
    rec_anchor = _make_record(total_point=8.5, snap_time="00:00")
    rec_latest = _make_record(total_point=8.0, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.key_number_crosses == []


def test_total_point_lands_on_key():
    """anchor 7.5 → latest 7.0 (降落到 key 7) → 應發出『降落 key 7』訊號。

    Sharp 含義：書商選擇停在 key 上承擔 push，與「穿過 key」不同但同樣是接盤訊號。
    """
    rec_anchor = _make_record(total_point=7.5, snap_time="00:00")
    rec_latest = _make_record(total_point=7.0, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert any("降落 key 7" in c for c in report.key_number_crosses)
    # land 不應同時被當成 cross
    assert not any("跨越" in c and "key 7" in c for c in report.key_number_crosses)


def test_total_point_leaves_key_no_signal():
    """anchor 7.0 → latest 7.5 (離開 key 7) → 不應發出任何 key 7 訊號。

    避免拆 signal 設計時的 over-trigger：書商離開 key magnet 的方向解讀不一致，保守處理。
    """
    rec_anchor = _make_record(total_point=7.0, snap_time="00:00")
    rec_latest = _make_record(total_point=7.5, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert not any("key 7" in c for c in report.key_number_crosses)


def test_total_point_crosses_key_emits_cross_signal():
    """anchor 7.5 → latest 6.5 (嚴格穿過 key 7) → 應發出『跨越 key 7』訊號（既有行為保留）。"""
    rec_anchor = _make_record(total_point=7.5, snap_time="00:00")
    rec_latest = _make_record(total_point=6.5, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert any("跨越 key 7" in c for c in report.key_number_crosses)
    # 嚴格穿過不應同時觸發 land 訊號
    assert not any("降落" in c for c in report.key_number_crosses)


def test_rl_price_flip_detected():
    """RL home_odds 從 < 2.0 跳到 > 2.0 → flag rl_price_flip。"""
    rec_anchor = _make_record(rl_home_odds=1.95, rl_away_odds=1.91, snap_time="00:00")
    rec_latest = _make_record(rl_home_odds=2.05, rl_away_odds=1.83, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert any("price flip" in c.lower() for c in report.key_number_crosses)


def test_juice_skipped_when_point_changes():
    """anc_point != lat_point → 不發出 total_juice_* fields（不同 prop，比較失真）。"""
    rec_anchor = _make_record(total_point=9.5, over_imp=49.0, under_imp=54.0, snap_time="00:00")
    rec_latest = _make_record(total_point=9.0, over_imp=54.6, under_imp=48.8, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    field_names = {f.field for f in report.fields}
    assert "total_juice_over" not in field_names
    assert "total_juice_under" not in field_names
    assert report.tier in ("watch", "quiet")


def test_rl_no_flip_same_side():
    """RL home_odds 1.85 → 1.65 同熱門側 → 無 flip。"""
    rec_anchor = _make_record(rl_home_odds=1.85, rl_away_odds=2.05, snap_time="00:00")
    rec_latest = _make_record(rl_home_odds=1.65, rl_away_odds=2.40, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert all("price flip" not in c.lower() for c in report.key_number_crosses)


def test_direction_label_format():
    """ML home implied +Xpp → label 應含縮寫 + 方向箭頭 + +pp + no-vig 標記。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    ml_home = next(f for f in report.fields if f.field == "ml_home")
    assert "CLE" in ml_home.direction_label
    assert "+" in ml_home.direction_label
    # F4: 百分比應明示 no-vig basis（避免讀者誤以為是 raw）
    assert "no-vig" in ml_home.direction_label


def test_total_juice_direction_label_marks_no_vig():
    """Total juice direction_label 應明示 no-vig basis（與 ML / RL label 對稱）。"""
    rec_anchor = _make_record(over_imp=50.0, under_imp=50.0, snap_time="00:00")
    rec_latest = _make_record(over_imp=54.0, under_imp=46.0, snap_time="04:00")
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    juice_over = next(f for f in report.fields if f.field == "total_juice_over")
    assert "no-vig" in juice_over.direction_label


def test_report_carries_commence_et():
    """GameMovementReport 應持有 commence_et，格式 'YYYY-MM-DD HH:MM ET'。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.commence_et.endswith(" ET")
    # ET 4/27 18:11 (= UTC 22:11)
    assert "2026-04-27 18:11 ET" == report.commence_et


# ── No-vig 專屬 cases ─────────────────────────────────────────────────────────

def test_no_vig_filters_out_juice_thickening():
    """雙邊 raw 各 +1pp（純 vig 加厚）→ no-vig delta ≈ 0，tier = quiet。

    驗證 no-vig 化能濾掉「莊家整體調 juice」這種無方向訊號的雜訊。
    """
    rec_anchor = _make_record(
        snap_time="00:00",
        ml_home_imp=51.0, ml_away_imp=51.0,    # raw sum 102
    )
    rec_latest = _make_record(
        snap_time="04:00",
        ml_home_imp=52.0, ml_away_imp=52.0,    # raw sum 104，雙邊各 +1pp
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    ml_home = next(f for f in report.fields if f.field == "ml_home")
    # no-vig anchor = 50/50；no-vig latest = 50/50 → delta ≈ 0
    assert abs(ml_home.delta_pp) < 0.5
    assert report.tier == "quiet"


def test_no_vig_preserves_asymmetric_move():
    """home raw +3pp、away raw 不動 → no-vig home 仍應推升至少 watch。

    驗證 no-vig 化保留真正的單邊位移訊號（非純 juice 變化）。
    """
    rec_anchor = _make_record(
        snap_time="00:00",
        ml_home_imp=50.0, ml_away_imp=50.0,    # 對稱起點
    )
    rec_latest = _make_record(
        snap_time="04:00",
        ml_home_imp=53.0, ml_away_imp=50.0,    # home +3pp、away 不動
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    ml_home = next(f for f in report.fields if f.field == "ml_home")
    # no-vig anchor = 50/50；no-vig latest = 53/103 ≈ 51.5 → delta ≈ +1.5pp
    assert ml_home.delta_pp >= 1.0
    assert ml_home.latest_value_raw == 53.0    # raw 仍保留
    assert report.tier in ("watch", "significant", "major")


# ── 輔助：構造單筆 GameRecord ────────────────────────────────────────────────

def _make_record(
    home: str = "Atlanta Braves",
    away: str = "New York Mets",
    commence_utc_iso: str = "2026-05-01T17:05:00Z",
    snap_time: str = "00:00",
    ml_home_odds: float = 1.91, ml_home_imp: float = 52.4,
    ml_away_odds: float = 1.95, ml_away_imp: float = 51.3,
    total_point: float = 8.5,
    over_imp: float = 50.0, under_imp: float = 50.0,
    over_odds: float = 1.91, under_odds: float = 1.95,
    rl_home_odds: float = 1.91, rl_home_imp: float = 52.4,
    rl_away_odds: float = 1.95, rl_away_imp: float = 51.3,
) -> GameRecord:
    snap_time_dt = datetime.strptime(f"2026-05-01 {snap_time}", "%Y-%m-%d %H:%M")
    commence_dt = datetime.fromisoformat(commence_utc_iso.replace("Z", "+00:00"))
    commence_et = commence_dt.astimezone(ET)
    pinnacle = {
        "title": "Pinnacle",
        "ml": {
            home: {"odds": ml_home_odds, "implied_pct": ml_home_imp},
            away: {"odds": ml_away_odds, "implied_pct": ml_away_imp},
        },
        "ou": {
            "Over":  {"odds": over_odds,  "point": total_point, "implied_pct": over_imp},
            "Under": {"odds": under_odds, "point": total_point, "implied_pct": under_imp},
        },
        "rl": {
            home: {"odds": rl_home_odds, "point": -1.5, "implied_pct": rl_home_imp},
            away: {"odds": rl_away_odds, "point":  1.5, "implied_pct": rl_away_imp},
        },
    }
    return GameRecord(
        game_key=(away, home, commence_utc_iso),
        away=away,
        home=home,
        commence_utc=commence_dt,
        commence_et_label=commence_et.strftime("%Y-%m-%d %H:%M ET"),
        pinnacle=pinnacle,
        snapshot_time_et=snap_time_dt,
        snapshot_time_et_label=snap_time_dt.strftime("%m-%d %H:%M"),
        game_date_et=commence_et.strftime("%Y-%m-%d"),
    )
