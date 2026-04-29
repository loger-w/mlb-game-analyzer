"""Tests for odds.lib.movement."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from odds_math import decimal_to_implied
from snapshot_loader import GameRecord, Snapshot, collect_game_timeline, load_snapshots_for_et_date
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

def test_tbr_cle_is_major_with_total_cross():
    """TBR@CLE: CLE ML 54.1→62.5 (+8.4pp) = major; Total 8.5→9.5 跨 9。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]

    # 把 now 設成遠在 commence 之前（不薄盤）
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)   # 22h 前
    report = compute_game_movement(timeline, now_utc)

    assert report.tier == "major"
    assert report.tier_downgraded is False
    # ML home 應有 +8.4pp 左右
    ml_home = next(f for f in report.fields if f.field == "ml_home")
    assert ml_home.delta_pp > 8.0
    # Total point cross
    assert any("跨越 key 9" in c for c in report.key_number_crosses)


def test_stl_pit_is_watch():
    """STL@PIT: PIT ML 56.2→57.5 (+1.3pp) = watch；最大 |delta_pp| 落 watch 帶。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "St. Louis Cardinals"][0]

    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.tier == "watch"
    # 不該觸發 key number cross
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
    # commence ET 17:00 = UTC 21:00；latest snapshot ET 14:00 = UTC 18:00；3h gap → thin
    rec_anchor = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="08:00",
        ml_home_odds=1.85, ml_home_imp=54.1,
        ml_away_odds=2.10, ml_away_imp=47.6,
    )
    rec_latest = _make_record(
        commence_utc_iso="2026-05-01T21:00:00Z",
        snap_time="14:00",   # 3h pre-commence
        ml_home_odds=1.50, ml_home_imp=66.7,   # +12.6pp → major
        ml_away_odds=2.80, ml_away_imp=35.7,
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)   # now 不影響 thin
    report = compute_game_movement(timeline, now_utc)

    assert report.is_thin_market is True
    assert report.tier_downgraded is True
    # 原本 major → 降到 significant
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
        snap_time="14:00",   # 3h pre-commence
        ml_home_imp=52.5, ml_away_imp=51.2,
    )
    timeline = [rec_anchor, rec_latest]
    now_utc = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    assert report.is_thin_market is True
    # quiet 沒得降，仍 quiet，但 tier_downgraded=False（因為沒實際降）
    assert report.tier == "quiet"
    assert report.tier_downgraded is False


def test_thin_market_false_for_historical_data():
    """latest snapshot 取於 commence 之前但 now_utc 已過開球 → 不應視為薄盤。

    此測試覆蓋：使用者在賽後分析過往日期，hours_to_game 為負，但 snap-to-commence 仍正、且寬。
    """
    rec_anchor = _make_record(
        commence_utc_iso="2026-04-20T22:00:00Z",
        snap_time="00:00",   # 18h pre-commence
        ml_home_imp=52.4, ml_away_imp=51.3,
    )
    rec_latest = _make_record(
        commence_utc_iso="2026-04-20T22:00:00Z",
        snap_time="12:00",   # 6h pre-commence — 不薄盤
        ml_home_imp=52.5, ml_away_imp=51.2,
    )
    timeline = [rec_anchor, rec_latest]
    # now_utc 設成幾天後（賽後），hours_to_game 會很負
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)
    assert report.is_thin_market is False
    assert report.tier_downgraded is False


def test_just_appeared_anchor_no_crash():
    """timeline 只 1 筆 → window_delta=0 / anchor_age=0、不 crash。

    snapshots[1] = 04-27 00:00 ET（第一個含 04-27 場次的 snapshot）；
    snapshots[0] 現在是 04-26 20:00 ET（loader 改為按 UTC 排序返回全部）。
    """
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
    # 構造兩 record：anchor over implied 50.0 / latest over implied 54.0 → +4pp = significant
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
    # 應該完全沒有 total_juice_over / total_juice_under field
    field_names = {f.field for f in report.fields}
    assert "total_juice_over" not in field_names
    assert "total_juice_under" not in field_names
    # tier 不該因 juice 機械效應升到 major
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
    """ML home implied +Xpp → label 應含縮寫 + 方向箭頭 + +pp。"""
    snapshots = load_snapshots_for_et_date("2026-04-27", FIXTURES)
    timelines = collect_game_timeline(snapshots, "2026-04-27")
    timeline = [t for k, t in timelines.items() if k[0] == "Tampa Bay Rays"][0]
    now_utc = datetime(2026, 4, 27, 0, 0, tzinfo=timezone.utc)
    report = compute_game_movement(timeline, now_utc)

    ml_home = next(f for f in report.fields if f.field == "ml_home")
    # CLE 是 home，移動方向往 home → 標籤應含 "CLE" 與 "+8" 前段
    assert "CLE" in ml_home.direction_label
    assert "+" in ml_home.direction_label


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
        commence_utc=datetime.fromisoformat(commence_utc_iso.replace("Z", "+00:00")),
        commence_et_label=f"2026-05-01 13:05 ET",
        pinnacle=pinnacle,
        snapshot_time_et=snap_time_dt,
        snapshot_time_et_label=snap_time,
    )
