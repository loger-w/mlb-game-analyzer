"""計算 Pinnacle 線位移動：snapshot-to-snapshot delta、tier 分類、key number 跨越。

設計重點：
- Tier 由 max(|delta_pp|) on ML/RL/Total juice 的 6 個 pp-fields 決定
- Total point 位移獨立計算：≥1.0 runs → 至少 significant；≥0.5 → 至少 watch
- Key number 跨越（Total 7/9/11）+ RL price flip（odds 跨 2.0）為旗標，獨立於 tier
- hours_to_game < 4 → tier 降一檔（quiet 場除外）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from snapshot_loader import GameKey, GameRecord


# ── 常數 ──────────────────────────────────────────────────────────────────────

THRESHOLDS = {"major": 5.0, "significant": 3.0, "watch": 1.0}   # |delta_pp|
TIER_ORDER = ["quiet", "watch", "significant", "major"]
THIN_MARKET_HOURS = 4.0

KEY_NUMBERS_TOTAL = (7, 9, 11)   # MLB Total key numbers (Boyd's Bets / 文獻)

# 隊名 → 縮寫（與 scripts/predict.py 同步；新模組不 import scripts/）
TEAM_ABBREV = {
    "New York Yankees": "NYY", "New York Mets": "NYM", "Boston Red Sox": "BOS",
    "Los Angeles Dodgers": "LAD", "Los Angeles Angels": "LAA", "Houston Astros": "HOU",
    "Atlanta Braves": "ATL", "Philadelphia Phillies": "PHI", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "St. Louis Cardinals": "STL", "Milwaukee Brewers": "MIL",
    "Pittsburgh Pirates": "PIT", "Arizona Diamondbacks": "ARI", "Colorado Rockies": "COL",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB", "Toronto Blue Jays": "TOR",
    "Minnesota Twins": "MIN", "Kansas City Royals": "KC", "Detroit Tigers": "DET",
    "Cleveland Guardians": "CLE", "Seattle Mariners": "SEA", "Athletics": "OAK",
    "Texas Rangers": "TEX", "Miami Marlins": "MIA", "Washington Nationals": "WSH",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FieldMovement:
    field: str                    # ml_home / ml_away / total_point / total_juice_over / total_juice_under / rl_home / rl_away
    anchor_value: float           # 隱含勝率 pp，或 total_point 的 runs
    latest_value: float
    delta_pp: float               # 對 total_point 為 runs，其餘為 pp
    direction_label: str          # 例 "→ CLE +8.4pp" / "Total 8.5 → 9.5"


@dataclass
class GameMovementReport:
    game_key: GameKey
    away: str
    home: str
    commence_tw: str
    hours_to_game: float
    is_thin_market: bool
    snapshot_count: int
    anchor_age_hours: float
    fields: list[FieldMovement]
    timeline: list[GameRecord] = field(default_factory=list)
    key_number_crosses: list[str] = field(default_factory=list)
    tier: str = "quiet"
    tier_downgraded: bool = False


# ── Tier 工具 ──────────────────────────────────────────────────────────────────

def _tier_from_delta(delta_pp: float) -> str:
    a = abs(delta_pp)
    if a >= THRESHOLDS["major"]:       return "major"
    if a >= THRESHOLDS["significant"]: return "significant"
    if a >= THRESHOLDS["watch"]:       return "watch"
    return "quiet"


def _max_tier(*tiers: str) -> str:
    return max(tiers, key=lambda t: TIER_ORDER.index(t))


def _downgrade_tier(tier: str) -> tuple[str, bool]:
    """tier 降一檔；若已是 quiet 則保持 quiet 並回 (tier, False)。"""
    idx = TIER_ORDER.index(tier)
    if idx == 0:
        return tier, False
    return TIER_ORDER[idx - 1], True


# ── 隱含勝率讀取（容錯）──────────────────────────────────────────────────────

def _get_implied(odds_dict: dict) -> float:
    """讀 odds dict 的 implied_pct；缺則從 odds 計算。0 / 缺 → 0.0。"""
    imp = odds_dict.get("implied_pct")
    if imp is not None:
        try:
            return float(imp)
        except (TypeError, ValueError):
            pass
    odds = odds_dict.get("odds")
    try:
        d = float(odds)
    except (TypeError, ValueError):
        return 0.0
    if d <= 1.0:
        return 0.0
    return round(100.0 / d, 1)


# ── 主計算 ────────────────────────────────────────────────────────────────────

def compute_game_movement(
    timeline: list[GameRecord],
    now_utc: datetime,
) -> GameMovementReport:
    """anchor = timeline[0]，latest = timeline[-1]。"""
    if not timeline:
        raise ValueError("timeline must not be empty")

    anchor = timeline[0]
    latest = timeline[-1]
    home, away = latest.home, latest.away

    fields: list[FieldMovement] = []

    # ── ML：home / away 隱含勝率 ──
    anc_ml = anchor.pinnacle.get("ml", {}) or {}
    lat_ml = latest.pinnacle.get("ml", {}) or {}
    for side, team in (("home", home), ("away", away)):
        anc = _get_implied(anc_ml.get(team, {}))
        lat = _get_implied(lat_ml.get(team, {}))
        delta = round(lat - anc, 1)
        fields.append(FieldMovement(
            field=f"ml_{side}",
            anchor_value=anc,
            latest_value=lat,
            delta_pp=delta,
            direction_label=_implied_direction_label(team, anc, lat, delta),
        ))

    # ── Total：point + juice ──（兩端皆有有效 Over/point 才發出）
    anc_ou = anchor.pinnacle.get("ou", {}) or {}
    lat_ou = latest.pinnacle.get("ou", {}) or {}
    anc_over = anc_ou.get("Over") or {}
    lat_over = lat_ou.get("Over") or {}
    anc_point_raw = anc_over.get("point")
    lat_point_raw = lat_over.get("point")
    has_total = anc_point_raw is not None and lat_point_raw is not None
    if has_total:
        anc_point = float(anc_point_raw)
        lat_point = float(lat_point_raw)
        fields.append(FieldMovement(
            field="total_point",
            anchor_value=anc_point,
            latest_value=lat_point,
            delta_pp=round(lat_point - anc_point, 2),   # 注意：runs，不是 pp
            direction_label=_total_point_label(anc_point, lat_point),
        ))
        # Juice 比較只在 point 相同（且兩端皆有 Over/Under）時有意義
        if (
            anc_point == lat_point
            and anc_ou.get("Over") and anc_ou.get("Under")
            and lat_ou.get("Over") and lat_ou.get("Under")
        ):
            for side in ("Over", "Under"):
                anc = _get_implied(anc_ou.get(side, {}))
                lat = _get_implied(lat_ou.get(side, {}))
                delta = round(lat - anc, 1)
                fields.append(FieldMovement(
                    field=f"total_juice_{side.lower()}",
                    anchor_value=anc,
                    latest_value=lat,
                    delta_pp=delta,
                    direction_label=f"{side} {anc:.1f}% → {lat:.1f}% ({delta:+.1f}pp)",
                ))
    else:
        anc_point = lat_point = None   # 用於下游 tier / cross 判斷

    # ── RL：home / away 隱含勝率 ──（兩端皆有同隊 RL 才發出）
    anc_rl = anchor.pinnacle.get("rl", {}) or {}
    lat_rl = latest.pinnacle.get("rl", {}) or {}
    for side, team in (("home", home), ("away", away)):
        anc_team = anc_rl.get(team) or {}
        lat_team = lat_rl.get(team) or {}
        if not anc_team or not lat_team:
            continue
        anc = _get_implied(anc_team)
        lat = _get_implied(lat_team)
        delta = round(lat - anc, 1)
        fields.append(FieldMovement(
            field=f"rl_{side}",
            anchor_value=anc,
            latest_value=lat,
            delta_pp=delta,
            direction_label=_implied_direction_label(team, anc, lat, delta, prefix="RL"),
        ))

    # ── Tier：以 6 個 pp-fields 的 max |delta| 為主 ──
    pp_fields = [f for f in fields if f.field != "total_point"]
    max_pp = max((abs(f.delta_pp) for f in pp_fields), default=0.0)
    tier = _tier_from_delta(max_pp)

    # ── Total point 位移額外推 tier（僅在兩端有有效 point 時）──
    crosses: list[str] = []
    if has_total:
        point_shift = abs(lat_point - anc_point)
        if point_shift >= 1.0:
            tier = _max_tier(tier, "significant")
        elif point_shift >= 0.5:
            tier = _max_tier(tier, "watch")
        # ── Key number 跨越（Total）──
        for k in KEY_NUMBERS_TOTAL:
            lo, hi = min(anc_point, lat_point), max(anc_point, lat_point)
            if lo < k <= hi and lo != hi:
                arrow = f"{anc_point} → {lat_point}"
                crosses.append(f"Total {arrow} 跨越 key {k}")

    # ── RL price flip（odds 跨 2.0）──
    for side, team in (("home", home), ("away", away)):
        anc_odds = (anc_rl.get(team) or {}).get("odds")
        lat_odds = (lat_rl.get(team) or {}).get("odds")
        if anc_odds is None or lat_odds is None:
            continue
        if (anc_odds < 2.0) != (lat_odds < 2.0):
            crosses.append(
                f"RL price flip: {_abbr(team)} {anc_odds:.2f} → {lat_odds:.2f}"
            )

    # ── Hours to game ──
    # 「薄盤」定義 = latest snapshot 抓取時，距離 commence 不到 4h（snapshot 當下的市場狀態）
    # 不用 now_utc，因為對歷史分析（已過 commence 的日期）會誤判全部為薄盤
    if latest.commence_utc.tzinfo is None:
        commence_utc = latest.commence_utc.replace(tzinfo=timezone.utc)
    else:
        commence_utc = latest.commence_utc
    if now_utc.tzinfo is None:
        now_utc_aware = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc_aware = now_utc
    hours_to_game = (commence_utc - now_utc_aware).total_seconds() / 3600.0
    # 薄盤判斷使用 latest snapshot 距 commence
    snap_et_as_utc = latest.snapshot_time_et.replace(tzinfo=timezone(timedelta(hours=-4))).astimezone(timezone.utc)
    hours_snap_to_commence = (commence_utc - snap_et_as_utc).total_seconds() / 3600.0

    # ── 薄盤降級（基於 latest snapshot 與 commence 的 ET-差，避免歷史分析誤判）──
    is_thin = 0.0 <= hours_snap_to_commence < THIN_MARKET_HOURS
    tier_downgraded = False
    if is_thin:
        tier, tier_downgraded = _downgrade_tier(tier)

    # ── Anchor age ──
    anchor_age_hours = (latest.snapshot_time_et - anchor.snapshot_time_et).total_seconds() / 3600.0

    return GameMovementReport(
        game_key=latest.game_key,
        away=away,
        home=home,
        commence_tw=latest.commence_tw_label,
        hours_to_game=round(hours_to_game, 1),
        is_thin_market=is_thin,
        snapshot_count=len(timeline),
        anchor_age_hours=round(anchor_age_hours, 1),
        fields=fields,
        timeline=list(timeline),
        key_number_crosses=crosses,
        tier=tier,
        tier_downgraded=tier_downgraded,
    )


# ── Direction label helpers ───────────────────────────────────────────────────

def _abbr(team: str) -> str:
    """3-字母縮寫；查不到則回原名最後一字。"""
    return TEAM_ABBREV.get(team, team.split()[-1][:3].upper())


def _implied_direction_label(team: str, anc: float, lat: float, delta: float, prefix: str = "") -> str:
    """例 '→ CLE +8.4pp' / 'RL → CLE +4.3pp'。"""
    abbr = _abbr(team)
    arrow = "→"
    sign_str = f"{delta:+.1f}pp"
    label = f"{arrow} {abbr} {sign_str} ({anc:.1f}% → {lat:.1f}%)"
    if prefix:
        label = f"{prefix} {label}"
    return label


def _total_point_label(anc: float, lat: float) -> str:
    """例 'Total 8.5 → 9.5'。"""
    return f"Total {anc} → {lat}"
