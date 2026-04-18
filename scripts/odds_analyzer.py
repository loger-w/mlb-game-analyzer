#!/usr/bin/env python3
"""MLB Odds Analyzer — 盤口價值計算（EV + Kelly Criterion）"""

import argparse
import json
import math
import re
import sys


# ── 格式解析 ──────────────────────────────────────────────

def parse_bet_format(s: str) -> tuple[int, int]:
    """解析 'X-Y' 或 'X+Y' 格式的盤口線。
    '9-20' → (base=9, split_pct=20)
    '8+50' → (base=8, split_pct=50)
    '9'    → (base=9, split_pct=50)  純整數 → 預設標準四分球 50/50
    '9.5'  → ValueError，直線用 --total 傳入
    """
    s = str(s).strip()
    m = re.fullmatch(r"(\d+)[+\-](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if re.fullmatch(r"\d+", s):
        return int(s), 50
    raise ValueError(
        f"無法解析盤口格式 '{s}'。"
        "範例：'9-20'、'8+50'、'9'（純直線請用 --total）"
    )


# ── 機率估算（常態分佈近似）──────────────────────────────

_MLB_TOTAL_STD = 4.5  # 對齊 reference/prediction.md D2/D5 紀律（原 3.5 為既有 bug）


def _normal_cdf(x: float, mean: float, std: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2))))


def _p_at_most(n: int, mean: float) -> float:
    """P(total <= n)"""
    return _normal_cdf(n + 0.5, mean, _MLB_TOTAL_STD)


def _p_exactly(n: int, mean: float) -> float:
    """P(total == n)"""
    return _normal_cdf(n + 0.5, mean, _MLB_TOTAL_STD) - _normal_cdf(n - 0.5, mean, _MLB_TOTAL_STD)


def _p_at_least(n: int, mean: float) -> float:
    """P(total >= n)"""
    return 1.0 - _normal_cdf(n - 0.5, mean, _MLB_TOTAL_STD)


# ─────────────────────────────────────────────────────────

def ml_to_implied_prob(ml: int) -> float:
    """Moneyline → 隱含勝率"""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def hk_to_american(hk: float) -> int:
    """HK 賠率（獲利倍率）→ American Moneyline"""
    if hk >= 1.0:
        return int(round(hk * 100))   # 1.24 → +124
    else:
        return int(round(-100 / hk))  # 0.65 → -154


def american_to_hk(ml: int) -> float:
    """American Moneyline → HK 賠率"""
    if ml > 0:
        return ml / 100
    else:
        return 100 / abs(ml)


def decimal_to_american(dec: float) -> int:
    """Decimal odds → American moneyline."""
    if dec <= 1.0:
        raise ValueError(f"Invalid decimal odds: {dec}")
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def p_margin_ge_2_given_win(favorite_ml: int) -> float:
    """P(margin >= 2 | win)，對齊 reference/prediction.md 的 Run Line -1.5 機率表。"""
    ml = abs(favorite_ml)
    if ml <= 130:
        return 0.59
    if ml <= 170:
        return 0.615
    if ml <= 220:
        return 0.65
    return 0.695


def calc_ev(model_prob: float, ml: int) -> float:
    """計算期望值（Expected Value）"""
    if ml > 0:
        payout = ml / 100
    else:
        payout = 100 / abs(ml)
    ev = (model_prob * payout) - ((1 - model_prob) * 1)
    return round(ev * 100, 2)


def calc_kelly(model_prob: float, ml: int) -> float:
    """Kelly Criterion 建議注碼比例"""
    if ml > 0:
        odds = ml / 100
    else:
        odds = 100 / abs(ml)
    if odds <= 0:
        return 0
    kelly = (model_prob * (odds + 1) - 1) / odds
    return round(max(0, kelly) * 100, 2)


def calc_fractional_kelly(
    model_prob: float,
    ml: int,
    divisor: int = 4,
    cap_pct: float = 3.0,
    unit_size_pct: float = 1.0,
) -> dict:
    """Fractional Kelly with hard cap + unit conversion.

    Args:
        model_prob: 模型估計勝率 (0.0-1.0)
        ml: American moneyline (正數或負數)
        divisor: Kelly 分數係數（4 = quarter）
        cap_pct: 每注上限（% of bankroll，3.0 = 3%）
        unit_size_pct: 1 單位代表幾 % bankroll（1.0 = 1u = 1%）

    Returns:
        {raw_kelly_pct, fractional_pct, capped_pct, units}
        無 edge 時全部 0（不是 None — 0 是合法的「不下注」訊號）。
    """
    raw = calc_kelly(model_prob, ml)          # already returns 0 if negative
    fractional = round(raw / divisor, 4)
    capped = round(min(fractional, cap_pct), 4)
    # units：以 unit_size_pct 為 1u，round 到最近 0.5
    units = round(capped / unit_size_pct * 2) / 2 if unit_size_pct > 0 else 0.0
    return {
        "raw_kelly_pct": raw,
        "fractional_pct": fractional,
        "capped_pct": capped,
        "units": units,
    }


def get_stars_ml(prob_diff: float) -> int:
    """根據 ML 勝率差距決定推薦星級（reference ML 星級表）"""
    abs_diff = abs(prob_diff)
    if abs_diff >= 15:
        return 5
    elif abs_diff >= 10:
        return 4
    elif abs_diff >= 5:
        return 3
    elif abs_diff >= 2:
        return 2
    else:
        return 1


def get_stars_ou(run_diff: float) -> int:
    """G1: 根據 O/U run 差距決定推薦星級（對齊 reference/prediction.md）

    < 1.5 run = 不推薦（SD ≈ 4.5，在噪音範圍）
    1.5-2.0 = ⭐⭐⭐
    2.0-3.0 = ⭐⭐⭐⭐
    > 3.0   = ⭐⭐⭐⭐⭐
    """
    abs_diff = abs(run_diff)
    if abs_diff > 3.0:
        return 5
    elif abs_diff >= 2.0:
        return 4
    elif abs_diff >= 1.5:
        return 3
    else:
        return 0  # 不推薦


def analyze_moneyline(
    home_ml: int,
    away_ml: int,
    model_win_pct: float,
    kelly_params: dict = None,
) -> dict:
    """分析 Moneyline 盤口"""
    home_implied = ml_to_implied_prob(home_ml)
    away_implied = ml_to_implied_prob(away_ml)

    home_ev = calc_ev(model_win_pct, home_ml)
    away_ev = calc_ev(1 - model_win_pct, away_ml)

    home_kelly = calc_kelly(model_win_pct, home_ml)
    away_kelly = calc_kelly(1 - model_win_pct, away_ml)

    # 推薦方向：取 EV 較高的一方
    if home_ev > away_ev:
        direction = "HOME"
        best_ev = home_ev
        best_kelly = home_kelly
        prob_diff = (model_win_pct - home_implied) * 100
        kelly_prob = model_win_pct
        kelly_ml = home_ml
    else:
        direction = "AWAY"
        best_ev = away_ev
        best_kelly = away_kelly
        prob_diff = ((1 - model_win_pct) - away_implied) * 100
        kelly_prob = 1 - model_win_pct
        kelly_ml = away_ml

    stars = get_stars_ml(prob_diff)

    # Fractional Kelly
    kp = kelly_params or {}
    kf = calc_fractional_kelly(
        kelly_prob, kelly_ml,
        divisor=kp.get("divisor", 4),
        cap_pct=kp.get("cap_pct", 3.0),
        unit_size_pct=kp.get("unit_size_pct", 1.0),
    )
    kf["direction"] = direction

    return {
        "home_ml": home_ml,
        "away_ml": away_ml,
        "home_implied_pct": round(home_implied * 100, 1),
        "away_implied_pct": round(away_implied * 100, 1),
        "model_home_pct": round(model_win_pct * 100, 1),
        "model_away_pct": round((1 - model_win_pct) * 100, 1),
        "home_ev": home_ev,
        "away_ev": away_ev,
        "direction": direction,
        "prob_diff": round(prob_diff, 1),
        "kelly": round(best_kelly, 2),       # 既有 raw 欄位保留
        "kelly_fractional": kf,               # 新區塊
        "stars": stars,
    }


def analyze_over_under(
    line: float,
    predicted_total: float,
    over_odds_ml: int = None,
    under_odds_ml: int = None,
    kelly_params: dict = None,
) -> dict:
    """分析直線大小分盤口（無拆注）— 使用 run 差距制。

    O/U 幾乎都是 .5 整數線，忽略 push 處理。
    """
    diff = predicted_total - line
    stars = get_stars_ou(diff)

    if stars == 0:
        direction = "PASS"
    elif diff > 0:
        direction = "OVER"
    else:
        direction = "UNDER"

    # 機率：P(Over) = 1 - Φ(line; μ=predicted_total, σ=_MLB_TOTAL_STD)
    p_over = 1.0 - _normal_cdf(line, predicted_total, _MLB_TOTAL_STD)
    p_under = 1.0 - p_over

    # Kelly（若有 odds）
    kelly_fractional = None
    if over_odds_ml is not None or under_odds_ml is not None:
        kp = kelly_params or {}
        kelly_fractional = {"over": None, "under": None}
        if over_odds_ml is not None:
            kf = calc_fractional_kelly(
                p_over, over_odds_ml,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(over_odds_ml) + 1, 3)
            kelly_fractional["over"] = kf
        if under_odds_ml is not None:
            kf = calc_fractional_kelly(
                p_under, under_odds_ml,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(under_odds_ml) + 1, 3)
            kelly_fractional["under"] = kf

    return {
        "line": line,
        "predicted_total": round(predicted_total, 1),
        "diff": round(diff, 1),
        "direction": direction,
        "stars": stars,
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "kelly_fractional": kelly_fractional,
    }


def analyze_weighted_ou(
    base_line: int,
    split_pct: int,
    predicted_total: float,
    odds_hk: float,
) -> dict:
    """加權大小分分析（支援任意拆注比例）。

    原理：注金拆成兩份。
      · (100 - split_pct)% 押在基準線 base_line（整數）
      · split_pct%          押在高線 base_line + 0.5

    範例：
      '9-20' → base=9, split=20 → 80% 押 9 / 20% 押 9.5
      '8+50' → base=8, split=50 → 50% 押 8 / 50% 押 8.5（標準四分球）

    分數 = base_line 時：
      · 基準線那份：push（不輸不贏）
      · 高線那份：Over 輸 / Under 贏
    """
    high_w = split_pct / 100.0          # 押高線的比例
    effective_line = base_line + high_w * 0.5

    # 機率估算
    p_under_full = _p_at_most(base_line - 1, predicted_total)   # total <= base-1
    p_exact      = _p_exactly(base_line, predicted_total)        # total == base
    p_over_full  = _p_at_least(base_line + 1, predicted_total)  # total >= base+1

    # EV（每注 100 元的期望損益）
    # Over: 全贏(≥base+1) / 輸高線份(=base) / 全輸(≤base-1)
    ev_over  = (p_over_full * odds_hk) - (p_exact * high_w) - p_under_full
    # Under: 全贏(≤base-1) / 贏高線份(=base) / 全輸(≥base+1)
    ev_under = (p_under_full * odds_hk) + (p_exact * high_w * odds_hk) - p_over_full

    # 情境說明
    partial_loss   = round(100 * high_w, 1)
    partial_gain   = round(100 * high_w * odds_hk, 1)
    full_payout    = round(100 * odds_hk, 1)

    scenarios_over = {
        f"總分 >={base_line + 1}（全贏）":          f"+{full_payout}",
        f"總分 = {base_line}（輸 {split_pct}%）":   f"-{partial_loss}",
        f"總分 <={base_line - 1}（全輸）":          "-100",
    }
    scenarios_under = {
        f"總分 <={base_line - 1}（全贏）":          f"+{full_payout}",
        f"總分 = {base_line}（贏 {split_pct}%）":   f"+{partial_gain}",
        f"總分 >={base_line + 1}（全輸）":          "-100",
    }

    # G1: 方向 & 星級（使用 run 差距制）
    diff = predicted_total - effective_line
    stars = get_stars_ou(diff)
    if stars == 0:
        direction = "PASS"
    elif diff > 0:
        direction = "OVER"
    else:
        direction = "UNDER"

    return {
        "type": "weighted_ou",
        "base_line": base_line,
        "split_pct": split_pct,
        "effective_line": round(effective_line, 2),
        "odds_hk": odds_hk,
        "predicted_total": round(predicted_total, 1),
        "diff_vs_effective": round(diff, 2),
        "scenarios_over": scenarios_over,
        "scenarios_under": scenarios_under,
        "ev_over_pct":  round(ev_over  * 100, 2),
        "ev_under_pct": round(ev_under * 100, 2),
        "prob_breakdown": {
            f"P(<={base_line - 1})": round(p_under_full * 100, 1),
            f"P(={base_line})":      round(p_exact      * 100, 1),
            f"P(>={base_line + 1})": round(p_over_full  * 100, 1),
        },
        "direction": direction,
        "stars": stars,
    }


def analyze_run_line(
    predicted_margin: float,
    model_home_win_pct: float = None,
    home_ml: int = None,
    away_ml: int = None,
    home_rl_odds_ml: int = None,
    away_rl_odds_ml: int = None,
    home_point: float = None,       # Pinnacle snapshot 主隊 RL point（±1.5）
    kelly_params: dict = None,
) -> dict:
    """分析讓分盤（-1.5）。

    C2/C3 fix: 熱門方用市場 ML 判定（非 model margin）；side 標籤優先用 Pinnacle point。
    """
    if abs(predicted_margin) < 1.5:
        direction = "NEUTRAL"
        stars = 1
    elif predicted_margin >= 2.5:
        direction = "FAVORITE_COVER"
        stars = min(int(predicted_margin), 5)
    elif predicted_margin <= -2.5:
        direction = "UNDERDOG_COVER"
        stars = min(int(abs(predicted_margin)), 5)
    else:
        direction = "LEAN_FAVORITE" if predicted_margin > 0 else "LEAN_UNDERDOG"
        stars = 2

    # Kelly：需要 model_home_win_pct + 市場 ML（判熱門）+ RL odds
    kelly_fractional = None
    have_ml = home_ml is not None and away_ml is not None
    have_rl_odds = home_rl_odds_ml is not None or away_rl_odds_ml is not None
    if model_home_win_pct is not None and have_ml and have_rl_odds:
        # C2: 市場熱門方判定用 American ML 較負那方（不用 predicted_margin）
        fav_is_home = home_ml < away_ml
        fav_win_pct = model_home_win_pct if fav_is_home else (1 - model_home_win_pct)
        fav_ml      = home_ml if fav_is_home else away_ml
        fav_rl_odds = home_rl_odds_ml if fav_is_home else away_rl_odds_ml
        dog_rl_odds = away_rl_odds_ml if fav_is_home else home_rl_odds_ml

        p_cover_fav = fav_win_pct * p_margin_ge_2_given_win(fav_ml)
        p_cover_dog = 1 - p_cover_fav

        # C3: Side 標籤優先用 Pinnacle snapshot 的 point（source of truth）
        if home_point is not None:
            fav_side = "HOME_-1.5" if home_point < 0 else "AWAY_-1.5"
            dog_side = "AWAY_+1.5" if home_point < 0 else "HOME_+1.5"
        else:
            fav_side = "HOME_-1.5" if fav_is_home else "AWAY_-1.5"
            dog_side = "AWAY_+1.5" if fav_is_home else "HOME_+1.5"

        kp = kelly_params or {}
        kelly_fractional = {"favorite_cover": None, "underdog_cover": None}

        if fav_rl_odds is not None:
            kf = calc_fractional_kelly(
                p_cover_fav, fav_rl_odds,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(fav_rl_odds) + 1, 3)
            kf["side"] = fav_side
            kelly_fractional["favorite_cover"] = kf

        if dog_rl_odds is not None:
            kf = calc_fractional_kelly(
                p_cover_dog, dog_rl_odds,
                divisor=kp.get("divisor", 4),
                cap_pct=kp.get("cap_pct", 3.0),
                unit_size_pct=kp.get("unit_size_pct", 1.0),
            )
            kf["decimal_odds"] = round(american_to_hk(dog_rl_odds) + 1, 3)
            kf["side"] = dog_side
            kelly_fractional["underdog_cover"] = kf

    return {
        "predicted_margin": round(predicted_margin, 1),
        "direction": direction,
        "stars": stars,
        "kelly_fractional": kelly_fractional,
    }


def analyze_quarter_handicap(
    low_line: float,
    high_line: float,
    giving_side: str,
    odds_hk: float,
    predicted_home: float,
    predicted_away: float,
    split_pct: int = 50,
) -> dict:
    """亞洲讓分四分球分析（支援任意拆注比例）。

    split_pct：押在高線（high_line）的百分比，其餘押低線（low_line）。
    預設 50 = 標準四分球（50/50）。
    '1-20' → low=1.0, high=1.5, split_pct=20（80% 押 1.0 / 20% 押 1.5）
    '1-50' → low=1.0, high=1.5, split_pct=50（標準四分球）
    """
    if giving_side == "home":
        margin = predicted_home - predicted_away
    else:
        margin = predicted_away - predicted_home

    high_w = split_pct / 100.0
    payout_full_win  = round(100 * odds_hk, 1)
    payout_split_win = round(100 * high_w * odds_hk, 1)   # 贏半（高線份）
    partial_loss     = round(100 * high_w, 1)              # 輸半（高線份）

    scenarios_giving = {
        f"贏 {int(high_line) + 1}+ 分（全贏）":          f"+{payout_full_win}",
        f"贏 {int(high_line)} 分（輸 {split_pct}%）":    f"-{partial_loss}",
        "輸（全輸）":                                      "-100",
    }
    scenarios_receiving = {
        "贏（全贏）":                                      f"+{payout_full_win}",
        f"輸 {int(high_line)} 分（贏 {split_pct}%）":    f"+{payout_split_win}",
        f"輸 {int(high_line) + 1}+ 分（全輸）":          "-100",
    }

    if margin > high_line + 0.5:
        direction = "GIVING"
        stars = min(int((margin - high_line) * 3), 5)
    elif margin > high_line:
        direction = "GIVING"
        stars = 2
    elif margin > low_line:
        direction = "LEAN_GIVING"
        stars = 2
    elif margin > 0:
        direction = "LEAN_RECEIVING"
        stars = 2
    else:
        direction = "RECEIVING"
        stars = min(int(abs(margin) * 3) + 1, 5)

    return {
        "type": "quarter_handicap",
        "low_line": low_line,
        "high_line": high_line,
        "split_pct": split_pct,
        "giving_side": giving_side,
        "odds_hk": odds_hk,
        "predicted_margin": round(margin, 1),
        "scenarios_giving": scenarios_giving,
        "scenarios_receiving": scenarios_receiving,
        "direction": direction,
        "stars": stars,
    }


def consistency_check(over_under: dict, predicted_home: float, predicted_away: float) -> str:
    """一致性檢查（相容直線與加權格式）"""
    issues = []
    predicted_total = predicted_home + predicted_away
    # 直線用 "line"，加權格式用 "effective_line"
    line = over_under.get("line") if over_under.get("line") is not None else over_under.get("effective_line")

    if line is None:
        return "SKIP: 無法取得盤口線"
    if over_under["direction"] == "OVER" and predicted_total <= line:
        issues.append(f"CONFLICT: 推大分但預測總分 {predicted_total:.1f} <= 盤口 {line}")
    if over_under["direction"] == "UNDER" and predicted_total >= line:
        issues.append(f"CONFLICT: 推小分但預測總分 {predicted_total:.1f} >= 盤口 {line}")

    return "PASS" if not issues else "; ".join(issues)


def main():
    parser = argparse.ArgumentParser(description="Analyze betting odds")
    # ML（二擇一：American 或 HK）
    parser.add_argument("--home-ml", type=int, help="Home moneyline American (e.g. -150)")
    parser.add_argument("--away-ml", type=int, help="Away moneyline American (e.g. +130)")
    parser.add_argument("--hk-home", type=float, help="Home HK odds (e.g. 0.65)")
    parser.add_argument("--hk-away", type=float, help="Away HK odds (e.g. 1.24)")
    # O/U + 預測
    parser.add_argument("--total", type=float, help="Over/Under line")
    parser.add_argument("--model-win-pct", type=float, help="Model home win prob (0-1)")
    parser.add_argument("--predicted-home", type=float, help="Predicted home score")
    parser.add_argument("--predicted-away", type=float, help="Predicted away score")
    # 加權大小分（新格式，優先於 --total）
    parser.add_argument("--ou-format", type=str,
                        help="大小分格式，如 '9-20' 或 '8+50'（解析為 base line + split%%）")
    parser.add_argument("--ou-odds-hk", type=float,
                        help="大小分 HK 賠率（搭配 --ou-format 使用）")
    # Quarter Handicap（可選）
    parser.add_argument("--quarter-handicap", action="store_true", help="Enable Quarter Handicap analysis")
    parser.add_argument("--low-line", type=float, default=0.5, help="Lower handicap line (e.g. 0.5)")
    parser.add_argument("--high-line", type=float, default=1.0, help="Higher handicap line (e.g. 1.0)")
    parser.add_argument("--handicap-giving", choices=["home", "away"], default="home")
    parser.add_argument("--handicap-odds-hk", type=float, help="Quarter handicap HK odds")
    parser.add_argument("--handicap-split-pct", type=int, default=50,
                        help="讓分拆注百分比，預設 50（標準四分球）。如 '1-20' → 傳入 20")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "odds_analyzer test mode"}, indent=2))
        return

    # HK → American 轉換
    home_ml = args.home_ml
    away_ml = args.away_ml
    if args.hk_home is not None:
        home_ml = hk_to_american(args.hk_home)
    if args.hk_away is not None:
        away_ml = hk_to_american(args.hk_away)

    if home_ml is None or away_ml is None:
        parser.error("Must provide either --home-ml/--away-ml or --hk-home/--hk-away")
    if args.ou_format is None and args.total is None:
        parser.error("必須提供 --ou-format（如 '9-20'）或 --total（直線）其中一個")
    if args.model_win_pct is None or args.predicted_home is None or args.predicted_away is None:
        parser.error("--model-win-pct, --predicted-home, --predicted-away are required")

    predicted_total = args.predicted_home + args.predicted_away
    predicted_margin = args.predicted_home - args.predicted_away

    moneyline = analyze_moneyline(home_ml, away_ml, args.model_win_pct)
    run_line = analyze_run_line(predicted_margin)

    # 大小分：--ou-format 優先，否則用 --total（直線）
    if args.ou_format:
        if args.ou_odds_hk is None:
            parser.error("--ou-format 必須搭配 --ou-odds-hk 使用")
        ou_base, ou_split = parse_bet_format(args.ou_format)
        over_under = analyze_weighted_ou(ou_base, ou_split, predicted_total, args.ou_odds_hk)
    elif args.total is not None:
        over_under = analyze_over_under(args.total, predicted_total)
    else:
        parser.error("必須提供 --ou-format 或 --total")

    check = consistency_check(over_under, args.predicted_home, args.predicted_away)

    result = {
        "moneyline": moneyline,
        "over_under": over_under,
        "run_line": run_line,
        "consistency_check": check,
    }

    # Quarter Handicap（可選）
    if args.quarter_handicap and args.handicap_odds_hk is not None:
        result["quarter_handicap"] = analyze_quarter_handicap(
            low_line=args.low_line,
            high_line=args.high_line,
            giving_side=args.handicap_giving,
            odds_hk=args.handicap_odds_hk,
            predicted_home=args.predicted_home,
            predicted_away=args.predicted_away,
            split_pct=args.handicap_split_pct,
        )

    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
