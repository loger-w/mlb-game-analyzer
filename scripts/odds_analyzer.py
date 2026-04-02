#!/usr/bin/env python3
"""MLB Odds Analyzer — 盤口價值計算（EV + Kelly Criterion）"""

import argparse
import json


def ml_to_implied_prob(ml: int) -> float:
    """Moneyline → 隱含勝率"""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


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


def get_stars(prob_diff: float) -> int:
    """根據勝率差距決定推薦星級"""
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


def analyze_moneyline(home_ml: int, away_ml: int, model_win_pct: float) -> dict:
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
    else:
        direction = "AWAY"
        best_ev = away_ev
        best_kelly = away_kelly
        prob_diff = ((1 - model_win_pct) - away_implied) * 100

    stars = get_stars(prob_diff)

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
        "kelly": round(best_kelly, 2),
        "stars": stars,
    }


def analyze_over_under(line: float, predicted_total: float) -> dict:
    """分析大小分盤口"""
    diff = predicted_total - line
    abs_diff = abs(diff)

    if abs_diff < 0.5:
        direction = "NEUTRAL"
        stars = 1
    elif diff > 0:
        direction = "OVER"
        stars = get_stars(abs_diff * 5)
    else:
        direction = "UNDER"
        stars = get_stars(abs_diff * 5)

    return {
        "line": line,
        "predicted_total": round(predicted_total, 1),
        "diff": round(diff, 1),
        "direction": direction,
        "stars": min(stars, 5),
    }


def analyze_run_line(predicted_margin: float) -> dict:
    """分析讓分盤（-1.5）"""
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

    return {
        "predicted_margin": round(predicted_margin, 1),
        "direction": direction,
        "stars": stars,
    }


def consistency_check(over_under: dict, predicted_home: float, predicted_away: float) -> str:
    """一致性檢查"""
    issues = []
    predicted_total = predicted_home + predicted_away

    if over_under["direction"] == "OVER" and predicted_total <= over_under["line"]:
        issues.append(f"CONFLICT: 推大分但預測總分 {predicted_total:.1f} <= 盤口 {over_under['line']}")
    if over_under["direction"] == "UNDER" and predicted_total >= over_under["line"]:
        issues.append(f"CONFLICT: 推小分但預測總分 {predicted_total:.1f} >= 盤口 {over_under['line']}")

    return "PASS" if not issues else "; ".join(issues)


def main():
    parser = argparse.ArgumentParser(description="Analyze betting odds")
    parser.add_argument("--home-ml", type=int, required=True, help="Home moneyline (e.g. -150)")
    parser.add_argument("--away-ml", type=int, required=True, help="Away moneyline (e.g. +130)")
    parser.add_argument("--total", type=float, required=True, help="Over/Under line (e.g. 8.5)")
    parser.add_argument("--model-win-pct", type=float, required=True, help="Model home win probability (0-1)")
    parser.add_argument("--predicted-home", type=float, required=True, help="Predicted home score")
    parser.add_argument("--predicted-away", type=float, required=True, help="Predicted away score")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print(json.dumps({"test": "OK", "message": "odds_analyzer test mode"}, indent=2))
        return

    predicted_total = args.predicted_home + args.predicted_away
    predicted_margin = args.predicted_home - args.predicted_away

    moneyline = analyze_moneyline(args.home_ml, args.away_ml, args.model_win_pct)
    over_under = analyze_over_under(args.total, predicted_total)
    run_line = analyze_run_line(predicted_margin)
    check = consistency_check(over_under, args.predicted_home, args.predicted_away)

    result = {
        "moneyline": moneyline,
        "over_under": over_under,
        "run_line": run_line,
        "consistency_check": check,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
