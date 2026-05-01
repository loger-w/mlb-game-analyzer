#!/usr/bin/env python3
"""比分公式 — Log5 勝率 + Pythagenport 期望得分。

被 phase3_summary_renderer 用來產出 `## 修正後預期得分` 段的 base 列。
"""

import math


def log5(home_pct: float, away_pct: float) -> float:
    """Log5 勝率公式"""
    p = (home_pct * (1 - away_pct)) / (home_pct * (1 - away_pct) + away_pct * (1 - home_pct))
    return p


def pythagorean_runs(rs: float, ra: float, g: float = 10) -> float:
    """Pythagenport 動態指數公式（Smyth & Patriot, 2003）

    exponent = 1.50 × log10[(RS + RA) / G] + 0.45
    Pythagenport RMSE = 3.991 勝（優於固定指數 1.83 的 4.126）
    """
    if rs + ra == 0:
        return 0.5
    exponent = 1.50 * math.log10((rs + ra) / g) + 0.45
    return (rs ** exponent) / (rs ** exponent + ra ** exponent)


def predict_with_formula(data: dict) -> dict:
    """Log5 + 期望得分公式（納入對方投手壓制力）。

    E[R] = 聯盟平均得分 × (打線 xwOBA / 聯盟 xwOBA) × (對方投手 ERA / 聯盟 ERA) × (PF / 100)
    聯盟平均（2024-2025 基準）：R/G ≈ 4.5, xwOBA ≈ 0.315, ERA ≈ 4.20
    """
    LEAGUE_RPG = 4.5
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA = 4.20

    home_rs = data.get("home_recent_rs", 4.5)
    home_ra = data.get("home_recent_ra", 4.5)
    away_rs = data.get("away_recent_rs", 4.5)
    away_ra = data.get("away_recent_ra", 4.5)
    pf = data.get("park_factor", 100)

    home_pct = pythagorean_runs(home_rs, home_ra)
    away_pct = pythagorean_runs(away_rs, away_ra)
    log5_pct = log5(home_pct, away_pct)
    log5_pct = min(log5_pct + 0.03, 0.95)  # 主場優勢 +3%

    home_xwoba = data.get("home_batting_xwoba", LEAGUE_XWOBA)
    away_xwoba = data.get("away_batting_xwoba", LEAGUE_XWOBA)
    away_pitcher_fip = data.get("away_starter_fip", LEAGUE_ERA)
    home_pitcher_fip = data.get("home_starter_fip", LEAGUE_ERA)

    pf_mult = pf / 100
    home_score = round(
        LEAGUE_RPG * (home_xwoba / LEAGUE_XWOBA) * (away_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )
    away_score = round(
        LEAGUE_RPG * (away_xwoba / LEAGUE_XWOBA) * (home_pitcher_fip / LEAGUE_ERA) * pf_mult, 1
    )

    return {
        "log5_pct": round(log5_pct * 100, 1),
        "pythag_home_pct": round(home_pct * 100, 1),
        "pythag_away_pct": round(away_pct * 100, 1),
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }
