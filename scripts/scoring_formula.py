#!/usr/bin/env python3
"""比分公式 — Path B baseline，簡單期望得分（formula-as-guardrail）。

被 summary_renderer 用來產出 `## 修正後預期得分` 段的 base 列。
複雜建模刻意外置給 AI 在 summary `+ 信號` 欄做 magnitude judgment
（見 reference/matchup-factors.md §量級錨點）。
"""


def predict_with_formula(data: dict) -> dict:
    """Path B baseline：簡單期望得分公式，當 AI judgment 的 sanity-check rail。

    E[R] = 聯盟平均得分 × (打線 xwOBA / 聯盟 xwOBA) × (對方投手 FIP / 聯盟 ERA) × (PF / 100)

    聯盟平均（2024-2025 基準）：R/G ≈ 4.5, xwOBA ≈ 0.315, ERA ≈ 4.20

    Empirical calibration (2026-05 backtest n=114, two methodologies agreed):
    - HOME field advantage: skill HOME dir hit 61.8% vs AWAY 51.7% (gap 10pp)
      → add +0.3 run to HOME side
    - Total over-estimate: skill avg bias +0.96 run vs actual
      → subtract 1.0 from total (split −0.5 each side)
    Net effect per side: HOME = raw + 0.3 − 0.5 = raw − 0.2 ; AWAY = raw − 0.5
    """
    LEAGUE_RPG = 4.5
    LEAGUE_XWOBA = 0.315
    LEAGUE_ERA = 4.20

    HOME_FIELD_ADVANTAGE = 0.3
    TOTAL_SIDE_DOWNSHIFT = 0.5  # half of −1.0 total-bias correction

    pf = data.get("park_factor", 100)
    home_xwoba = data.get("home_batting_xwoba", LEAGUE_XWOBA)
    away_xwoba = data.get("away_batting_xwoba", LEAGUE_XWOBA)
    away_pitcher_fip = data.get("away_starter_fip", LEAGUE_ERA)
    home_pitcher_fip = data.get("home_starter_fip", LEAGUE_ERA)

    pf_mult = pf / 100
    home_score_raw = LEAGUE_RPG * (home_xwoba / LEAGUE_XWOBA) * (away_pitcher_fip / LEAGUE_ERA) * pf_mult
    away_score_raw = LEAGUE_RPG * (away_xwoba / LEAGUE_XWOBA) * (home_pitcher_fip / LEAGUE_ERA) * pf_mult

    home_score = round(home_score_raw + HOME_FIELD_ADVANTAGE - TOTAL_SIDE_DOWNSHIFT, 1)
    away_score = round(away_score_raw - TOTAL_SIDE_DOWNSHIFT, 1)

    return {
        "home_score": home_score,
        "away_score": away_score,
        "total": round(home_score + away_score, 1),
    }
